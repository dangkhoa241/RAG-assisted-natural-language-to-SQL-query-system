"""Retrieval-augmented text-to-SQL: LLM (Groq or Cerebras) + schema context + retrieved question->SQL examples.

Modes:
  llm_zero_shot - the prompt contains the schema only
  llm_rag       - schema + up to k similar examples from the example bank
                  (with k=0 the prompt is identical to llm_zero_shot)
  llm_doc_rag   - schema + the top-k business-glossary chunks for the dataset (doc_retrieval.py),
                  or an explicit list of chunks via `docs` (used for the oracle upper bound)

Example retrieval (example_style="fixed", the default):
  - each example is shown with its own table name and schema line, never the table name `data`
  - examples are filtered to the predicted intent (falls back to all intents if none match)
  - examples scoring below EXAMPLE_MIN_SCORE are dropped, so a question with no close example gets none
example_style="stage2" reproduces the Stage 2 prompts (every example used `data`, top-k by similarity only).

Providers: "groq" (the default, used by the app) and "cerebras" (OpenAI-compatible API, used for the Stage 3
gpt-oss-120b runs). Model IDs are the provider's own: "openai/gpt-oss-120b" on Groq, "gpt-oss-120b" on Cerebras.

If the LLM call fails or its SQL fails the safety check, generation falls back to the
rule-based generator in sql_builder.py.
"""
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_context import Dataset
from sql_safety import UnsafeSQLError, validate_sql

ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Configuration ---------------------------------------------------------
GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_PROVIDER = "groq"
PROVIDER_API_KEYS = {"groq": "GROQ_API_KEY", "cerebras": "CEREBRAS_API_KEY"}   # .env variable per provider
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
# Minimum seconds between request starts, per provider (the Cerebras account allows 5 requests/min).
MIN_REQUEST_INTERVAL_S = {"cerebras": 12.5}
LLM_REASONING_EFFORT = "low"   # gpt-oss reasoning budget; "low" keeps latency and token use down
LLM_MAX_COMPLETION_TOKENS = 1024
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXAMPLE_BANK_PATH = ROOT_DIR / "eval" / "sql_benchmark" / "example_bank.jsonl"
INDEX_CACHE_DIR = ROOT_DIR / ".rag_cache"
DEFAULT_K = 3
# Cosine similarity below which a bank example is not shown. Calibrated on the bank itself (leave-one-out
# against the other schemas), not on a test set: at 0.35, 71% of bank questions still get >= 1 example.
EXAMPLE_MIN_SCORE = 0.35
MIN_INTENT_MATCHES = 1   # use intent-filtered examples if at least this many clear the threshold
DOC_K = 3
DOC_RETRIEVER = "hybrid"
MAX_CATEGORY_VALUES_IN_PROMPT = 20
MAX_LLM_RETRIES = 8

SYSTEM_PROMPT = """You translate questions into SQLite queries over a single table named `data`.
Rules:
- Return exactly one SQLite SELECT statement and nothing else: no explanation, no markdown.
- Use only columns listed in the schema, and wrap every column name in backticks.
- Match categorical values exactly as they are listed in the schema (comparisons are case-sensitive).
- Dates are stored as 'YYYY-MM-DD' text. Bucket months with strftime('%Y-%m', col) and years with strftime('%Y', col).
- When the question asks to show or list records, select all columns with SELECT *.
- For comparisons between groups, return one row per compared group with the requested metric.
- For trends, return one row per time period with the metric, ordered by period.
- Only add ORDER BY or LIMIT when the question asks for a ranking or a top N."""


# --- Schema context --------------------------------------------------------
def _quote(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def build_schema_context(dataset: Dataset) -> str:
    """Column names and types, plus distinct values for categorical columns and
    min/max for numeric and date columns."""
    df = dataset.df
    lines = [f"Table `data` ({len(df)} rows). Columns:"]
    for col in df.columns:
        s = df[col].dropna()
        if col in dataset.date_cols:
            lines.append(f"- `{col}` DATE stored as 'YYYY-MM-DD' text, min {_quote(s.min())}, max {_quote(s.max())}")
        elif col in dataset.numeric_cols:
            kind = "INTEGER" if pd.api.types.is_integer_dtype(df[col]) else "REAL"
            lo, hi = s.min(), s.max()
            lines.append(f"- `{col}` {kind}, min {lo:g}, max {hi:g}")
        elif col in dataset.categorical_cols:
            values = s.astype(str).value_counts().index.tolist()
            shown = ", ".join(_quote(v) for v in values[:MAX_CATEGORY_VALUES_IN_PROMPT])
            more = f" (+{len(values) - MAX_CATEGORY_VALUES_IN_PROMPT} more)" if len(values) > MAX_CATEGORY_VALUES_IN_PROMPT else ""
            lines.append(f"- `{col}` TEXT, categorical, values: {shown}{more}")
        else:
            examples = ", ".join(_quote(v) for v in s.astype(str).head(2))
            lines.append(f"- `{col}` TEXT, free text, e.g. {examples}")
        n_null = int(df[col].isna().sum())
        if n_null:  # only columns with empty cells get this note, so fully populated schemas are unchanged
            lines[-1] += f"; NULL in {n_null} rows"
    return "\n".join(lines)


# --- Retrieval (sentence-transformers + FAISS) -----------------------------
class ExampleRetriever:
    """Exact cosine-similarity search over the example bank. The FAISS index is cached
    on disk, keyed by the bank's content and the embedding model, so it only rebuilds when either changes."""

    def __init__(self, bank_path: Path = EXAMPLE_BANK_PATH, model_name: str = EMBEDDING_MODEL):
        import faiss
        from sentence_transformers import SentenceTransformer

        raw = Path(bank_path).read_bytes()
        self.examples = [_with_own_table(json.loads(line)) for line in raw.decode("utf-8").splitlines() if line.strip()]
        self.model = SentenceTransformer(model_name, device="cpu")

        key = hashlib.sha256(raw + model_name.encode()).hexdigest()[:16]
        index_path = INDEX_CACHE_DIR / f"examples_{key}.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            vectors = self.embed([e["question"] for e in self.examples])
            self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on unit vectors = cosine
            self.index.add(vectors)
            INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(index_path))

    def embed(self, texts) -> np.ndarray:
        return self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    def search(self, question: str, k: int, intent: str = None, min_score: float = None):
        """Up to k examples, most similar first. With `min_score`, weaker matches are dropped; with
        `intent`, only examples of that intent are kept unless fewer than MIN_INTENT_MATCHES survive."""
        if k <= 0:
            return []
        if intent is None and min_score is None:
            scores, ids = self.index.search(self.embed([question]), k)
            return [dict(self.examples[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
        scores, ids = self.index.search(self.embed([question]), len(self.examples))
        ranked = [dict(self.examples[i], score=float(s)) for s, i in zip(scores[0], ids[0])
                  if i >= 0 and (min_score is None or s >= min_score)]
        if intent is not None:
            same = [e for e in ranked if e["intent"] == intent]
            if len(same) >= MIN_INTENT_MATCHES:
                ranked = same
        return ranked[:k]


def _with_own_table(example: dict) -> dict:
    """Add `table`, `table_schema` and `table_sql`: the example rewritten to use its own table name
    (e.g. hr_employees) instead of `data`, so the LLM can't mistake it for the user's table."""
    table = example["schema_name"]
    columns = example["schema"].split(": data(", 1)[1][:-1]
    return dict(example, table=table,
                table_schema=f"Table `{table}` columns: {columns}",
                table_sql=re.sub(r"\bFROM data\b", f"FROM {table}", example["sql"]))


_retriever = None
_retriever_lock = threading.Lock()


def get_retriever() -> ExampleRetriever:
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            _retriever = ExampleRetriever()
    return _retriever


def retrieve_examples(question: str, k: int = DEFAULT_K, intent: str = None, example_style: str = "fixed"):
    """Up to k example-bank pairs similar to `question` (each with a `score`). The "fixed" style filters
    by `intent` and applies EXAMPLE_MIN_SCORE; "stage2" is plain top-k."""
    if example_style == "stage2":
        return get_retriever().search(question, k)
    return get_retriever().search(question, k, intent=intent, min_score=EXAMPLE_MIN_SCORE)


def retrieve_docs(question: str, dataset_name: str, k: int = DOC_K, method: str = DOC_RETRIEVER):
    """The top-k glossary chunks for `question` from docs/glossary/<dataset_name>.md."""
    from doc_retrieval import get_glossary_retriever

    return get_glossary_retriever(dataset_name).search(question, k, method)


# --- Prompt + LLM call -----------------------------------------------------
def build_messages(question: str, schema_context: str, examples, docs=None, example_style: str = "fixed") -> list:
    parts = [f"Schema:\n{schema_context}"]
    if docs:
        defs = "\n".join(f"- {d['term']}: {d['definition']}" for d in docs)
        parts.append("Business definitions (when the question uses one of these terms, apply its definition "
                     f"exactly; ignore definitions the question doesn't use):\n{defs}")
    if examples and example_style == "stage2":
        shots = "\n\n".join(f"Question: {e['question']}\nSQL: {e['sql']}" for e in examples)
        parts.append(f"Examples from other tables (same conventions):\n{shots}")
    elif examples:
        shots = "\n\n".join(f"{e['table_schema']}\nQuestion: {e['question']}\nSQL: {e['table_sql']}"
                              for e in examples)
        parts.append("Examples written for OTHER tables. Copy their SQL patterns, but query only the table "
                     f"`data` and its columns:\n{shots}")
    parts.append(f"Question: {question}\nSQL:")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


class LLMError(RuntimeError):
    """The LLM call failed (after retries)."""


class DailyLimitError(LLMError):
    """The provider's daily request/token quota is exhausted; retrying right away won't help.
    `retry_after_s` is the wait the provider suggests ("Please try again in 7m12.5s"), if it gave one."""

    def __init__(self, message: str):
        super().__init__(message)
        m = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", message)
        self.retry_after_s = (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3) or 0)
                              if m and any(m.groups()) else None)


class LLMCache:
    """Append-only JSONL cache of LLM responses, keyed by provider + model + prompt hash."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    self.entries[e["key"]] = e

    @staticmethod
    def make_key(model: str, messages: list, params: dict, provider: str = DEFAULT_PROVIDER) -> str:
        payload = {"model": model, "messages": messages, "params": params}
        if provider != DEFAULT_PROVIDER:  # Groq keys predate providers; leaving them unchanged keeps old caches valid
            payload["provider"] = provider
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def get(self, key):
        e = self.entries.get(key)
        return None if e is None else e["response"]

    def put(self, key, model, response, usage=None, latency_s=None, provider=None, time_info=None):
        e = {"key": key, "model": model, "response": response, "usage": usage}
        if provider is not None:
            e["provider"] = provider
        if latency_s is not None:
            e["latency_s"] = latency_s  # wall time of the successful request (excludes rate-limit waits)
        if time_info is not None:
            e["time_info"] = time_info  # Cerebras's server-side timings (Groq reports them inside `usage`)
        self.entries[key] = e
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(e) + "\n")


_clients = {}
_last_request_start = {}


def _client_for(provider: str):
    """The SDK client for `provider`. SDK retries are off; call_llm retries with backoff."""
    if provider not in _clients:
        from dotenv import load_dotenv

        load_dotenv(ROOT_DIR / ".env")
        if provider == "groq":
            from groq import Groq

            if not os.environ.get("GROQ_API_KEY"):
                raise LLMError("GROQ_API_KEY is not set (add it to .env)")
            _clients[provider] = Groq(max_retries=0)
        elif provider == "cerebras":
            from openai import OpenAI

            if not os.environ.get("CEREBRAS_API_KEY"):
                raise LLMError("CEREBRAS_API_KEY is not set (add it to .env)")
            _clients[provider] = OpenAI(base_url=CEREBRAS_BASE_URL, api_key=os.environ["CEREBRAS_API_KEY"],
                                        max_retries=0)
        else:
            raise ValueError(f"unknown provider: {provider}")
    return _clients[provider]


def _sdk(provider: str):
    """The provider's SDK module (groq and openai have the same exception classes)."""
    if provider == "cerebras":
        import openai
        return openai
    import groq
    return groq


def _throttle(provider: str):
    """Sleep so request starts to `provider` are at least MIN_REQUEST_INTERVAL_S apart (no bursts)."""
    gap = MIN_REQUEST_INTERVAL_S.get(provider)
    if gap:
        wait = _last_request_start.get(provider, float("-inf")) + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
    _last_request_start[provider] = time.monotonic()


def _is_daily_limit(e) -> bool:
    msg = str(e).lower()
    if any(s in msg for s in ("per day", "(tpd)", "(rpd)", "daily", "tokens_per_day", "requests_per_day")):
        return True
    headers = e.response.headers if getattr(e, "response", None) is not None else {}
    return any(headers.get(h) == "0" for h in ("x-ratelimit-remaining-requests-day", "x-ratelimit-remaining-tokens-day"))


def llm_params(model: str = GROQ_MODEL) -> dict:
    params = {"temperature": 0, "max_completion_tokens": LLM_MAX_COMPLETION_TOKENS}
    if "gpt-oss" in model:
        params["reasoning_effort"] = LLM_REASONING_EFFORT
    return params


def cache_key(messages: list, model: str = GROQ_MODEL, provider: str = DEFAULT_PROVIDER) -> str:
    return LLMCache.make_key(model, messages, llm_params(model), provider)


def call_llm(messages: list, cache: LLMCache = None, model: str = GROQ_MODEL, provider: str = DEFAULT_PROVIDER) -> str:
    """One chat completion at temperature 0, with exponential backoff on rate limits."""
    params = llm_params(model)
    key = LLMCache.make_key(model, messages, params, provider)
    if cache is not None and (hit := cache.get(key)) is not None:
        return hit

    client = _client_for(provider)
    sdk = _sdk(provider)
    for attempt in range(MAX_LLM_RETRIES):
        try:
            _throttle(provider)
            started = time.perf_counter()
            resp = client.chat.completions.create(model=model, messages=messages, **params)
            latency = time.perf_counter() - started
            text = resp.choices[0].message.content or ""
            if cache is not None:
                time_info = (resp.model_extra or {}).get("time_info") if provider != "groq" else None
                cache.put(key, model, text, resp.usage.model_dump() if resp.usage else None, round(latency, 3),
                          provider=provider, time_info=time_info)
            return text
        except sdk.RateLimitError as e:
            if _is_daily_limit(e):
                raise DailyLimitError(str(e)) from e
            retry_after = e.response.headers.get("retry-after") if e.response is not None else None
            wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
            wait = max(wait, MIN_REQUEST_INTERVAL_S.get(provider, 0))
            print(f"  429 from {provider} (attempt {attempt + 1}), backing off {wait:.0f}s: {str(e)[:300]}", flush=True)
        except (sdk.APIConnectionError, sdk.InternalServerError) as e:
            wait = min(2 ** attempt, 60)
            if attempt == MAX_LLM_RETRIES - 1:
                raise LLMError(str(e)) from e
        except sdk.APIError as e:
            raise LLMError(str(e)) from e
        time.sleep(wait + 0.25 * attempt)
    raise LLMError(f"gave up after {MAX_LLM_RETRIES} rate-limited attempts")


def extract_sql(text: str) -> str:
    """Pull the SQL out of an LLM reply (strips markdown fences and a leading 'SQL:')."""
    fenced = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql = fenced.group(1) if fenced else text
    sql = re.sub(r"^\s*SQL:\s*", "", sql, flags=re.IGNORECASE)
    return sql.strip()


# --- Generation ------------------------------------------------------------
@dataclass
class SQLGeneration:
    sql: str
    source: str                  # "llm" or "fallback"
    mode: str
    llm_sql: str = None          # what the LLM proposed, even if rejected
    error: str = None            # why we fell back, if we did
    rejected: bool = False       # the LLM's SQL failed the safety check
    examples: list = field(default_factory=list)
    docs: list = field(default_factory=list)
    cache_key: str = None        # key of the LLM call in LLMCache (for usage/latency lookups)


def _rule_based_sql(question: str, dataset: Dataset, intent: str = None) -> str:
    from sql_builder import build_sql

    if intent is None:
        from intent import guess_intent_by_keywords
        intent = guess_intent_by_keywords(question)
    return build_sql(question, intent, dataset)[1]


def generate_sql_detailed(question: str, dataset: Dataset, mode: str = "llm_rag", k: int = DEFAULT_K,
                          intent: str = None, cache: LLMCache = None, raise_on_quota: bool = False,
                          schema_context: str = None, example_style: str = "fixed", dataset_name: str = None,
                          docs: list = None, doc_method: str = DOC_RETRIEVER,
                          model: str = GROQ_MODEL, provider: str = DEFAULT_PROVIDER) -> SQLGeneration:
    """`intent` is the predicted intent: it filters examples (llm_rag) and drives the rule-based fallback.
    llm_doc_rag retrieves k glossary chunks for `dataset_name`, unless `docs` gives the chunks directly."""
    if mode not in ("llm_zero_shot", "llm_rag", "llm_doc_rag"):
        raise ValueError(f"unknown mode: {mode}")
    examples = retrieve_examples(question, k, intent, example_style) if mode == "llm_rag" else []
    if mode == "llm_doc_rag" and docs is None:
        docs = retrieve_docs(question, dataset_name, k, doc_method)
    docs = docs or []
    schema_context = schema_context or build_schema_context(dataset)
    messages = build_messages(question, schema_context, examples, docs, example_style)
    key = cache_key(messages, model, provider)

    try:
        llm_sql = extract_sql(call_llm(messages, cache=cache, model=model, provider=provider))
    except DailyLimitError:
        if raise_on_quota:
            raise
        return SQLGeneration(_rule_based_sql(question, dataset, intent), "fallback", mode,
                             error="LLM daily quota exhausted", examples=examples, docs=docs, cache_key=key)
    except LLMError as e:
        return SQLGeneration(_rule_based_sql(question, dataset, intent), "fallback", mode,
                             error=f"LLM call failed: {e}", examples=examples, docs=docs, cache_key=key)

    try:
        safe_sql = validate_sql(llm_sql)
    except UnsafeSQLError as e:
        return SQLGeneration(_rule_based_sql(question, dataset, intent), "fallback", mode, llm_sql=llm_sql,
                             error=f"rejected by safety check: {e}", rejected=True, examples=examples, docs=docs, cache_key=key)
    return SQLGeneration(safe_sql, "llm", mode, llm_sql=llm_sql, examples=examples, docs=docs, cache_key=key)


def generate_sql(question: str, dataset: Dataset, mode: str = "llm_rag", **kwargs) -> str:
    """Return only the SQL for `question` (falling back to the rule-based generator if needed)."""
    return generate_sql_detailed(question, dataset, mode, **kwargs).sql
