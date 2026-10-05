"""Retrieval-augmented text-to-SQL: Groq LLM + schema context + retrieved question->SQL examples.

Modes:
  llm_zero_shot - the prompt contains the schema only
  llm_rag       - schema + the k most similar examples from the example bank
                  (with k=0 the prompt is identical to llm_zero_shot)

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
LLM_REASONING_EFFORT = "low"   # gpt-oss reasoning budget; "low" keeps latency and token use down
LLM_MAX_COMPLETION_TOKENS = 1024
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXAMPLE_BANK_PATH = ROOT_DIR / "eval" / "sql_benchmark" / "example_bank.jsonl"
INDEX_CACHE_DIR = ROOT_DIR / ".rag_cache"
DEFAULT_K = 3
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
    return "\n".join(lines)


# --- Retrieval (sentence-transformers + FAISS) -----------------------------
class ExampleRetriever:
    """Exact cosine-similarity search over the example bank. The FAISS index is cached
    on disk, keyed by the bank's content and the embedding model, so it only rebuilds when either changes."""

    def __init__(self, bank_path: Path = EXAMPLE_BANK_PATH, model_name: str = EMBEDDING_MODEL):
        import faiss
        from sentence_transformers import SentenceTransformer

        raw = Path(bank_path).read_bytes()
        self.examples = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
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

    def search(self, question: str, k: int):
        if k <= 0:
            return []
        scores, ids = self.index.search(self.embed([question]), k)
        return [dict(self.examples[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]


_retriever = None
_retriever_lock = threading.Lock()


def get_retriever() -> ExampleRetriever:
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            _retriever = ExampleRetriever()
    return _retriever


def retrieve_examples(question: str, k: int = DEFAULT_K):
    """The top-k example-bank pairs most similar to `question` (each with a `score`)."""
    return get_retriever().search(question, k)


# --- Prompt + LLM call -----------------------------------------------------
def build_messages(question: str, schema_context: str, examples) -> list:
    parts = [f"Schema:\n{schema_context}"]
    if examples:
        shots = "\n\n".join(f"Question: {e['question']}\nSQL: {e['sql']}" for e in examples)
        parts.append(f"Examples from other tables (same conventions):\n{shots}")
    parts.append(f"Question: {question}\nSQL:")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


class LLMError(RuntimeError):
    """The LLM call failed (after retries)."""


class DailyLimitError(LLMError):
    """The provider's daily request/token quota is exhausted; retrying today won't help."""


class LLMCache:
    """Append-only JSONL cache of LLM responses, keyed by model + prompt hash."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    self.entries[e["key"]] = e

    @staticmethod
    def make_key(model: str, messages: list, params: dict) -> str:
        payload = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key):
        e = self.entries.get(key)
        return None if e is None else e["response"]

    def put(self, key, model, response, usage=None):
        e = {"key": key, "model": model, "response": response, "usage": usage}
        self.entries[key] = e
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(e) + "\n")


_client = None


def _groq_client():
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from groq import Groq

        load_dotenv(ROOT_DIR / ".env")
        if not os.environ.get("GROQ_API_KEY"):
            raise LLMError("GROQ_API_KEY is not set (add it to .env)")
        _client = Groq(max_retries=0)  # retries are handled below, with backoff
    return _client


def call_llm(messages: list, cache: LLMCache = None, model: str = GROQ_MODEL) -> str:
    """One chat completion at temperature 0, with exponential backoff on rate limits."""
    import groq

    params = {"temperature": 0, "reasoning_effort": LLM_REASONING_EFFORT,
              "max_completion_tokens": LLM_MAX_COMPLETION_TOKENS}
    key = LLMCache.make_key(model, messages, params)
    if cache is not None and (hit := cache.get(key)) is not None:
        return hit

    client = _groq_client()
    for attempt in range(MAX_LLM_RETRIES):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, **params)
            text = resp.choices[0].message.content or ""
            if cache is not None:
                cache.put(key, model, text, resp.usage.model_dump() if resp.usage else None)
            return text
        except groq.RateLimitError as e:
            msg = str(e).lower()
            if "per day" in msg or "(tpd)" in msg or "(rpd)" in msg:
                raise DailyLimitError(str(e)) from e
            retry_after = e.response.headers.get("retry-after") if e.response is not None else None
            wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
        except (groq.APIConnectionError, groq.InternalServerError) as e:
            wait = min(2 ** attempt, 60)
            if attempt == MAX_LLM_RETRIES - 1:
                raise LLMError(str(e)) from e
        except groq.APIError as e:
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


def _rule_based_sql(question: str, dataset: Dataset, intent: str = None) -> str:
    from sql_builder import build_sql

    if intent is None:
        from intent import guess_intent_by_keywords
        intent = guess_intent_by_keywords(question)
    return build_sql(question, intent, dataset)[1]


def generate_sql_detailed(question: str, dataset: Dataset, mode: str = "llm_rag", k: int = DEFAULT_K,
                          intent: str = None, cache: LLMCache = None, raise_on_quota: bool = False,
                          schema_context: str = None) -> SQLGeneration:
    if mode not in ("llm_zero_shot", "llm_rag"):
        raise ValueError(f"unknown mode: {mode}")
    examples = retrieve_examples(question, k) if mode == "llm_rag" else []
    schema_context = schema_context or build_schema_context(dataset)
    messages = build_messages(question, schema_context, examples)

    try:
        llm_sql = extract_sql(call_llm(messages, cache=cache))
    except DailyLimitError:
        if raise_on_quota:
            raise
        return SQLGeneration(_rule_based_sql(question, dataset, intent), "fallback", mode,
                             error="LLM daily quota exhausted", examples=examples)
    except LLMError as e:
        return SQLGeneration(_rule_based_sql(question, dataset, intent), "fallback", mode,
                             error=f"LLM call failed: {e}", examples=examples)

    try:
        safe_sql = validate_sql(llm_sql)
    except UnsafeSQLError as e:
        return SQLGeneration(_rule_based_sql(question, dataset, intent), "fallback", mode, llm_sql=llm_sql,
                             error=f"rejected by safety check: {e}", rejected=True, examples=examples)
    return SQLGeneration(safe_sql, "llm", mode, llm_sql=llm_sql, examples=examples)


def generate_sql(question: str, dataset: Dataset, mode: str = "llm_rag", **kwargs) -> str:
    """Return only the SQL for `question` (falling back to the rule-based generator if needed)."""
    return generate_sql_detailed(question, dataset, mode, **kwargs).sql
