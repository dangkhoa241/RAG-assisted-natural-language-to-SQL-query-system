"""Retrieval-augmented text-to-SQL: Groq LLM + schema context + retrieved question->SQL examples.

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
    docs: list = field(default_factory=list)


def _rule_based_sql(question: str, dataset: Dataset, intent: str = None) -> str:
    from sql_builder import build_sql

    if intent is None:
        from intent import guess_intent_by_keywords
        intent = guess_intent_by_keywords(question)
    return build_sql(question, intent, dataset)[1]


def generate_sql_detailed(question: str, dataset: Dataset, mode: str = "llm_rag", k: int = DEFAULT_K,
                          intent: str = None, cache: LLMCache = None, raise_on_quota: bool = False,
                          schema_context: str = None, example_style: str = "fixed", dataset_name: str = None,
                          docs: list = None, doc_method: str = DOC_RETRIEVER) -> SQLGeneration:
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

    try:
        llm_sql = extract_sql(call_llm(messages, cache=cache))
    except DailyLimitError:
        if raise_on_quota:
            raise
        return SQLGeneration(_rule_based_sql(question, dataset, intent), "fallback", mode,
                             error="LLM daily quota exhausted", examples=examples, docs=docs)
    except LLMError as e:
        return SQLGeneration(_rule_based_sql(question, dataset, intent), "fallback", mode,
                             error=f"LLM call failed: {e}", examples=examples, docs=docs)

    try:
        safe_sql = validate_sql(llm_sql)
    except UnsafeSQLError as e:
        return SQLGeneration(_rule_based_sql(question, dataset, intent), "fallback", mode, llm_sql=llm_sql,
                             error=f"rejected by safety check: {e}", rejected=True, examples=examples, docs=docs)
    return SQLGeneration(safe_sql, "llm", mode, llm_sql=llm_sql, examples=examples, docs=docs)


def generate_sql(question: str, dataset: Dataset, mode: str = "llm_rag", **kwargs) -> str:
    """Return only the SQL for `question` (falling back to the rule-based generator if needed)."""
    return generate_sql_detailed(question, dataset, mode, **kwargs).sql
