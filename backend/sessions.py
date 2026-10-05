"""In-memory dataset sessions: the built-in samples (never expire) and uploads (TTL + LRU cap).

Every session owns its own in-memory SQLite connection holding only its own table, so a query
on one dataset has no way to name, let alone read, another dataset's data.
"""
import io
import re
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import pandas as pd

from data_context import Dataset, load_dataset
from rag_sql import build_schema_context

MAX_SAMPLE_VALUES = 3
MAX_VALUE_CHARS = 60
# Cell values and column names reach the LLM prompt; these caps bound what an uploaded CSV can put there.
MAX_SCHEMA_LINE_CHARS = 300
MAX_SCHEMA_CONTEXT_CHARS = 12_000
MAX_COLUMN_NAME_CHARS = 64
_BAD_COLUMN_CHARS = re.compile(r"[`\x00-\x1f\x7f]")


class DatasetError(ValueError):
    """The upload isn't a usable CSV. The message is safe to show to the user."""


@dataclass
class Session:
    id: str
    name: str
    dataset: Dataset
    schema: list
    schema_context: str
    glossary: str = None             # docs/glossary/<glossary>.md, for the built-in samples only
    description: str = ""
    example_questions: list = field(default_factory=list)
    expires_at: float = None         # None = never expires


def _thread_safe_copy(conn: sqlite3.Connection) -> sqlite3.Connection:
    """load_dataset() makes a connection bound to the creating thread; FastAPI serves requests on a
    thread pool, so move the table into a connection usable from any thread (one per dataset). The
    source is never written after this, and each query reads a private read-only copy of it
    (sql_safety.readonly_connection); Python's sqlite3 is built in serialized mode (threadsafety 3)."""
    copy = sqlite3.connect(":memory:", check_same_thread=False)
    copy.deserialize(conn.serialize())
    conn.close()
    return copy


def _json_value(v):
    if isinstance(v, str):
        return v if len(v) <= MAX_VALUE_CHARS else v[:MAX_VALUE_CHARS - 1] + "…"
    return v.item() if hasattr(v, "item") else v


def describe_schema(dataset: Dataset) -> list:
    out = []
    for col in dataset.df.columns:
        if col in dataset.date_cols:
            kind = "date"
        elif col in dataset.numeric_cols:
            kind = "numeric"
        elif col in dataset.categorical_cols:
            kind = "categorical"
        else:
            kind = "text"
        values = dataset.df[col].dropna().unique()[:MAX_SAMPLE_VALUES]
        out.append({"name": col, "type": kind, "sample_values": [_json_value(v) for v in values]})
    return out


def bounded_schema_context(dataset: Dataset) -> str:
    """rag_sql's schema context, with every line and the whole text length-capped."""
    lines = [ln if len(ln) <= MAX_SCHEMA_LINE_CHARS else ln[:MAX_SCHEMA_LINE_CHARS] + " …"
             for ln in build_schema_context(dataset).splitlines()]
    text = "\n".join(lines)
    return text if len(text) <= MAX_SCHEMA_CONTEXT_CHARS else text[:MAX_SCHEMA_CONTEXT_CHARS] + "\n…"


def load_session_dataset(source, max_columns: int = None, max_rows: int = None) -> Dataset:
    """Load a CSV (path or bytes) with data_context.load_dataset, validate it, and give it its own connection."""
    if isinstance(source, (bytes, bytearray)):
        # Checked before parsing: a 10 MB file of one-character rows is ~5M rows and takes seconds of CPU.
        # (Newlines inside quoted cells only over-count, so this can't let a too-long file through.)
        if max_rows and source.count(b"\n") > max_rows + 1:
            raise DatasetError(f"The CSV has more than {max_rows:,} rows.")
        source = io.BytesIO(source)
    try:
        dataset = load_dataset(source)
    except UnicodeDecodeError:
        raise DatasetError("The file isn't UTF-8 text. Save it as a UTF-8 CSV and try again.")
    except Exception:
        raise DatasetError("The file couldn't be parsed as a CSV.")
    df = dataset.df
    if df.empty or len(df.columns) == 0:
        raise DatasetError("The CSV has no data rows.")
    if max_rows and len(df) > max_rows:
        raise DatasetError(f"The CSV has more than {max_rows:,} rows.")
    if max_columns and len(df.columns) > max_columns:
        raise DatasetError(f"The CSV has {len(df.columns)} columns; the limit is {max_columns}.")
    for col in df.columns:
        name = str(col)
        if not name.strip() or len(name) > MAX_COLUMN_NAME_CHARS or _BAD_COLUMN_CHARS.search(name):
            raise DatasetError("Column names must be 1-64 characters, with no backticks or control characters.")
    dataset.conn = _thread_safe_copy(dataset.conn)
    return dataset


def make_session(dataset: Dataset, name: str, session_id: str = None, **kwargs) -> Session:
    return Session(id=session_id or uuid.uuid4().hex, name=name, dataset=dataset,
                   schema=describe_schema(dataset), schema_context=bounded_schema_context(dataset), **kwargs)


class SessionStore:
    def __init__(self, max_uploads: int, ttl_s: float, clock=time.monotonic):
        self.max_uploads = max_uploads
        self.ttl_s = ttl_s
        self.clock = clock
        self._samples = {}
        self._uploads = OrderedDict()   # least recently used first
        self._lock = threading.Lock()

    def add_sample(self, session: Session) -> None:
        self._samples[session.id] = session

    def samples(self) -> list:
        return list(self._samples.values())

    def add_upload(self, session: Session) -> Session:
        with self._lock:
            self._purge()
            while len(self._uploads) >= self.max_uploads:
                self._uploads.popitem(last=False)
            session.expires_at = self.clock() + self.ttl_s
            self._uploads[session.id] = session
        return session

    def get(self, session_id: str):
        """The session, or None if unknown or expired. Using an upload refreshes its TTL."""
        if session_id in self._samples:
            return self._samples[session_id]
        with self._lock:
            self._purge()
            session = self._uploads.get(session_id)
            if session is not None:
                session.expires_at = self.clock() + self.ttl_s
                self._uploads.move_to_end(session_id)
            return session

    def __len__(self):
        return len(self._uploads)

    def _purge(self) -> None:
        now = self.clock()
        for sid in [sid for sid, s in self._uploads.items() if s.expires_at <= now]:
            del self._uploads[sid]
