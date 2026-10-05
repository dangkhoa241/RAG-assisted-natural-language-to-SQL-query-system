"""Guardrails for running generated SQL: one read-only SELECT, with a timeout and a row cap.

Three independent layers, so a gap in one is caught by the next:
  1. validate_sql()        - static check: a single SELECT/WITH statement, no write/admin keywords
  2. readonly_connection() - a private copy of the database with PRAGMA query_only, an authorizer
                             that only permits reads, and a cap on the size of any one value
  3. run_safe_query()      - a wall-clock timeout and a cap on returned rows
"""
import re
import sqlite3
import time

import pandas as pd

QUERY_TIMEOUT_S = 5.0
MAX_RESULT_ROWS = 1000
# SQLite's default is 1 GB, so one `SELECT zeroblob(...)` or printf('%.*c', ...) could exhaust memory.
# Together with MAX_RESULT_ROWS this bounds a result at ~100 MB even if every value is maximal.
MAX_VALUE_BYTES = 100_000

FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "REPLACE", "UPSERT", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "ANALYZE", "BEGIN", "COMMIT", "ROLLBACK",
    "SAVEPOINT", "RELEASE", "TRANSACTION", "LOAD_EXTENSION",
}

# String literals, quoted identifiers and comments, in the order SQLite tokenizes them.
_LITERALS_AND_COMMENTS = re.compile(
    r"'(?:[^']|'')*'"        # 'string literal' with '' escapes
    r'|"(?:[^"]|"")*"'       # "quoted identifier"
    r"|`(?:[^`]|``)*`"       # `quoted identifier`
    r"|\[[^\]]*\]"           # [quoted identifier]
    r"|--[^\n]*"             # -- line comment
    r"|/\*.*?(?:\*/|$)",     # /* block comment */ (unterminated runs to the end)
    re.DOTALL,
)

_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


class UnsafeSQLError(ValueError):
    """The SQL failed the static safety check and must not be executed."""


class QueryTimeoutError(RuntimeError):
    """The query ran longer than the allowed time and was interrupted."""


def _mask(sql: str) -> str:
    """Blank out literals, quoted identifiers and comments so keyword checks only see SQL syntax."""
    return _LITERALS_AND_COMMENTS.sub(" ", sql)


def validate_sql(sql: str) -> str:
    """Return the statement without trailing semicolons, or raise UnsafeSQLError."""
    if not sql or not sql.strip():
        raise UnsafeSQLError("empty SQL")

    statement = sql.strip()
    while statement.endswith(";"):
        statement = statement[:-1].rstrip()

    masked = _mask(statement)
    if ";" in masked:
        raise UnsafeSQLError("multiple statements are not allowed")

    words = re.findall(r"[A-Za-z_]+", masked)
    if not words or words[0].upper() not in ("SELECT", "WITH"):
        raise UnsafeSQLError("only SELECT statements are allowed")

    bad = sorted({w.upper() for w in words} & FORBIDDEN_KEYWORDS)
    if bad:
        raise UnsafeSQLError(f"forbidden keyword(s): {', '.join(bad)}")

    return statement


def _authorizer(action, arg1, arg2, db_name, trigger):
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() == "load_extension":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def readonly_connection(source: sqlite3.Connection) -> sqlite3.Connection:
    """A private, read-only in-memory copy of `source`. Writes can't reach the original."""
    conn = sqlite3.connect(":memory:")
    conn.deserialize(source.serialize())
    conn.execute("PRAGMA query_only = ON")
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
    conn.set_authorizer(_authorizer)
    return conn


def run_safe_query(sql: str, source: sqlite3.Connection,
                   timeout_s: float = QUERY_TIMEOUT_S, max_rows: int = MAX_RESULT_ROWS) -> pd.DataFrame:
    """Validate and run `sql` read-only. Results beyond `max_rows` are dropped
    (df.attrs["truncated"] is set). Raises UnsafeSQLError, QueryTimeoutError or sqlite3.Error."""
    statement = validate_sql(sql)
    conn = readonly_connection(source)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(statement)
        columns = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            raise QueryTimeoutError(f"query exceeded {timeout_s}s") from e
        raise
    finally:
        conn.close()

    df = pd.DataFrame(rows[:max_rows], columns=columns)
    df.attrs["truncated"] = len(rows) > max_rows
    return df
