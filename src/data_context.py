"""CSV loading, type inference, and the SQLite table backing a query session."""
from dataclasses import dataclass

import pandas as pd
import sqlite3

MAX_CATEGORY_VALUES = 50


@dataclass
class Dataset:
    df: pd.DataFrame
    conn: sqlite3.Connection
    numeric_cols: list
    categorical_cols: list
    date_cols: set


def _coerce_numeric_columns(df: pd.DataFrame) -> None:
    """Coerce numeric-looking text columns in place (handles "$1,234.50" style values)."""
    for c in df.columns:
        clean = df[c].astype(str).str.replace(r"[,$ ]", "", regex=True)
        if clean.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.7:
            df[c] = pd.to_numeric(clean, errors="coerce")


def _detect_date_columns(df: pd.DataFrame) -> set:
    """Detect date-like columns either by name or by sampling parse success, in place,
    so this works on any schema, not just columns literally named "*date*"."""
    date_cols = set()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        if "date" in c.lower():
            parsed = pd.to_datetime(df[c], errors="coerce")
        else:
            sample = df[c].dropna().astype(str).head(200)
            if sample.empty:
                continue
            if pd.to_datetime(sample, errors="coerce").notna().mean() < 0.8:
                continue
            parsed = pd.to_datetime(df[c], errors="coerce")
        # Judge parse success on non-empty cells, so a mostly-empty date column (e.g. a cancel date that is
        # blank for active accounts) is still a date column.
        if parsed.notna().sum() < 0.5 * df[c].notna().sum() or parsed.notna().sum() == 0:
            continue
        df[c] = parsed.dt.strftime("%Y-%m-%d")
        date_cols.add(c)
    return date_cols


def load_dataset(uploaded_file) -> Dataset:
    df = pd.read_csv(uploaded_file)

    _coerce_numeric_columns(df)
    date_cols = _detect_date_columns(df)

    # Classify remaining columns as numeric / categorical / free text, driven
    # entirely by the uploaded data instead of a hardcoded schema.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in date_cols]
    categorical_cols = [
        c for c in df.columns
        if c not in date_cols and c not in numeric_cols and df[c].nunique(dropna=True) <= MAX_CATEGORY_VALUES
    ]

    conn = sqlite3.connect(":memory:")
    df.to_sql("data", conn, index=False, if_exists="replace")

    return Dataset(df=df, conn=conn, numeric_cols=numeric_cols, categorical_cols=categorical_cols, date_cols=date_cols)
