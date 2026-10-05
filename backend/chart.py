"""Picks a default chart for a query result from the predicted intent and the result's shape."""
import re

import pandas as pd

MAX_BAR_CATEGORIES = 50
MAX_PIE_SLICES = 6
PIE_INTENTS = ("count", "compare")
# Averages, minimums and maximums aren't parts of a whole, so they never get a pie.
_NON_ADDITIVE = re.compile(r"\b(AVG|MIN|MAX)\s*\(", re.IGNORECASE)
_PERIOD = re.compile(r"^\d{4}(-\d{2}){0,2}$|^\d{4}-?Q[1-4]$|^FY\d{4}", re.IGNORECASE)


def _looks_like_period(values: pd.Series) -> bool:
    sample = values.dropna().astype(str).head(20)
    return not sample.empty and sample.map(lambda v: bool(_PERIOD.match(v))).all()


def suggest_chart(intent: str, df: pd.DataFrame, sql: str = "") -> dict:
    """{"type": bar|line|pie|table|stat, "x", "y", "reason"}. x/y name result columns (None if unused)."""
    def chart(kind, x, y, reason):
        return {"type": kind, "x": x, "y": y, "reason": reason}

    if df.empty:
        return chart("table", None, None, "no rows")

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    others = [c for c in df.columns if c not in numeric]

    if len(df) == 1 and len(numeric) == 1 and not others:
        return chart("stat", None, numeric[0], "a single number")
    if intent == "filter":
        return chart("table", None, None, "a list of records")
    if not numeric:
        return chart("table", None, None, "no numeric column to plot")
    if not others:
        # e.g. "per year" results where the period came back as an integer column
        if len(numeric) >= 2 and _looks_like_period(df[numeric[0]]):
            return chart("line", numeric[0], numeric[1], "time periods × one metric")
        return chart("table", None, None, "no category column to plot against")

    x, y = others[0], numeric[0]
    if intent == "trend" or _looks_like_period(df[x]):
        return chart("line", x, y, "time periods × one metric")
    if len(df) > MAX_BAR_CATEGORIES:
        return chart("table", None, None, f"more than {MAX_BAR_CATEGORIES} categories")
    if (intent in PIE_INTENTS and 2 <= len(df) <= MAX_PIE_SLICES and (df[y].dropna() >= 0).all()
            and not _NON_ADDITIVE.search(sql)):
        return chart("pie", x, y, "a few categories sharing one total")
    return chart("bar", x, y, "categories × one metric")
