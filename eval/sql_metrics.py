"""Execution-accuracy comparison: does a predicted result set match the gold result set?

Rules:
- Column names and column order are ignored. Gold columns are matched to predicted columns by their values.
- The prediction may contain extra columns (e.g. a group label next to a scalar answer), but
  every gold column must be matched by a distinct predicted column.
- Row count must match exactly. Rows are compared as a multiset, or as a sequence if `ordered`.
- Values are normalized: numbers (and numeric-looking strings, e.g. '2023') are rounded to 2 decimals,
  other strings are stripped and lowercased, and NULL/NaN stay NULL.
"""
import math
import re
from collections import Counter

import pandas as pd

_NUMERIC = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_value(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)) or hasattr(v, "dtype"):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v).strip().lower()
        return None if math.isnan(f) else round(f, 2) + 0.0  # +0.0 turns -0.0 into 0.0
    s = str(v).strip()
    if _NUMERIC.fullmatch(s):
        return round(float(s), 2) + 0.0
    return s.lower()


def _columns(df: pd.DataFrame):
    return [[normalize_value(v) for v in df.iloc[:, i].tolist()] for i in range(df.shape[1])]


def _sort_key(v):
    # Sort mixed None/float/str values deterministically.
    return (0, 0.0, "") if v is None else (1, v, "") if isinstance(v, float) else (2, 0.0, v)


def results_match(gold: pd.DataFrame, pred: pd.DataFrame, ordered: bool = False) -> bool:
    if pred is None or len(gold) != len(pred):
        return False
    if gold.shape[1] == 0:
        return True
    if pred.shape[1] < gold.shape[1]:
        return False

    g_cols, p_cols = _columns(gold), _columns(pred)

    def signature(col):
        return tuple(col) if ordered else tuple(sorted(col, key=_sort_key))

    g_sigs = [signature(c) for c in g_cols]
    p_sigs = [signature(c) for c in p_cols]
    candidates = [[j for j, ps in enumerate(p_sigs) if ps == gs] for gs in g_sigs]
    if any(not c for c in candidates):
        return False

    gold_rows = list(zip(*g_cols))

    def rows_equal(mapping):
        pred_rows = list(zip(*(p_cols[j] for j in mapping)))
        if ordered:
            return pred_rows == gold_rows
        return Counter(pred_rows) == Counter(gold_rows)

    # Backtracking over column assignments. Usually each gold column has exactly one candidate.
    def assign(i, used, mapping):
        if i == len(g_cols):
            return rows_equal(mapping)
        for j in candidates[i]:
            if j not in used and assign(i + 1, used | {j}, mapping + [j]):
                return True
        return False

    return assign(0, frozenset(), [])
