"""Schema-aware, rule-based NL -> SQL generation.

Every lookup here is driven by the uploaded dataset's actual columns and values
(via the `Dataset` context), not by any hardcoded column names.
"""
import re
from datetime import date, timedelta

from data_context import Dataset


def escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


# CATEGORICAL FILTERS (values pulled straight from the uploaded data, not a hardcoded list)
def extract_categorical_filters(query: str, dataset: Dataset):
    ql = f" {query.lower()} "
    filters = {}

    for col in dataset.categorical_cols:
        values = sorted((str(v) for v in dataset.df[col].dropna().unique()), key=len, reverse=True)
        for val in values:
            val_l = val.lower().strip()
            if not val_l:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(val_l)}(?![a-z0-9])"
            if re.search(pattern, ql):
                # A list (not a set) keeps the generated SQL independent of the hash seed.
                matched = filters.setdefault(col, [])
                if val not in matched:
                    matched.append(val)

    return filters


# NUMERIC FILTERS (matches a comparison word + number to whichever numeric column's
# name appears nearby in the question, so it works for any numeric column, not just Age/Billing)
COMPARISON_OPS = {
    "greater than": ">", "more than": ">", "at least": ">=", "over": ">", "above": ">",
    "less than": "<", "under than": "<", "at most": "<=", "under": "<", "below": "<",
}
# The optional middle word handles phrasing like "under age 40" or "over salary 90000",
# where a column-name word sits between the comparison term and the number.
COMPARISON_PATTERN = re.compile(
    r"(" + "|".join(sorted(COMPARISON_OPS, key=len, reverse=True)) + r")\s+(?:([a-z]+)\s+)?(\d+(?:\.\d+)?)"
)


def extract_numeric_filters(query: str, dataset: Dataset):
    if not dataset.numeric_cols:
        return []

    ql = query.lower().replace("$", "").replace(",", "")
    col_keywords = {col: set(re.findall(r"[a-z]+", col.lower())) for col in dataset.numeric_cols}

    def best_column(text: str):
        text_tokens = set(re.findall(r"[a-z]+", text))
        best_col, best_score = None, 0
        for col, kws in col_keywords.items():
            score = len(kws & text_tokens)
            if score > best_score:
                best_col, best_score = col, score
        return best_col

    conds = []
    for m in COMPARISON_PATTERN.finditer(ql):
        op_word, mid_word, num_str = m.groups()
        op = COMPARISON_OPS[op_word]
        num = float(num_str)

        start, end = m.span()
        window = ql[max(0, start - 30): min(len(ql), end + 30)]
        col = (best_column(mid_word) if mid_word else None) or best_column(window) or best_column(ql)
        if col is None and len(dataset.numeric_cols) == 1:
            col = dataset.numeric_cols[0]
        if col is None:
            continue
        conds.append(f"`{col}` {op} {num:g}")

    return list(dict.fromkeys(conds))


# DATE RANGE (relative/absolute date phrases — domain agnostic)
def parse_date_range(query: str):
    q = query.lower()
    today = date.today()

    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        return date(y, 1, 1), date(y, 12, 31)

    if "last year" in q:
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)

    if "last month" in q:
        end = today.replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
        return start, end

    if "last 30 days" in q:
        return today - timedelta(days=30), today

    return None, None


# GROUP-BY COLUMN DETECTION
def detect_group_column(query: str, dataset: Dataset):
    if not dataset.categorical_cols:
        return None

    ql = query.lower()
    m = re.search(r"\bby ([a-z0-9 _-]+)", ql)
    phrase = None
    if m:
        phrase = m.group(1)
        phrase = re.split(r"\b(for|with|who|that|where|and)\b", phrase)[0].strip()

    def col_tokens(col):
        return set(re.findall(r"[a-z0-9]+", col.lower()))

    for text in filter(None, [phrase, ql]):
        text_tokens = set(re.findall(r"[a-z0-9]+", text))
        best_col, best_score = None, 0
        for col in dataset.categorical_cols:
            score = len(col_tokens(col) & text_tokens)
            if score > best_score:
                best_col, best_score = col, score
        if best_col:
            return best_col

    # Fall back to the lowest-cardinality categorical column — the most
    # "grouping-like" dimension when the question doesn't name one.
    return min(dataset.categorical_cols, key=lambda c: dataset.df[c].nunique(dropna=True))


# DATE COLUMN CHOICE
def pick_date_column(dataset: Dataset):
    """The date column to filter and bucket on: the first column (in CSV order) whose name
    contains "date", else the first detected date column. None if the data has no dates.
    Deterministic, unlike iterating over the `date_cols` set."""
    ordered = [c for c in dataset.df.columns if c in dataset.date_cols]
    named = [c for c in ordered if "date" in c.lower()]
    return (named or ordered or [None])[0]


# WHERE CLAUSE BUILDER
def build_where_clauses(query: str, dataset: Dataset):
    clauses = []

    cat_filters = extract_categorical_filters(query, dataset)
    for col, vals in cat_filters.items():
        if len(vals) == 1:
            val = vals[0]
            clauses.append(f"LOWER(`{col}`) = LOWER('{escape_sql_literal(val)}')")
        else:
            parts = [f"LOWER(`{col}`) = LOWER('{escape_sql_literal(v)}')" for v in vals]
            clauses.append("(" + " OR ".join(parts) + ")")

    clauses.extend(extract_numeric_filters(query, dataset))

    start, end = parse_date_range(query)
    date_col = pick_date_column(dataset)
    if start and end and date_col:
        clauses.append(f"date(`{date_col}`) BETWEEN date('{start}') AND date('{end}')")

    return clauses


def detect_metric(ql: str, dataset: Dataset):
    """Pick an aggregation function + numeric column based on the question text."""
    if any(w in ql for w in ["how many", "count", "number of"]):
        return "COUNT(*)"

    ql_tokens = set(re.findall(r"[a-z]+", ql))
    metric_col, best_score = None, 0
    for col in dataset.numeric_cols:
        tokens = set(re.findall(r"[a-z]+", col.lower()))
        score = len(tokens & ql_tokens)
        if score > best_score:
            metric_col, best_score = col, score
    if metric_col is None and len(dataset.numeric_cols) == 1:
        metric_col = dataset.numeric_cols[0]

    if metric_col is None:
        return "COUNT(*)"

    if any(w in ql for w in ["total", "sum"]):
        agg_func = "SUM"
    elif any(w in ql for w in ["max", "maximum", "highest"]):
        agg_func = "MAX"
    elif any(w in ql for w in ["min", "minimum", "lowest"]):
        agg_func = "MIN"
    else:
        agg_func = "AVG"

    return f"{agg_func}(`{metric_col}`)"


# SQL BUILDER
def build_sql(query: str, intent: str, dataset: Dataset):
    ql = query.lower()

    where_clauses = build_where_clauses(query, dataset)
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    if intent == "filter":
        return intent, f"SELECT * FROM data {where_sql} LIMIT 200;"

    if intent == "count":
        return intent, f"SELECT COUNT(*) AS value FROM data {where_sql};"

    if intent in ("aggregate", "compare"):
        group_col = detect_group_column(query, dataset)
        if group_col is None:
            return "filter", f"SELECT * FROM data {where_sql} LIMIT 200;"

        metric_expr = detect_metric(ql, dataset)
        sql = f"""
        SELECT `{group_col}` AS category,
            {metric_expr} AS value
        FROM data
        {where_sql}
        GROUP BY `{group_col}`
        ORDER BY value DESC;
        """
        return intent, sql

    if intent == "trend":
        date_col = pick_date_column(dataset)
        if date_col is None:
            return "filter", f"SELECT * FROM data {where_sql} LIMIT 200;"

        if "by year" in ql:
            bucket_expr = f"strftime('%Y', `{date_col}`)"
        elif "by month" in ql:
            bucket_expr = f"strftime('%Y-%m', `{date_col}`)"
        else:
            bucket_expr = f"`{date_col}`"

        metric_expr = detect_metric(ql, dataset)
        sql = f"""
        SELECT {bucket_expr} AS period,
            {metric_expr} AS value
        FROM data
        {where_sql}
        GROUP BY period
        ORDER BY period;
        """
        return intent, sql

    return intent, f"SELECT * FROM data {where_sql} LIMIT 200;"
