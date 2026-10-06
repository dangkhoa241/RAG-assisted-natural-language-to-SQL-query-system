"""Intent detection: a trained BERT classifier when available, with a keyword-based fallback."""
import os
from functools import lru_cache

from transformers import pipeline

# Defaults to the local training output. Set INTENT_MODEL_PATH to a Hugging Face
# Hub repo id (e.g. "username/intent-model") to load a hosted model instead --
# useful when deploying, since the trained weights aren't checked into git.
INTENT_MODEL_PATH = os.environ.get("INTENT_MODEL_PATH", "intent_model")


@lru_cache(maxsize=1)
def load_intent_classifier():
    """Loaded once per process; None if the model can't be loaded (callers then use keywords)."""
    try:
        return pipeline("text-classification", model=INTENT_MODEL_PATH, tokenizer=INTENT_MODEL_PATH)
    except Exception:
        return None


def guess_intent_by_keywords(query: str) -> str:
    ql = query.lower()
    if any(w in ql for w in ["how many", "count", "number of"]):
        return "count"
    if any(w in ql for w in ["trend", "over time", "by month", "by year", "by day", "growth"]):
        return "trend"
    if any(w in ql for w in ["compare", "versus", " vs ", "compared to"]):
        return "compare"
    if any(w in ql for w in ["average", "avg", "total", "sum", "maximum", "minimum", "highest", "lowest", "mean"]):
        return "aggregate"
    return "filter"


def get_intent(query: str, clf) -> str:
    if clf is not None:
        try:
            return clf(query)[0]["label"]
        except Exception:
            pass
    return guess_intent_by_keywords(query)
