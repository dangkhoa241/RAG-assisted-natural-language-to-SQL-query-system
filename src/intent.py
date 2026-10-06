"""Intent detection: a trained BERT classifier when available, with a keyword-based fallback."""
import os
from functools import lru_cache

# Defaults to the local training output. Set INTENT_MODEL_PATH to a Hugging Face
# Hub repo id (e.g. "username/intent-model") to load a hosted model instead --
# useful when deploying, since the trained weights aren't checked into git.
DEFAULT_INTENT_MODEL_PATH = "intent_model"
# INTENT_QUANTIZE=int8 applies dynamic int8 quantization to the model's Linear layers at load time
# (weights stored as int8, activations quantized on the fly). Measured in eval/results/intent_quantization.md.
QUANTIZE_CHOICES = ("none", "int8")


def quantize_int8(model):
    """Dynamic int8 quantization of every nn.Linear (the encoder's attention/FFN layers and the classifier
    head). Embeddings and LayerNorm stay fp32. In place: the default deep-copies the whole fp32 model first,
    which more than doubles peak memory while loading."""
    import torch

    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:   # older torch
        from torch.quantization import quantize_dynamic
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)


def intent_quantization() -> str:
    mode = os.environ.get("INTENT_QUANTIZE", "none").strip().lower() or "none"
    if mode not in QUANTIZE_CHOICES:
        raise ValueError(f"INTENT_QUANTIZE must be one of {QUANTIZE_CHOICES}, got {mode!r}")
    return mode


@lru_cache(maxsize=1)
def load_intent_classifier():
    """Loaded once per process; None if the model can't be loaded (callers then use keywords).
    transformers (and torch) are imported here, so the keyword path never pays for them."""
    path = os.environ.get("INTENT_MODEL_PATH", DEFAULT_INTENT_MODEL_PATH)
    quantize = intent_quantization()
    try:
        from transformers import pipeline

        clf = pipeline("text-classification", model=path, tokenizer=path)
        if quantize == "int8":
            clf.model = quantize_int8(clf.model)
        return clf
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
