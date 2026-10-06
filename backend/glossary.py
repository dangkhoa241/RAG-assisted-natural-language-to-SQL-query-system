"""Gated glossary RAG with the settings frozen in Stage 3C (eval/stage3c_config.json, "doc_rag_gated").

A definition is sent only if its term or one of its aliases appears in the question
(doc_retrieval.match_terms). With no match, nothing is sent and the prompt is exactly the zero-shot one.
This needs no embedding model: the glossaries are parsed once and matched as text.
"""
import json
from functools import lru_cache

from doc_retrieval import load_glossary, match_terms

from backend import ROOT_DIR

STAGE3C_CONFIG = ROOT_DIR / "eval" / "stage3c_config.json"


def _load_gate() -> dict:
    gate = json.loads(STAGE3C_CONFIG.read_text(encoding="utf-8"))["doc_rag_gated"]
    if gate["doc_method"] != "gated" or gate["retrieval_fallback"] is not None:
        raise RuntimeError(f"{STAGE3C_CONFIG.name}: the app implements only term gating without a retrieval fallback")
    return gate


GATE = _load_gate()


@lru_cache(maxsize=None)
def glossary_chunks(name: str) -> tuple:
    return tuple(load_glossary(name))


def _matched_alias(question: str, chunk: dict) -> str:
    """The term or alias that triggered the match, for display."""
    for alias in chunk["aliases"]:
        if match_terms(question, [dict(chunk, aliases=[alias])]):
            return alias
    return chunk["term"]


def matched_definitions(question: str, name: str) -> list:
    """The chunks of glossary `name` whose term or alias appears in `question`, in glossary order,
    each with `matched` set to the phrase that matched. Empty when no term appears."""
    chunks = match_terms(question, list(glossary_chunks(name)))
    if GATE["max_chunks"]:
        chunks = chunks[:GATE["max_chunks"]]
    return [dict(c, matched=_matched_alias(question, c)) for c in chunks]
