"""Retrieval over the business glossaries in docs/glossary/<dataset>.md.

Each `## <chunk_id> — <term>` section is one chunk (one term + its definition). Three retrievers rank a
dataset's chunks for a question:
  dense  - cosine similarity of sentence-transformers embeddings (same model as the example bank)
  bm25   - BM25Okapi over lowercased word tokens (rank_bm25)
  hybrid - reciprocal rank fusion of the dense and BM25 rankings: score = sum 1 / (RRF_K + rank)
"""
import re
import threading
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
GLOSSARY_DIR = ROOT_DIR / "docs" / "glossary"
RRF_K = 60  # the standard RRF constant (Cormack et al., 2009)
RETRIEVERS = ("dense", "bm25", "hybrid")

_HEADING = re.compile(r"^## (\S+) — (.+)$", re.MULTILINE)
_TOKEN = re.compile(r"[a-z0-9]+")


def load_glossary(dataset: str, glossary_dir: Path = GLOSSARY_DIR) -> list:
    """The chunks of docs/glossary/<dataset>.md as dicts with `id`, `term`, `definition` and `text`."""
    md = (Path(glossary_dir) / f"{dataset}.md").read_text(encoding="utf-8")
    heads = list(_HEADING.finditer(md))
    chunks = []
    for h, nxt in zip(heads, heads[1:] + [None]):
        definition = " ".join(md[h.end(): nxt.start() if nxt else len(md)].split())
        term = h.group(2).strip()
        chunks.append({"id": h.group(1), "term": term, "definition": definition,
                       "text": f"{term}: {definition}"})
    return chunks


def tokenize(text: str) -> list:
    return _TOKEN.findall(text.lower())


def rrf_fuse(rankings, k: int = RRF_K) -> list:
    """Fuse ranked lists of indices; returns all indices sorted by fused score (ties by first appearance)."""
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: -scores[i])


class GlossaryRetriever:
    """Ranks one dataset's glossary chunks for a question with the dense, BM25 or hybrid retriever."""

    def __init__(self, dataset: str, model=None, glossary_dir: Path = GLOSSARY_DIR):
        from rank_bm25 import BM25Okapi

        self.chunks = load_glossary(dataset, glossary_dir)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.model = model if model is not None else _embedding_model()
        self.vectors = self._embed([c["text"] for c in self.chunks])

    def _embed(self, texts) -> np.ndarray:
        return self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)

    def rank(self, question: str, method: str = "hybrid") -> list:
        """All chunk indices, best first."""
        if method == "dense":
            sims = self.vectors @ self._embed([question])[0]
            return list(np.argsort(-sims, kind="stable"))
        if method == "bm25":
            return list(np.argsort(-self.bm25.get_scores(tokenize(question)), kind="stable"))
        if method == "hybrid":
            return rrf_fuse([self.rank(question, "dense"), self.rank(question, "bm25")])
        raise ValueError(f"unknown retriever: {method}")

    def search(self, question: str, k: int = 3, method: str = "hybrid") -> list:
        """The top-k chunks (dicts) for `question`."""
        return [self.chunks[i] for i in self.rank(question, method)[:k]]


_model = None
_retrievers = {}
_lock = threading.Lock()


def _embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        from rag_sql import EMBEDDING_MODEL
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _model


def get_glossary_retriever(dataset: str) -> GlossaryRetriever:
    with _lock:
        if dataset not in _retrievers:
            _retrievers[dataset] = GlossaryRetriever(dataset)
    return _retrievers[dataset]
