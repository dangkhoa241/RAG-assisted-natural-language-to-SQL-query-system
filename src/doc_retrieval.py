"""Retrieval over the business glossaries in docs/glossary/<dataset>.md.

Each `## <chunk_id> — <term>` section is one chunk (one term + its definition). An optional
`Aliases: a; b; c` line in a section lists other names for the term; it is kept out of the definition
text, so adding aliases never changes a prompt. Three retrievers rank a dataset's chunks for a question:
  dense  - cosine similarity of sentence-transformers embeddings (same model as the example bank)
  bm25   - BM25Okapi over lowercased word tokens (rank_bm25)
  hybrid - reciprocal rank fusion of the dense and BM25 rankings: score = sum 1 / (RRF_K + rank)

Gating (`match_terms`, method "gated") is not a ranking: it returns only the chunks whose term or an alias
appears in the question, so a question that uses no glossary term gets no definitions.
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
_ALIASES = re.compile(r"^Aliases:(.*)$", re.MULTILINE)
_TOKEN = re.compile(r"[a-z0-9]+")
_FISCAL_YEAR_LABEL = re.compile(r"^fy\d{2,4}$")  # "FY2024" matches the alias "FY"


def load_glossary(dataset: str, glossary_dir: Path = GLOSSARY_DIR) -> list:
    """The chunks of docs/glossary/<dataset>.md as dicts with `id`, `term`, `definition`, `text` and
    `aliases` (the term itself, any parenthesized short form such as "AOV", and the `Aliases:` line)."""
    md = (Path(glossary_dir) / f"{dataset}.md").read_text(encoding="utf-8")
    heads = list(_HEADING.finditer(md))
    chunks = []
    for h, nxt in zip(heads, heads[1:] + [None]):
        body = md[h.end(): nxt.start() if nxt else len(md)]
        listed = [a.strip() for m in _ALIASES.finditer(body) for a in m.group(1).split(";") if a.strip()]
        definition = " ".join(_ALIASES.sub("", body).split())
        term = h.group(2).strip()
        short = re.findall(r"\(([^)]+)\)", term)
        base = re.sub(r"\s*\([^)]*\)", "", term).strip()
        aliases = list(dict.fromkeys([base, *short, *listed]))
        chunks.append({"id": h.group(1), "term": term, "definition": definition,
                       "text": f"{term}: {definition}", "aliases": aliases})
    return chunks


def tokenize(text: str) -> list:
    return _TOKEN.findall(text.lower())


def _stem(token: str) -> str:
    """Light normalization for term matching: plural -s and fiscal-year labels."""
    if _FISCAL_YEAR_LABEL.match(token):
        return "fy"
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def match_terms(question: str, chunks: list) -> list:
    """Chunks whose term or alias occurs in `question` as a whole-word phrase (case-insensitive, hyphens
    as spaces, plural -s ignored), in glossary order. A match inside a longer match for a different chunk
    is dropped, so "flagged claims" gives only the flagged-claim chunk, not "flagged" (flagged result)."""
    words = [_stem(t) for t in tokenize(question)]
    spans = []
    for i, c in enumerate(chunks):
        for alias in c["aliases"]:
            a = [_stem(t) for t in tokenize(alias)]
            for s in range(len(words) - len(a) + 1):
                if a and words[s:s + len(a)] == a:
                    spans.append((s, s + len(a), i))
    keep = {i for s, e, i in spans
            if not any(j != i and s2 <= s and e <= e2 and (e2 - s2) > (e - s) for s2, e2, j in spans)}
    return [chunks[i] for i in sorted(keep)]


def rrf_fuse(rankings, k: int = RRF_K) -> list:
    """Fuse ranked lists of indices; returns all indices sorted by fused score (ties by first appearance)."""
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: -scores[i])


class GlossaryRetriever:
    """Ranks one dataset's glossary chunks for a question with the dense, BM25 or hybrid retriever."""

    def __init__(self, dataset: str, model=None, glossary_dir: Path = GLOSSARY_DIR, index_aliases: bool = False):
        """With `index_aliases`, the dense and BM25 indexes also see each chunk's aliases (the definitions
        sent to the LLM are the same either way)."""
        from rank_bm25 import BM25Okapi

        self.chunks = load_glossary(dataset, glossary_dir)
        texts = [c["text"] + (" Also called: " + "; ".join(c["aliases"]) if index_aliases else "")
                 for c in self.chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.model = model if model is not None else _embedding_model()
        self.vectors = self._embed(texts)

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
        """The top-k chunks (dicts) for `question`; with method "gated", every term-matched chunk (k unused)."""
        if method == "gated":
            return match_terms(question, self.chunks)
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
