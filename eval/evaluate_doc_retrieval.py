"""Glossary retrieval quality on the glossary benchmark (no LLM calls).

For each question, the retriever ranks the chunks of its dataset's glossary; the gold chunks are the
question's `required_chunks`. Metrics, averaged over questions:
  recall@k   - fraction of a question's required chunks found in the top k
  all@k      - share of questions whose required chunks are ALL in the top k (what doc_rag needs)
  MRR        - reciprocal rank of the first required chunk

Writes eval/results/doc_retrieval_results.{json,md}.
Usage (from the repo root):  python eval/evaluate_doc_retrieval.py
"""
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from doc_retrieval import RETRIEVERS, RRF_K, GlossaryRetriever  # noqa: E402
from rag_sql import EMBEDDING_MODEL  # noqa: E402

QUESTIONS_PATH = ROOT_DIR / "eval" / "sql_benchmark" / "glossary_questions.jsonl"
RESULTS_DIR = ROOT_DIR / "eval" / "results"
KS = [1, 3, 5]
DATASETS = ["healthcare", "retail"]


def metrics(rows):
    n = len(rows)
    out = {"n": n}
    for k in KS:
        out[f"recall@{k}"] = sum(len(set(r["ranked"][:k]) & set(r["gold"])) / len(r["gold"]) for r in rows) / n
        out[f"all@{k}"] = sum(set(r["gold"]) <= set(r["ranked"][:k]) for r in rows) / n
    out["mrr"] = sum(1 / (1 + min(r["ranked"].index(g) for g in r["gold"])) for r in rows) / n
    return out


def main():
    questions = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    retrievers = {name: GlossaryRetriever(name) for name in DATASETS}

    per_question, results = [], {}
    for method in RETRIEVERS:
        rows = []
        for q in questions:
            r = retrievers[q["dataset"]]
            ranked = [r.chunks[i]["id"] for i in r.rank(q["question"], method)]
            rows.append({"id": q["id"], "dataset": q["dataset"], "method": method,
                         "gold": q["required_chunks"], "ranked": ranked})
        per_question += rows
        results[method] = {"overall": metrics(rows),
                           **{d: metrics([x for x in rows if x["dataset"] == d]) for d in DATASETS}}

    best = max(RETRIEVERS, key=lambda m: (results[m]["overall"]["all@3"], results[m]["overall"]["recall@3"]))
    report = {"config": {"embedding_model": EMBEDDING_MODEL, "rrf_k": RRF_K, "n_questions": len(questions),
                         "chunks_per_glossary": {d: len(retrievers[d].chunks) for d in DATASETS}},
              "results": results, "best_by_all@3": best}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "doc_retrieval_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    misses = [{"id": x["id"], "method": x["method"], "gold": x["gold"], "top3": x["ranked"][:3]}
              for x in per_question if not set(x["gold"]) <= set(x["ranked"][:3])]
    lines = ["# Glossary retrieval results", "",
             f"{len(questions)} glossary questions; each dataset's glossary has "
             f"{report['config']['chunks_per_glossary']['healthcare']} (healthcare) / "
             f"{report['config']['chunks_per_glossary']['retail']} (retail) chunks, including distractors. "
             f"Dense = `{EMBEDDING_MODEL}`, BM25 = rank_bm25 BM25Okapi, hybrid = reciprocal rank fusion (k={RRF_K}).",
             "", "recall@k = share of required chunks in the top k; all@k = share of questions with every "
             "required chunk in the top k.", "",
             "| Retriever | recall@1 | recall@3 | recall@5 | all@3 | all@5 | MRR | recall@3 healthcare | recall@3 retail |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in RETRIEVERS:
        o = results[m]["overall"]
        lines.append(f"| {m} | {o['recall@1']:.1%} | **{o['recall@3']:.1%}** | {o['recall@5']:.1%} | "
                     f"{o['all@3']:.1%} | {o['all@5']:.1%} | {o['mrr']:.3f} | "
                     f"{results[m]['healthcare']['recall@3']:.1%} | {results[m]['retail']['recall@3']:.1%} |")
    lines += ["", f"Best retriever by all@3: **{best}** (used for doc_rag with k=3).", "",
              "## Questions with a required chunk missing from the top 3", "",
              "| Question | Retriever | Required | Retrieved top 3 |", "|---|---|---|---|"]
    lines += [f"| {x['id']} | {x['method']} | {', '.join(x['gold'])} | {', '.join(x['top3'])} |" for x in misses]
    md = "\n".join(lines) + "\n"
    (RESULTS_DIR / "doc_retrieval_results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
