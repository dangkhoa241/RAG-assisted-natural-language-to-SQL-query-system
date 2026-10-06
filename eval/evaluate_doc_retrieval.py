"""Glossary retrieval quality on the glossary benchmark (no LLM calls).

For each question, the retriever ranks the chunks of its dataset's glossary; the gold chunks are the
question's `required_chunks`. Metrics, averaged over questions:
  recall@k   - fraction of a question's required chunks found in the top k
  all@k      - share of questions whose required chunks are ALL in the top k (what doc_rag needs)
  MRR        - reciprocal rank of the first required chunk

Each retriever is scored twice: on the term + definition text (Stage 3A/3B) and with the chunk's aliases
added to the index ("+aliases", Stage 3C). Term gating (doc_rag_gated) is scored separately, because it
returns a set rather than a ranking: on the glossary questions, how often the matched set contains every
required chunk and how many extra chunks it adds; on the original 120 questions (which need no definitions),
how many questions get any definition at all.

Only the dev datasets (healthcare, retail) are scored here; the SaaS questions are a held-out test.
Writes eval/results/doc_retrieval_results.{json,md}.
Usage (from the repo root):  python eval/evaluate_doc_retrieval.py
"""
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from doc_retrieval import RETRIEVERS, RRF_K, GlossaryRetriever, match_terms  # noqa: E402
from rag_sql import EMBEDDING_MODEL  # noqa: E402

QUESTIONS_PATH = ROOT_DIR / "eval" / "sql_benchmark" / "glossary_questions.jsonl"
ORIGINAL_PATH = ROOT_DIR / "eval" / "sql_benchmark" / "test_questions.jsonl"
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


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gating_metrics(questions, originals, chunks):
    matched = {q["id"]: [c["id"] for c in match_terms(q["question"], chunks[q["dataset"]])] for q in questions}
    n_matched = sum(len(m) for m in matched.values())
    n_required = sum(len(set(q["required_chunks"]) & set(matched[q["id"]])) for q in questions)
    false_matches = {q["id"]: [c["id"] for c in match_terms(q["question"], chunks[q["dataset"]])] for q in originals}
    return {"n": len(questions),
            "recall": sum(len(set(q["required_chunks"]) & set(matched[q["id"]])) / len(q["required_chunks"])
                          for q in questions) / len(questions),
            "all_required": sum(set(q["required_chunks"]) <= set(matched[q["id"]]) for q in questions) / len(questions),
            "exact": sum(set(q["required_chunks"]) == set(matched[q["id"]]) for q in questions) / len(questions),
            "precision": n_required / n_matched if n_matched else None,
            "avg_chunks": n_matched / len(questions),
            "misses": {q["id"]: {"required": q["required_chunks"], "matched": matched[q["id"]]} for q in questions
                       if set(q["required_chunks"]) != set(matched[q["id"]])},
            "original_n": len(originals),
            "original_with_definitions": {k: v for k, v in false_matches.items() if v}}


def main():
    questions, originals = load(QUESTIONS_PATH), load(ORIGINAL_PATH)
    variants = {"": False, "+aliases": True}
    retrievers = {(name, suffix): GlossaryRetriever(name, index_aliases=flag)
                  for name in DATASETS for suffix, flag in variants.items()}

    per_question, results = [], {}
    for suffix in variants:
        for method in RETRIEVERS:
            rows = []
            for q in questions:
                r = retrievers[(q["dataset"], suffix)]
                ranked = [r.chunks[i]["id"] for i in r.rank(q["question"], method)]
                rows.append({"id": q["id"], "dataset": q["dataset"], "method": method + suffix,
                             "gold": q["required_chunks"], "ranked": ranked})
            per_question += rows
            results[method + suffix] = {"overall": metrics(rows),
                                        **{d: metrics([x for x in rows if x["dataset"] == d]) for d in DATASETS}}

    best = max(RETRIEVERS, key=lambda m: (results[m]["overall"]["all@3"], results[m]["overall"]["recall@3"]))
    gating = gating_metrics(questions, originals, {d: retrievers[(d, "")].chunks for d in DATASETS})
    report = {"config": {"embedding_model": EMBEDDING_MODEL, "rrf_k": RRF_K, "n_questions": len(questions),
                         "chunks_per_glossary": {d: len(retrievers[(d, "")].chunks) for d in DATASETS}},
              "results": results, "best_by_all@3": best, "gating": gating}
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
    for m in results:
        o = results[m]["overall"]
        lines.append(f"| {m} | {o['recall@1']:.1%} | **{o['recall@3']:.1%}** | {o['recall@5']:.1%} | "
                     f"{o['all@3']:.1%} | {o['all@5']:.1%} | {o['mrr']:.3f} | "
                     f"{results[m]['healthcare']['recall@3']:.1%} | {results[m]['retail']['recall@3']:.1%} |")
    g = gating
    lines += ["", f"Best retriever by all@3 (without aliases): **{best}** (used for doc_rag with k=3).", "",
              "## Term gating (doc_rag_gated)", "",
              "Gating returns every chunk whose term or alias appears in the question, and nothing otherwise. "
              "The dev aliases were written while looking at these questions, so these numbers are optimistic "
              "by construction; the held-out SaaS set is the real test.", "",
              "| Recall | All required chunks | Exact set | Precision | Chunks per question | Original questions given any definition |",
              "|---|---|---|---|---|---|",
              f"| {g['recall']:.1%} | {g['all_required']:.1%} | {g['exact']:.1%} | "
              f"{'n/a' if g['precision'] is None else format(g['precision'], '.1%')} | {g['avg_chunks']:.2f} | "
              f"{len(g['original_with_definitions'])} / {g['original_n']} |", ""]
    lines += [f"- {qid}: required {', '.join(v['required'])}; matched {', '.join(v['matched']) or 'nothing'}"
              for qid, v in g["misses"].items()]
    lines += [f"- original {qid}: given {', '.join(v)}" for qid, v in g["original_with_definitions"].items()]
    lines += ["",
              "## Questions with a required chunk missing from the top 3", "",
              "| Question | Retriever | Required | Retrieved top 3 |", "|---|---|---|---|"]
    lines += [f"| {x['id']} | {x['method']} | {', '.join(x['gold'])} | {', '.join(x['top3'])} |" for x in misses]
    md = "\n".join(lines) + "\n"
    (RESULTS_DIR / "doc_retrieval_results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
