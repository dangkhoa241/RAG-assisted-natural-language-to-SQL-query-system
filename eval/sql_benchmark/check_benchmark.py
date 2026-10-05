"""Sanity checks for the SQL benchmark. Fails loudly if any check doesn't hold.

1. Every gold query runs and returns a non-empty result (filters: at most 200 rows, the
   rule-based generator's LIMIT). Ordered top-N questions have no tie at the LIMIT boundary.
2. No example-bank question is a near-duplicate of a test question:
   embedding cosine similarity <= 0.9 and difflib ratio <= 0.85.
Writes eval/results/sql_benchmark_checks.json.

Usage (from the repo root):  python eval/sql_benchmark/check_benchmark.py
"""
import difflib
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from data_context import load_dataset  # noqa: E402
from rag_sql import EMBEDDING_MODEL  # noqa: E402

BENCH_DIR = ROOT_DIR / "eval" / "sql_benchmark"
DATASETS = {"healthcare": ROOT_DIR / "data" / "healthcare_dataset.csv", "retail": ROOT_DIR / "data" / "retail_sales.csv"}
COSINE_LIMIT, DIFFLIB_LIMIT = 0.9, 0.85


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def check_gold(questions, datasets):
    problems = []
    for q in questions:
        conn = datasets[q["dataset"]].conn
        rows = conn.execute(q["gold_sql"]).fetchall()
        if not rows:
            problems.append(f"{q['id']}: empty result")
        if q["intent"] == "filter" and len(rows) > 200:
            problems.append(f"{q['id']}: {len(rows)} rows (> 200)")
        if q["intent"] == "count" and rows[0][0] == 0:
            problems.append(f"{q['id']}: count is 0")
        m = re.search(r"ORDER BY (`[^`]+`) (ASC|DESC) LIMIT (\d+)$", q["gold_sql"])
        if m:
            # The next row past the LIMIT must not tie with the last returned row.
            col, direction, n = m.group(1), m.group(2), int(m.group(3))
            base = q["gold_sql"][: m.start()].replace("SELECT *", f"SELECT {col}", 1)
            vals = [r[0] for r in conn.execute(f"{base} ORDER BY {col} {direction} LIMIT {n + 1}").fetchall()]
            if len(vals) > n and vals[n - 1] == vals[n]:
                problems.append(f"{q['id']}: tie at the LIMIT boundary")
    return problems


def check_leakage(questions, bank):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    tq = [q["question"] for q in questions]
    bq = [b["question"] for b in bank]
    sims = model.encode(tq, normalize_embeddings=True) @ model.encode(bq, normalize_embeddings=True).T

    i, j = divmod(int(sims.argmax()), sims.shape[1])
    best_cos = {"similarity": round(float(sims[i, j]), 4), "test_question": tq[i], "bank_question": bq[j]}

    best_ratio = {"similarity": 0.0}
    for t in tq:
        for b in bq:
            r = difflib.SequenceMatcher(None, t.lower(), b.lower()).ratio()
            if r > best_ratio["similarity"]:
                best_ratio = {"similarity": round(r, 4), "test_question": t, "bank_question": b}

    flagged = int((sims > COSINE_LIMIT).sum())
    return best_cos, best_ratio, flagged


def main():
    questions = load_jsonl(BENCH_DIR / "test_questions.jsonl")
    bank = load_jsonl(BENCH_DIR / "example_bank.jsonl")
    datasets = {name: load_dataset(path) for name, path in DATASETS.items()}

    problems = check_gold(questions, datasets)
    best_cos, best_ratio, flagged = check_leakage(questions, bank)
    if best_cos["similarity"] > COSINE_LIMIT or best_ratio["similarity"] > DIFFLIB_LIMIT:
        problems.append(f"near-duplicate between bank and test set: {best_cos} / {best_ratio}")

    report = {
        "n_test_questions": len(questions),
        "n_bank_examples": len(bank),
        "gold_queries_ok": not any("rows" in p or "empty" in p or "count" in p or "tie" in p for p in problems),
        "max_embedding_cosine": best_cos,
        "max_difflib_ratio": best_ratio,
        "pairs_above_cosine_limit": flagged,
        "limits": {"cosine": COSINE_LIMIT, "difflib": DIFFLIB_LIMIT},
        "problems": problems,
    }
    out = ROOT_DIR / "eval" / "results" / "sql_benchmark_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
