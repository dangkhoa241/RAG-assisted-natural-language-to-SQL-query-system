"""Benchmark text-to-SQL systems by execution accuracy.

Systems:
  rule_based             - sql_builder.build_sql with the intent predicted by the fine-tuned BERT model
                           (what the app does today)
  rule_based_gold_intent - sql_builder.build_sql given the true intent (upper bound for the rule-based path)
  llm_zero_shot          - Groq LLM with the schema only
  llm_rag                - Groq LLM with the schema + k=3 retrieved examples
Plus an ablation of llm_rag over k = 0, 1, 3, 5 (k=0 sends exactly the llm_zero_shot prompt).

LLM systems run exactly as they would in production: if the LLM's SQL fails the safety check, the
rule-based SQL is used instead (counted as a fallback). Every LLM response is cached in
eval/results/llm_cache.jsonl, so re-runs make no API calls. If the daily Groq quota runs out
mid-run, the script stops; re-running later resumes from the cache.

Usage (from the repo root):  python eval/evaluate_sql.py
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "eval"))

from data_context import load_dataset  # noqa: E402
from rag_sql import (  # noqa: E402
    DEFAULT_K, EMBEDDING_MODEL, GROQ_MODEL, LLM_REASONING_EFFORT, DailyLimitError, LLMCache,
    build_schema_context, generate_sql_detailed,
)
from sql_builder import build_sql  # noqa: E402
from sql_metrics import results_match  # noqa: E402
from sql_safety import QueryTimeoutError, UnsafeSQLError, run_safe_query  # noqa: E402

BENCH_DIR = ROOT_DIR / "eval" / "sql_benchmark"
RESULTS_DIR = ROOT_DIR / "eval" / "results"
CACHE_PATH = RESULTS_DIR / "llm_cache.jsonl"
MODEL_DIR = ROOT_DIR / "intent_model"
DATASETS = {"healthcare": ROOT_DIR / "data" / "healthcare_dataset.csv", "retail": ROOT_DIR / "data" / "retail_sales.csv"}
INTENTS = ["filter", "count", "aggregate", "compare", "trend"]
MAIN_SYSTEMS = ["rule_based", "rule_based_gold_intent", "llm_zero_shot", "llm_rag"]
ABLATION_K = [0, 1, 3, 5]
HASH_SEED = "0"


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def predict_intents(questions):
    from transformers import pipeline

    clf = pipeline("text-classification", model=str(MODEL_DIR), tokenizer=str(MODEL_DIR), device=-1)
    return [p["label"] for p in clf([q["question"] for q in questions], batch_size=32)]


def execute(sql, dataset):
    """Run `sql` through the same guarded executor the app will use. Returns (df, error)."""
    try:
        return run_safe_query(sql, dataset.conn), None
    except (UnsafeSQLError, QueryTimeoutError, sqlite3.Error) as e:
        return None, f"{type(e).__name__}: {e}"
    except Exception as e:  # e.g. pandas failing on a malformed result
        return None, f"{type(e).__name__}: {e}"


def score(records):
    n = len(records)
    correct = sum(r["correct"] for r in records)
    return {
        "n": n,
        "n_correct": correct,
        "accuracy": correct / n if n else 0.0,
        "n_errors": sum(r["error"] is not None for r in records),
        "error_rate": sum(r["error"] is not None for r in records) / n if n else 0.0,
        "n_rejected": sum(r.get("rejected", False) for r in records),
        "n_fallback": sum(r.get("source") == "fallback" for r in records),
    }


def breakdown(records):
    return {
        "overall": score(records),
        "by_intent": {i: score([r for r in records if r["intent"] == i]) for i in INTENTS},
        "by_dataset": {d: score([r for r in records if r["dataset"] == d]) for d in DATASETS},
    }


def run_llm(name, questions, datasets, schemas, gold, bert_intents, cache, mode, k):
    records = []
    for q, intent in zip(questions, bert_intents):
        ds = datasets[q["dataset"]]
        gen = generate_sql_detailed(q["question"], ds, mode=mode, k=k, intent=intent, cache=cache,
                                    raise_on_quota=True, schema_context=schemas[q["dataset"]])
        df, err = execute(gen.sql, ds)
        records.append({
            "id": q["id"], "dataset": q["dataset"], "intent": q["intent"], "system": name,
            "sql": gen.sql, "llm_sql": gen.llm_sql, "source": gen.source, "rejected": gen.rejected,
            "error": err, "fallback_reason": gen.error,
            "correct": df is not None and results_match(gold[q["id"]], df, q["ordered"]),
            "retrieved": [e["id"] for e in gen.examples],
        })
    return records


def to_markdown(results):
    s = results["systems"]
    lines = ["# Text-to-SQL benchmark results", ""]
    cfg = results["config"]
    lines += [f"Model: `{cfg['llm_model']}` (reasoning effort {cfg['reasoning_effort']}, temperature 0). "
              f"Embeddings: `{cfg['embedding_model']}`. llm_rag uses k={cfg['rag_k']}. "
              f"{cfg['n_questions']} questions.", ""]

    lines += ["## Execution accuracy", "",
              "| System | Overall | Healthcare | Retail | SQL errors | Rejected by safety | Fallbacks |",
              "|---|---|---|---|---|---|---|"]
    for name in MAIN_SYSTEMS:
        b = s[name]
        o = b["overall"]
        lines.append(f"| {name} | **{o['accuracy']:.1%}** ({o['n_correct']}/{o['n']}) | "
                     f"{b['by_dataset']['healthcare']['accuracy']:.1%} | {b['by_dataset']['retail']['accuracy']:.1%} | "
                     f"{o['n_errors']} ({o['error_rate']:.1%}) | {o['n_rejected']} | {o['n_fallback']} |")

    lines += ["", "## Accuracy by intent", "", "| System | " + " | ".join(INTENTS) + " |", "|---|" + "---|" * len(INTENTS)]
    for name in MAIN_SYSTEMS:
        cells = [f"{s[name]['by_intent'][i]['accuracy']:.0%}" for i in INTENTS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", "## llm_rag ablation: number of retrieved examples", "",
              "| k | Overall | " + " | ".join(INTENTS) + " | SQL errors |", "|---|---|" + "---|" * len(INTENTS) + "---|"]
    for k, b in results["ablation"].items():
        cells = [f"{b['by_intent'][i]['accuracy']:.0%}" for i in INTENTS]
        lines.append(f"| {k} | **{b['overall']['accuracy']:.1%}** | " + " | ".join(cells) + f" | {b['overall']['n_errors']} |")
    lines += ["", "k=0 sends exactly the llm_zero_shot prompt, so its row matches llm_zero_shot.", ""]
    return "\n".join(lines)


def main():
    # sql_builder picks the date column with next(iter(set)), whose order depends on Python's
    # per-process string hashing. Pin the hash seed so the rule-based numbers are reproducible.
    if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
        env = dict(os.environ, PYTHONHASHSEED=HASH_SEED)
        sys.exit(subprocess.run([sys.executable, *sys.argv], env=env).returncode)
    if not MODEL_DIR.exists():
        sys.exit(f"Trained intent model not found at {MODEL_DIR}. Run src/model_training.ipynb first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_jsonl(BENCH_DIR / "test_questions.jsonl")
    datasets = {name: load_dataset(path) for name, path in DATASETS.items()}
    schemas = {name: build_schema_context(ds) for name, ds in datasets.items()}
    gold = {q["id"]: run_safe_query(q["gold_sql"], datasets[q["dataset"]].conn) for q in questions}
    bert_intents = predict_intents(questions)
    cache = LLMCache(CACHE_PATH)
    cached_before = len(cache.entries)

    records = defaultdict(list)
    for q, bert_intent in zip(questions, bert_intents):
        ds = datasets[q["dataset"]]
        for name, intent in [("rule_based", bert_intent), ("rule_based_gold_intent", q["intent"])]:
            sql = build_sql(q["question"], intent, ds)[1]
            df, err = execute(sql, ds)
            records[name].append({
                "id": q["id"], "dataset": q["dataset"], "intent": q["intent"], "system": name,
                "sql": sql, "predicted_intent": intent, "source": "rule_based", "error": err,
                "correct": df is not None and results_match(gold[q["id"]], df, q["ordered"]),
            })

    # Main systems first, so a quota stop still leaves the headline comparison complete.
    runs = [("llm_zero_shot", "llm_zero_shot", 0), (f"llm_rag_k{DEFAULT_K}", "llm_rag", DEFAULT_K)]
    runs += [(f"llm_rag_k{k}", "llm_rag", k) for k in ABLATION_K if k != DEFAULT_K]
    start = time.time()
    try:
        for name, mode, k in runs:
            print(f"Running {name} ...", flush=True)
            records[name] = run_llm(name, questions, datasets, schemas, gold, bert_intents, cache, mode, k)
    except DailyLimitError as e:
        done = len(cache.entries) - cached_before
        sys.exit(f"\nGroq daily quota exhausted after {done} new calls ({e}).\n"
                 "Responses so far are cached; re-run this script later to resume.")
    records["llm_rag"] = records[f"llm_rag_k{DEFAULT_K}"]
    new_calls = len(cache.entries) - cached_before

    results = {
        "config": {"llm_model": GROQ_MODEL, "reasoning_effort": LLM_REASONING_EFFORT,
                   "embedding_model": EMBEDDING_MODEL, "rag_k": DEFAULT_K, "n_questions": len(questions),
                   "new_llm_calls_this_run": new_calls, "python_hash_seed": HASH_SEED},
        "systems": {name: breakdown(records[name]) for name in MAIN_SYSTEMS},
        "ablation": {str(k): breakdown(records[f"llm_rag_k{k}"]) for k in ABLATION_K},
        "bert_intent_accuracy_on_benchmark": sum(b == q["intent"] for b, q in zip(bert_intents, questions)) / len(questions),
    }
    (RESULTS_DIR / "sql_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = to_markdown(results)
    (RESULTS_DIR / "sql_results.md").write_text(md + "\n", encoding="utf-8")

    rows = []
    for i, q in enumerate(questions):
        recs = {name: records[name][i] for name in MAIN_SYSTEMS}
        if all(r["correct"] for r in recs.values()):
            continue
        row = {"id": q["id"], "dataset": q["dataset"], "intent": q["intent"], "question": q["question"],
               "gold_sql": q["gold_sql"], "bert_intent": bert_intents[i]}
        for name, r in recs.items():
            row[f"{name}_correct"] = r["correct"]
            row[f"{name}_sql"] = " ".join(r["sql"].split())
            row[f"{name}_error"] = r["error"] or r.get("fallback_reason") or ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "sql_failures.csv", index=False, encoding="utf-8")

    print(md)
    print(f"{new_calls} new LLM calls this run ({time.time() - start:.0f}s); "
          f"{len(rows)} questions failed by at least one system -> {RESULTS_DIR / 'sql_failures.csv'}")


if __name__ == "__main__":
    main()
