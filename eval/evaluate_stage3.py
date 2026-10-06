"""Stage 3 benchmark: does retrieval help when the LLM lacks business knowledge, and for a smaller model?

Systems (each run with every model in MODELS):
  glossary set (60 questions that need glossary definitions)
    zero_shot          - schema only
    doc_rag            - schema + top-3 glossary chunks (hybrid dense+BM25 retriever)
    oracle_doc         - schema + exactly the required chunks (upper bound for doc_rag)
  original set (the 120 Stage 2 questions)
    zero_shot          - schema only (identical prompt to Stage 2's llm_zero_shot, so gpt-oss-120b is cached)
    example_rag_fixed  - schema + up to 3 examples (own table names, intent-filtered, similarity threshold)
    doc_rag            - schema + top-3 glossary chunks (checks that irrelevant definitions don't hurt)

Models and providers: gpt-oss-20b runs on Groq, gpt-oss-120b on Cerebras (Groq's free-tier daily cap made the
120b run impractical there). Every response is cached per model + provider in its own file, so identical prompts
are never sent twice and re-runs are free. The provider is part of the cache key.

Usage (from the repo root):
  python eval/evaluate_stage3.py run --model gpt-oss-20b|gpt-oss-120b [--wait] [--limit N] [--set S] [--system X]
      Make the LLM calls for one model. Without --wait it stops when the daily quota runs out (re-run later
      to resume); with --wait it sleeps until the quota frees up and carries on. Every new call is logged with
      its tokens, and the total tokens used are printed at the end. --limit/--set/--system restrict the run
      (used for smoke tests).
  python eval/evaluate_stage3.py report
      Score everything from the caches (no API calls) and write eval/results/stage3_results.{json,md}
      and stage3_failures.csv. Questions whose responses aren't cached yet are reported as missing.
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "eval"))

import rag_sql  # noqa: E402
from data_context import load_dataset  # noqa: E402
from doc_retrieval import load_glossary  # noqa: E402
from evaluate_sql import execute, load_jsonl, predict_intents  # noqa: E402
from rag_sql import (DEFAULT_K, DOC_K, DOC_RETRIEVER, EMBEDDING_MODEL, EXAMPLE_MIN_SCORE,  # noqa: E402
                     LLM_REASONING_EFFORT, DailyLimitError, LLMCache, LLMError, build_schema_context,
                     generate_sql_detailed)
from sql_metrics import results_match  # noqa: E402
from sql_safety import run_safe_query  # noqa: E402

BENCH_DIR = ROOT_DIR / "eval" / "sql_benchmark"
RESULTS_DIR = ROOT_DIR / "eval" / "results"
DATASETS = {"healthcare": ROOT_DIR / "data" / "healthcare_dataset.csv", "retail": ROOT_DIR / "data" / "retail_sales.csv"}
INTENTS = ["filter", "count", "aggregate", "compare", "trend"]
# label -> (provider, provider's model ID, cache file); small first
MODELS = {"gpt-oss-20b": ("groq", "openai/gpt-oss-20b", "llm_cache_gpt-oss-20b.jsonl"),
          "gpt-oss-120b": ("cerebras", "gpt-oss-120b", "llm_cache_gpt-oss-120b_cerebras.jsonl")}
MODEL_ALIASES = {"openai/gpt-oss-20b": "gpt-oss-20b"}  # the 20b run was started with the Groq model ID
PROVIDER_NAMES = {"groq": "Groq", "cerebras": "Cerebras"}
SYSTEMS = {"glossary": ["zero_shot", "doc_rag", "oracle_doc"],
           "original": ["zero_shot", "example_rag_fixed", "doc_rag"]}
OFFLINE = "offline: response not cached"


def cache_path(label):
    return RESULTS_DIR / MODELS[label][2]


def provider_of(label):
    return PROVIDER_NAMES[MODELS[label][0]]


def setup():
    sets = {"glossary": load_jsonl(BENCH_DIR / "glossary_questions.jsonl"),
            "original": load_jsonl(BENCH_DIR / "test_questions.jsonl")}
    datasets = {name: load_dataset(path) for name, path in DATASETS.items()}
    schemas = {name: build_schema_context(ds) for name, ds in datasets.items()}
    chunks = {c["id"]: c for name in DATASETS for c in load_glossary(name)}
    gold = {(s, q["id"]): run_safe_query(q["gold_sql"], datasets[q["dataset"]].conn) for s, qs in sets.items() for q in qs}
    bert = {s: predict_intents(qs) for s, qs in sets.items()}
    return sets, datasets, schemas, chunks, gold, bert


def generate(system, q, bert_intent, ds, schema, chunks, cache, label):
    provider, model, _ = MODELS[label]
    common = dict(intent=bert_intent, cache=cache, raise_on_quota=True, schema_context=schema, model=model,
                  provider=provider)
    if system == "zero_shot":
        return generate_sql_detailed(q["question"], ds, mode="llm_zero_shot", **common)
    if system == "example_rag_fixed":
        return generate_sql_detailed(q["question"], ds, mode="llm_rag", k=DEFAULT_K, **common)
    if system == "doc_rag":
        return generate_sql_detailed(q["question"], ds, mode="llm_doc_rag", k=DOC_K, dataset_name=q["dataset"],
                                     doc_method=DOC_RETRIEVER, **common)
    if system == "oracle_doc":
        return generate_sql_detailed(q["question"], ds, mode="llm_doc_rag", docs=[chunks[c] for c in q["required_chunks"]],
                                     **common)
    raise ValueError(system)


def label_only_miss(gold_df, pred_df, ordered):
    """The prediction is right except that it left out the gold's label column(s)."""
    if pred_df is None or gold_df.shape[1] < 2 or pred_df.shape[1] >= gold_df.shape[1]:
        return False
    return results_match(gold_df.iloc[:, -1:], pred_df, ordered)  # the metric is the last gold column


def run(label, wait, limit=None, only_set=None, only_system=None):
    sets, datasets, schemas, chunks, gold, bert = setup()
    cache = LLMCache(cache_path(label))
    tag = f"{label} on {provider_of(label)}"
    totals = defaultdict(int)

    def summary():
        return (f"{totals['calls']} new calls, {totals['total']:,} tokens "
                f"(prompt {totals['prompt']:,}, cached prompt {totals['cached']:,}, "
                f"completion {totals['completion']:,}, of which reasoning {totals['reasoning']:,})")

    for set_name, systems in SYSTEMS.items():
        if only_set and set_name != only_set:
            continue
        for system in systems:
            if only_system and system != only_system:
                continue
            print(f"[{tag}] {set_name}/{system} ...", flush=True)
            for q, bert_intent in list(zip(sets[set_name], bert[set_name]))[:limit]:
                ds = datasets[q["dataset"]]
                n_cached = len(cache.entries)
                while True:
                    try:
                        gen = generate(system, q, bert_intent, ds, schemas[q["dataset"]], chunks, cache, label)
                        break
                    except DailyLimitError as e:
                        if not wait:
                            sys.exit(f"Daily quota for {tag} exhausted. Cached so far; re-run later to resume.\n"
                                     f"{e}\nSo far: {summary()}")
                        pause = (e.retry_after_s or 900) + 30
                        print(f"  daily quota reached ({summary()}); sleeping {pause / 60:.1f} min "
                              f"(until {time.strftime('%H:%M', time.localtime(time.time() + pause))})", flush=True)
                        time.sleep(pause)
                if len(cache.entries) == n_cached:
                    continue  # cache hit, no API call
                entry = cache.entries[gen.cache_key]
                u = entry.get("usage") or {}
                reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                totals["calls"] += 1
                totals["prompt"] += u.get("prompt_tokens", 0)
                totals["cached"] += cached
                totals["completion"] += u.get("completion_tokens", 0)
                totals["reasoning"] += reasoning
                totals["total"] += u.get("total_tokens", 0)
                print(f"  {time.strftime('%H:%M:%S')} #{totals['calls']} {q['id']}: prompt {u.get('prompt_tokens')} "
                      f"(cached {cached}), completion {u.get('completion_tokens')} (reasoning {reasoning}), "
                      f"total {u.get('total_tokens')}, {entry.get('latency_s')}s | running total "
                      f"{totals['total']:,} tokens", flush=True)
    print(f"[{tag}] done: {summary()}", flush=True)


def score(records):
    done = [r for r in records if not r["missing"]]
    n = len(done)
    return {"n": n, "n_missing": len(records) - n, "n_correct": sum(r["correct"] for r in done),
            "accuracy": sum(r["correct"] for r in done) / n if n else None,
            "n_errors": sum(r["error"] is not None for r in done),
            "n_fallback": sum(r["source"] == "fallback" for r in done),
            "n_label_only_miss": sum(r["label_only_miss"] for r in done)}


def pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def report():
    def offline_client(provider):
        raise LLMError(OFFLINE)
    rag_sql._client_for = offline_client  # scoring must never call the API

    sets, datasets, schemas, chunks, gold, bert = setup()
    records = []
    usage = {}
    for model in MODELS:
        cache = LLMCache(cache_path(model))
        foreign = {e.get("provider", "groq") for e in cache.entries.values()} - {MODELS[model][0]}
        assert not foreign, f"{cache_path(model).name} has responses from another provider: {foreign}"
        calls = {}
        for set_name, systems in SYSTEMS.items():
            for system in systems:
                for q, bert_intent in zip(sets[set_name], bert[set_name]):
                    ds = datasets[q["dataset"]]
                    gen = generate(system, q, bert_intent, ds, schemas[q["dataset"]], chunks, cache, model)
                    missing = gen.error is not None and OFFLINE in gen.error
                    df, err = (None, None) if missing else execute(gen.sql, ds)
                    g = gold[(set_name, q["id"])]
                    entry = cache.entries.get(gen.cache_key) or {}
                    if entry:
                        calls[gen.cache_key] = entry
                    retrieved = [d["id"] for d in gen.docs] if gen.docs else [e["id"] for e in gen.examples]
                    req = q.get("required_chunks", [])
                    records.append({
                        "set": set_name, "system": system, "model": model, "provider": provider_of(model),
                        "id": q["id"],
                        "dataset": q["dataset"], "intent": q["intent"], "question": q["question"],
                        "gold_sql": q["gold_sql"], "sql": " ".join(gen.sql.split()), "source": gen.source,
                        "missing": missing, "error": err, "fallback_reason": None if missing else gen.error,
                        "correct": (not missing) and df is not None and results_match(g, df, q["ordered"]),
                        "label_only_miss": (not missing) and df is not None and not results_match(g, df, q["ordered"])
                                           and label_only_miss(g, df, q["ordered"]),
                        "retrieved": retrieved, "required_chunks": req,
                        "retrieval_hit": bool(req) and set(req) <= set(retrieved) if system == "doc_rag" else None,
                        "naive_match": (not missing) and df is not None and "naive_sql" in q
                                       and results_match(run_safe_query(q["naive_sql"], ds.conn), df, q["ordered"]),
                    })
        usage[model] = dict(usage_stats(calls.values()), provider=provider_of(model))

    by = defaultdict(list)
    for r in records:
        by[(r["set"], r["system"], r["model"])].append(r)
    table = {}
    for (set_name, system, model), rs in by.items():
        table.setdefault(set_name, {}).setdefault(system, {})[model] = {
            "overall": score(rs),
            "by_dataset": {d: score([r for r in rs if r["dataset"] == d]) for d in DATASETS},
            "by_intent": {i: score([r for r in rs if r["intent"] == i]) for i in INTENTS},
        }
    doc_rag_failures = {}
    for model in MODELS:
        rs = [r for r in by[("glossary", "doc_rag", model)] if not r["missing"] and not r["correct"]]
        doc_rag_failures[model] = {"retrieval_miss": sum(not r["retrieval_hit"] for r in rs),
                                   "generation_miss": sum(bool(r["retrieval_hit"]) for r in rs)}
    small, large = list(MODELS)
    paired = {}
    for set_name, systems in SYSTEMS.items():
        for system in systems:
            s_rs = {r["id"]: r for r in by[(set_name, system, small)] if not r["missing"]}
            l_rs = {r["id"]: r for r in by[(set_name, system, large)] if not r["missing"]}
            ids = s_rs.keys() & l_rs.keys()
            cnt = lambda a, b: sum(s_rs[i]["correct"] == a and l_rs[i]["correct"] == b for i in ids)
            paired.setdefault(set_name, {})[system] = {"n": len(ids), "both": cnt(True, True), "only_small": cnt(True, False),
                                                       "only_large": cnt(False, True), "neither": cnt(False, False)}
    naive = {m: {"wrong": len(rs := [r for r in by[("glossary", "zero_shot", m)] if not r["missing"] and not r["correct"]]),
                 "naive_match": sum(r["naive_match"] for r in rs)} for m in MODELS}
    results = {
        "config": {"models": {m: {"provider": provider_of(m), "model_id": MODELS[m][1]} for m in MODELS},
                   "reasoning_effort": LLM_REASONING_EFFORT, "temperature": 0,
                   "embedding_model": EMBEDDING_MODEL, "doc_retriever": DOC_RETRIEVER, "doc_k": DOC_K,
                   "example_k": DEFAULT_K, "example_min_score": EXAMPLE_MIN_SCORE,
                   "n_questions": {s: len(qs) for s, qs in sets.items()}},
        "accuracy": table, "usage": usage, "doc_rag_glossary_failures": doc_rag_failures,
        "paired_small_vs_large": paired, "glossary_zero_shot_naive": naive,
        "n_missing": sum(r["missing"] for r in records),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "stage3_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fails = [r for r in records if not r["correct"] and not r["missing"]]
    pd.DataFrame(fails).drop(columns=["missing", "correct"]).to_csv(RESULTS_DIR / "stage3_failures.csv",
                                                                     index=False, encoding="utf-8")
    md = to_markdown(results)
    (RESULTS_DIR / "stage3_results.md").write_text(md, encoding="utf-8")
    print(md)


def usage_stats(entries):
    entries = list(entries)
    u = [e["usage"] for e in entries if e.get("usage")]
    lat = [e["latency_s"] for e in entries if e.get("latency_s") is not None]
    # server-side time: Groq reports it in usage.total_time, Cerebras in time_info.total_time
    server = [t for e in entries
              if (t := ((e.get("usage") or {}).get("total_time") or (e.get("time_info") or {}).get("total_time")))
              is not None]
    reasoning = [(x.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for x in u]

    def q(xs, p):
        return float(np.percentile(xs, p)) if xs else None
    return {"n_calls": len(entries), "n_with_client_latency": len(lat),
            "latency_median_s": q(lat, 50), "latency_p95_s": q(lat, 95),
            "server_time_median_s": q(server, 50), "server_time_p95_s": q(server, 95),
            "avg_prompt_tokens": statistics.mean(x["prompt_tokens"] for x in u) if u else None,
            "avg_completion_tokens": statistics.mean(x["completion_tokens"] for x in u) if u else None,
            "avg_reasoning_tokens": statistics.mean(reasoning) if reasoning else None,
            "avg_total_tokens": statistics.mean(x["total_tokens"] for x in u) if u else None}


def to_markdown(res):
    acc, models = res["accuracy"], list(MODELS)
    name = lambda m: f"{m} ({provider_of(m)})"
    L = ["# Stage 3 results: glossary RAG and model size", ""]
    if res["n_missing"]:
        L += [f"**Incomplete: {res['n_missing']} responses are not cached yet.** Accuracies below cover answered questions only.", ""]
    L += [f"Models: {', '.join(f'`{MODELS[m][1]}` on {provider_of(m)}' for m in MODELS)} "
          f"(reasoning effort {res['config']['reasoning_effort']}, temperature 0). Every response for a model comes "
          f"from the one provider listed. "
          f"doc_rag: top-{res['config']['doc_k']} glossary chunks, {res['config']['doc_retriever']} retriever. "
          f"example_rag_fixed: up to {res['config']['example_k']} intent-filtered examples with cosine >= "
          f"{res['config']['example_min_score']}.", ""]
    for set_name, title in [("glossary", "Glossary set (60 questions that need business definitions)"),
                            ("original", "Original set (120 Stage 2 questions)")]:
        L += [f"## {title}", "",
              "| System | " + " | ".join(f"{name(m)} overall | {m} healthcare | {m} retail" for m in models) + " |",
              "|---|" + "---|---|---|" * len(models)]
        for system in SYSTEMS[set_name]:
            cells = []
            for m in models:
                b = acc[set_name][system][m]
                o = b["overall"]
                cells += [f"**{pct(o['accuracy'])}** ({o['n_correct']}/{o['n']})",
                          pct(b["by_dataset"]["healthcare"]["accuracy"]), pct(b["by_dataset"]["retail"]["accuracy"])]
            L.append(f"| {system} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Glossary set by intent", "", "| System | Model | " + " | ".join(INTENTS) + " |", "|---|---|" + "---|" * len(INTENTS)]
    for system in SYSTEMS["glossary"]:
        for m in models:
            L.append(f"| {system} | {name(m)} | " + " | ".join(pct(acc['glossary'][system][m]["by_intent"][i]["accuracy"])
                                                        for i in INTENTS) + " |")
    L += ["", "## Headline: small model + RAG vs. large model zero-shot", "",
          f"| Set | {name('gpt-oss-20b')} + doc_rag | gpt-oss-20b + oracle_doc | {name('gpt-oss-120b')} zero_shot "
          f"| gpt-oss-120b + doc_rag |",
          "|---|---|---|---|---|"]
    for set_name in ["glossary", "original"]:
        a = lambda s, m: pct(acc[set_name][s][m]["overall"]["accuracy"]) if s in acc[set_name] else "n/a"
        L.append(f"| {set_name} | {a('doc_rag', 'gpt-oss-20b')} | {a('oracle_doc', 'gpt-oss-20b')} | "
                 f"{a('zero_shot', 'gpt-oss-120b')} | {a('doc_rag', 'gpt-oss-120b')} |")
    small, large = models
    L += ["", f"## Small vs. large, question by question ({small} vs. {large})", "",
          "Only questions both models answered are counted. \"Only small\" means the small model was right and the large "
          "model wrong.", "",
          f"| Set | System | n | {small} | {large} | Both right | Only small | Only large | Both wrong |",
          "|---|---|---|---|---|---|---|---|---|"]
    for set_name, systems in res["paired_small_vs_large"].items():
        for system, p in systems.items():
            n = p["n"] or None
            L.append(f"| {set_name} | {system} | {p['n']} | {pct(n and (p['both'] + p['only_small']) / n)} | "
                     f"{pct(n and (p['both'] + p['only_large']) / n)} | {p['both']} | {p['only_small']} | "
                     f"{p['only_large']} | {p['neither']} |")
    L += ["", "## Zero-shot on the glossary set: wrong answers that took the literal reading", "",
          "A naive match means the wrong answer returned exactly what the hand-written literal reading of the term "
          "returns (`naive_sql`), e.g. treating \"senior patient\" as age >= 65 instead of the glossary's cut-off.", "",
          "| Model | Wrong answers | Matched the naive reading |", "|---|---|---|"]
    for m, v in res["glossary_zero_shot_naive"].items():
        L.append(f"| {name(m)} | {v['wrong']} | {v['naive_match']} |")
    L += ["", "## doc_rag failures on the glossary set", "",
          "A retrieval miss means at least one required chunk was not in the top 3; a generation miss means "
          "all required chunks were in the prompt and the SQL was still wrong.", "",
          "| Model | Retrieval misses | Generation misses |", "|---|---|---|"]
    for m, f in res["doc_rag_glossary_failures"].items():
        L.append(f"| {name(m)} | {f['retrieval_miss']} | {f['generation_miss']} |")
    L += ["", "## Latency and tokens per call", "",
          "Client latency is the wall time of each successful request (rate-limit waits and throttling excluded); "
          "server time is the provider's reported `total_time`. **The two models ran on different providers, so "
          "latency is not comparable between them**: it measures Groq vs. Cerebras hardware as much as model size.", "",
          "| Model | Provider | Calls | Client median | Client p95 | Server median | Server p95 | Prompt tokens | Completion tokens (reasoning) |",
          "|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, d=2: "n/a" if x is None else f"{x:.{d}f}"
    for m, u in res["usage"].items():
        L.append(f"| {m} | {u['provider']} | {u['n_calls']} | {f(u['latency_median_s'])}s | {f(u['latency_p95_s'])}s | "
                 f"{f(u['server_time_median_s'])}s | {f(u['server_time_p95_s'])}s | {f(u['avg_prompt_tokens'], 0)} | "
                 f"{f(u['avg_completion_tokens'], 0)} ({f(u['avg_reasoning_tokens'], 0)}) |")
    L += ["", "## Notes", "",
          "- Errors, safety rejections and fallbacks per system are in `stage3_results.json`; every wrong answer is in "
          "`stage3_failures.csv` (with retrieved chunk IDs and whether it matched the naive-guess SQL).",
          f"- Answers that were right except for leaving out a gold label column count as wrong "
          f"(see tests/test_result_compare.py); there were "
          f"{sum(b['overall']['n_label_only_miss'] for s in acc.values() for x in s.values() for b in x.values())} such answers.",
          ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=list(MODELS) + list(MODEL_ALIASES))
    r.add_argument("--wait", action="store_true", help="sleep through daily-quota limits instead of stopping")
    r.add_argument("--limit", type=int, help="only the first N questions of each set (smoke tests)")
    r.add_argument("--set", choices=list(SYSTEMS), help="only this question set")
    r.add_argument("--system", help="only this system")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "run":
        run(MODEL_ALIASES.get(args.model, args.model), args.wait, args.limit, args.set, args.system)
    else:
        report()


if __name__ == "__main__":
    main()
