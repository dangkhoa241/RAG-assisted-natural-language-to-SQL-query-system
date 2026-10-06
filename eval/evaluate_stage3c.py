"""Stage 3C: term-gated glossary RAG, tuned on the dev domains and tested once on a held-out SaaS domain.

Parts:
  dev  - healthcare + retail (the only data used for tuning): the 60 glossary questions and the 120 original
         questions, with zero_shot, doc_rag (ungated top-3, as in Stage 3B), doc_rag_gated and oracle_doc.
         Everything but doc_rag_gated is already in the Stage 3B caches.
  saas - the held-out SaaS questions (40 glossary + 20 plain), run once with the settings frozen in
         eval/stage3c_config.json: zero_shot, doc_rag and doc_rag_gated on all 60, oracle_doc on the 40.

Systems:
  zero_shot      - schema only
  doc_rag        - schema + top-k glossary chunks (hybrid retriever), whether or not the question needs them
  doc_rag_gated  - schema + only the chunks whose term or alias appears in the question (doc_retrieval.match_terms);
                   with no match the prompt is byte-identical to zero_shot, so it reuses that cached response
  oracle_doc     - schema + exactly the required chunks (glossary questions only)

LLM responses share the Stage 3B per-model caches (provider + model are in the cache key), so any byte-identical
prompt is never sent twice.

Usage (from the repo root):
  python eval/evaluate_stage3c.py estimate --part dev|saas        # calls and tokens still needed (no API calls)
  python eval/evaluate_stage3c.py run --model gpt-oss-20b|gpt-oss-120b --part dev|saas [--wait]
  python eval/evaluate_stage3c.py report --part dev|saas          # score from the caches (no API calls)
  python eval/evaluate_stage3c.py bert                            # BERT intent accuracy on the SaaS questions
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "eval"))

import rag_sql  # noqa: E402
from data_context import load_dataset  # noqa: E402
from doc_retrieval import load_glossary  # noqa: E402
from evaluate_sql import execute, load_jsonl, predict_intents  # noqa: E402
from evaluate_stage3 import (MODELS, OFFLINE, cache_path, label_only_miss, pct, provider_of,  # noqa: E402
                             usage_stats)
from rag_sql import DailyLimitError, LLMCache, LLMError, build_schema_context, generate_sql_detailed  # noqa: E402
from sql_metrics import results_match  # noqa: E402
from sql_safety import run_safe_query  # noqa: E402

BENCH_DIR = ROOT_DIR / "eval" / "sql_benchmark"
RESULTS_DIR = ROOT_DIR / "eval" / "results"
LOG_DIR = ROOT_DIR / "eval" / "logs"
CONFIG = json.loads((ROOT_DIR / "eval" / "stage3c_config.json").read_text(encoding="utf-8"))
DATA = {"healthcare": ROOT_DIR / "data" / "healthcare_dataset.csv", "retail": ROOT_DIR / "data" / "retail_sales.csv",
        "saas": ROOT_DIR / "data" / "saas_subscriptions.csv"}
INTENTS = ["filter", "count", "aggregate", "compare", "trend"]
PARTS = {
    "dev": {"glossary": ["zero_shot", "doc_rag", "doc_rag_gated", "oracle_doc"],
            "original": ["zero_shot", "doc_rag", "doc_rag_gated"]},
    "saas": {"saas_glossary": ["zero_shot", "doc_rag", "doc_rag_gated", "oracle_doc"],
             "saas_plain": ["zero_shot", "doc_rag", "doc_rag_gated"]},
}
SET_TITLES = {"glossary": "Dev glossary set (60 healthcare + retail questions that need a definition)",
              "original": "Dev original set (120 Stage 2 questions, no definitions needed)",
              "saas_glossary": "SaaS glossary set (40 held-out questions that need a definition)",
              "saas_plain": "SaaS plain set (20 held-out questions, no definitions needed)"}
# Stage 3B's observed Cerebras price for gpt-oss-120b: 457,117 tokens cost $0.18 in credits.
CEREBRAS_USD_PER_TOKEN = 0.18 / 457_117


# --- Setup -----------------------------------------------------------------------------------------------
def load_sets(part):
    if part == "dev":
        return {"glossary": load_jsonl(BENCH_DIR / "glossary_questions.jsonl"),
                "original": load_jsonl(BENCH_DIR / "test_questions.jsonl")}
    saas = load_jsonl(BENCH_DIR / "saas_questions.jsonl")
    return {"saas_glossary": [q for q in saas if q["set"] == "glossary"],
            "saas_plain": [q for q in saas if q["set"] == "plain"]}


def setup(part):
    sets = load_sets(part)
    names = sorted({q["dataset"] for qs in sets.values() for q in qs})
    datasets = {n: load_dataset(DATA[n]) for n in names}
    schemas = {n: build_schema_context(ds) for n, ds in datasets.items()}
    chunks = {c["id"]: c for n in names for c in load_glossary(n)}
    bert = {s: predict_intents(qs) for s, qs in sets.items()}
    items = [{"set": s, "system": system, "q": q, "bert": b}
             for s, systems in PARTS[part].items() for system in systems for q, b in zip(sets[s], bert[s])]
    return sets, datasets, schemas, chunks, items


def generate(item, datasets, schemas, chunks, cache, label):
    provider, model, _ = MODELS[label]
    q, system = item["q"], item["system"]
    common = dict(intent=item["bert"], cache=cache, raise_on_quota=True, schema_context=schemas[q["dataset"]],
                  model=model, provider=provider)
    ds = datasets[q["dataset"]]
    if system == "zero_shot":
        return generate_sql_detailed(q["question"], ds, mode="llm_zero_shot", **common)
    if system == "doc_rag":
        assert not CONFIG["doc_rag"]["index_aliases"]
        return generate_sql_detailed(q["question"], ds, mode="llm_doc_rag", k=CONFIG["doc_rag"]["k"],
                                     dataset_name=q["dataset"], doc_method=CONFIG["doc_rag"]["retriever"], **common)
    if system == "doc_rag_gated":
        assert CONFIG["doc_rag_gated"]["retrieval_fallback"] is None and CONFIG["doc_rag_gated"]["max_chunks"] is None
        return generate_sql_detailed(q["question"], ds, mode="llm_doc_rag", dataset_name=q["dataset"],
                                     doc_method=CONFIG["doc_rag_gated"]["doc_method"], **common)
    if system == "oracle_doc":
        return generate_sql_detailed(q["question"], ds, mode="llm_doc_rag",
                                     docs=[chunks[c] for c in q["required_chunks"]], **common)
    raise ValueError(system)


class Offline:
    """Within this context no API call is made: cache misses fail with OFFLINE, and every prompt is recorded."""

    def __init__(self):
        self.prompts = {}

    def __enter__(self):
        self._call_llm, self._client_for = rag_sql.call_llm, rag_sql._client_for

        def offline_client(provider):
            raise LLMError(OFFLINE)

        def recording_call_llm(messages, cache=None, model=rag_sql.GROQ_MODEL, provider=rag_sql.DEFAULT_PROVIDER):
            self.prompts[rag_sql.cache_key(messages, model, provider)] = messages
            return self._call_llm(messages, cache=cache, model=model, provider=provider)

        rag_sql._client_for, rag_sql.call_llm = offline_client, recording_call_llm
        return self

    def __exit__(self, *exc):
        rag_sql.call_llm, rag_sql._client_for = self._call_llm, self._client_for


def prompt_keys(items, datasets, schemas, chunks, cache, label):
    """{cache key: messages} for every item, without calling the API."""
    with Offline() as off:
        for item in items:
            generate(item, datasets, schemas, chunks, cache, label)
    return off.prompts


# --- estimate --------------------------------------------------------------------------------------------
def estimate(part):
    _, datasets, schemas, chunks, items = setup(part)
    # Tokens per character of prompt, calibrated on the dev prompts already in each model's cache.
    cal = setup("dev") if part != "dev" else (None, datasets, schemas, chunks, items)
    print(f"Part {part}: {len(items)} (question, system) items")
    for label in MODELS:
        cache = LLMCache(cache_path(label))
        prompts = prompt_keys(items, datasets, schemas, chunks, cache, label)
        cal_prompts = prompt_keys(cal[4], cal[1], cal[2], cal[3], cache, label)
        hits = [(k, m) for k, m in cal_prompts.items() if k in cache.entries and (cache.entries[k].get("usage") or {})]
        tok_per_char = (sum(cache.entries[k]["usage"]["prompt_tokens"] for k, _ in hits)
                        / sum(len(json.dumps(m)) for _, m in hits))
        completion = sum(cache.entries[k]["usage"]["completion_tokens"] for k, _ in hits) / len(hits)
        new = {k: m for k, m in prompts.items() if k not in cache.entries}
        prompt_tokens = sum(len(json.dumps(m)) * tok_per_char for m in new.values())
        total = prompt_tokens + completion * len(new)
        line = (f"  {label} on {provider_of(label)}: {len(prompts)} unique prompts, {len(prompts) - len(new)} cached, "
                f"{len(new)} new calls, ~{total:,.0f} tokens (~{prompt_tokens:,.0f} prompt + ~{completion * len(new):,.0f} "
                f"completion; {tok_per_char:.3f} tokens/char, {completion:.0f} completion tokens/call)")
        if MODELS[label][0] == "cerebras":
            gap = rag_sql.MIN_REQUEST_INTERVAL_S["cerebras"]
            line += (f"; ~{len(new) * gap / 60:.0f} min at one call per {gap:g} s, "
                     f"~${total * CEREBRAS_USD_PER_TOKEN:.2f} at Stage 3B's rate")
        print(line)


# --- run -------------------------------------------------------------------------------------------------
def run(label, part, wait):
    _, datasets, schemas, chunks, items = setup(part)
    cache = LLMCache(cache_path(label))
    keys = prompt_keys(items, datasets, schemas, chunks, cache, label)
    status_path = LOG_DIR / f"stage3c_{part}_{label}_status.json"
    tag = f"{label} on {provider_of(label)}, part {part}"
    totals = defaultdict(int)

    def write_status(state):
        done = sum(k in cache.entries for k in keys)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps({"model": label, "provider": provider_of(label), "part": part,
                                           "state": state, "unique_prompts": len(keys), "cached": done,
                                           "new_calls_this_run": totals["calls"], "tokens_this_run": totals["total"],
                                           "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
        return done

    print(f"[{tag}] {len(keys)} unique prompts, {write_status('running')} already cached", flush=True)
    current = None
    for item in items:
        if (item["set"], item["system"]) != current:
            current = (item["set"], item["system"])
            print(f"[{tag}] {current[0]}/{current[1]} ...", flush=True)
        n_cached = len(cache.entries)
        while True:
            try:
                gen = generate(item, datasets, schemas, chunks, cache, label)
                break
            except DailyLimitError as e:
                if not wait:
                    write_status("stopped: daily quota")
                    sys.exit(f"Daily quota for {tag} exhausted; re-run later to resume.\n{e}")
                pause = (e.retry_after_s or 900) + 30
                write_status(f"sleeping until {time.strftime('%H:%M', time.localtime(time.time() + pause))}")
                print(f"  daily quota reached ({totals['calls']} new calls so far); sleeping {pause / 60:.1f} min",
                      flush=True)
                time.sleep(pause)
        if len(cache.entries) == n_cached:
            continue  # cache hit
        u = cache.entries[gen.cache_key].get("usage") or {}
        reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        for k, v in [("calls", 1), ("prompt", u.get("prompt_tokens", 0)), ("cached", cached),
                     ("completion", u.get("completion_tokens", 0)), ("reasoning", reasoning),
                     ("total", u.get("total_tokens", 0))]:
            totals[k] += v
        done = write_status("running")
        print(f"  {time.strftime('%H:%M:%S')} #{totals['calls']} ({done}/{len(keys)}) {item['q']['id']} "
              f"{item['system']}: prompt {u.get('prompt_tokens')} (cached {cached}), completion "
              f"{u.get('completion_tokens')} (reasoning {reasoning}), {cache.entries[gen.cache_key].get('latency_s')}s "
              f"| running total {totals['total']:,} tokens", flush=True)
    write_status("done")
    print(f"[{tag}] done: {totals['calls']} new calls, {totals['total']:,} tokens (prompt {totals['prompt']:,}, "
          f"cached prompt {totals['cached']:,}, completion {totals['completion']:,}, of which reasoning "
          f"{totals['reasoning']:,})", flush=True)


# --- report ----------------------------------------------------------------------------------------------
def score(rs):
    done = [r for r in rs if not r["missing"]]
    n = len(done)
    return {"n": n, "n_missing": len(rs) - n, "n_correct": sum(r["correct"] for r in done),
            "accuracy": sum(r["correct"] for r in done) / n if n else None,
            "n_errors": sum(r["error"] is not None for r in done),
            "n_fallback": sum(r["source"] == "fallback" for r in done)}


def report(part):
    sets, datasets, schemas, chunks, items = setup(part)
    gold = {(s, q["id"]): run_safe_query(q["gold_sql"], datasets[q["dataset"]].conn) for s, qs in sets.items() for q in qs}
    records, usage, keys_used = [], {}, {}
    with Offline():
        for label in MODELS:
            cache = LLMCache(cache_path(label))
            calls = {}
            for item in items:
                q, system = item["q"], item["system"]
                gen = generate(item, datasets, schemas, chunks, cache, label)
                missing = gen.error is not None and OFFLINE in gen.error
                ds = datasets[q["dataset"]]
                df, err = (None, None) if missing else execute(gen.sql, ds)
                g = gold[(item["set"], q["id"])]
                if gen.cache_key in cache.entries:
                    calls[gen.cache_key] = cache.entries[gen.cache_key]
                given = [d["id"] for d in gen.docs]
                req = q.get("required_chunks", [])
                ok = (not missing) and df is not None and results_match(g, df, q["ordered"])
                records.append({
                    "part": part, "set": item["set"], "system": system, "model": label, "provider": provider_of(label),
                    "id": q["id"], "dataset": q["dataset"], "intent": q["intent"], "question": q["question"],
                    "gold_sql": q["gold_sql"], "sql": " ".join(gen.sql.split()), "source": gen.source,
                    "missing": missing, "error": err, "fallback_reason": None if missing else gen.error, "correct": ok,
                    "label_only_miss": (not missing) and df is not None and not ok and label_only_miss(g, df, q["ordered"]),
                    "definitions_given": given, "required_chunks": req,
                    "has_all_required": set(req) <= set(given) if system in ("doc_rag", "doc_rag_gated") else None,
                    "naive_match": (not missing) and df is not None and "naive_sql" in q and not ok
                                   and results_match(run_safe_query(q["naive_sql"], ds.conn), df, q["ordered"]),
                })
            usage[label] = dict(usage_stats(calls.values()), provider=provider_of(label),
                                total_tokens=sum((e.get("usage") or {}).get("total_tokens", 0) for e in calls.values()),
                                cached_prompt_tokens=sum(((e.get("usage") or {}).get("prompt_tokens_details") or {})
                                                         .get("cached_tokens") or 0 for e in calls.values()))
            keys_used[label] = len(calls)

    by = defaultdict(list)
    for r in records:
        by[(r["set"], r["system"], r["model"])].append(r)
    acc = {}
    for (s, system, m), rs in by.items():
        acc.setdefault(s, {}).setdefault(system, {})[m] = {
            "overall": score(rs), "by_intent": {i: score([r for r in rs if r["intent"] == i]) for i in INTENTS}}
    if part == "saas":  # all 60 held-out questions, for the systems run on both sets
        for system in ["zero_shot", "doc_rag", "doc_rag_gated"]:
            for m in MODELS:
                rs = by[("saas_glossary", system, m)] + by[("saas_plain", system, m)]
                acc.setdefault("saas_all", {}).setdefault(system, {})[m] = {
                    "overall": score(rs), "by_intent": {i: score([r for r in rs if r["intent"] == i]) for i in INTENTS}}

    first = next(iter(MODELS))
    gating = {}
    for s in sets:
        for system in ["doc_rag", "doc_rag_gated"]:
            rs = by[(s, system, first)]  # what's retrieved doesn't depend on the model
            if not rs:
                continue
            needs = [r for r in rs if r["required_chunks"]]
            n_given = sum(len(r["definitions_given"]) for r in rs)
            n_needed_given = sum(len(set(r["required_chunks"]) & set(r["definitions_given"])) for r in rs)
            gating.setdefault(s, {})[system] = {
                "n": len(rs), "all_required": (sum(r["has_all_required"] for r in needs) / len(needs)) if needs else None,
                "exact": (sum(set(r["required_chunks"]) == set(r["definitions_given"]) for r in needs) / len(needs))
                         if needs else None,
                "precision": n_needed_given / n_given if n_given else None,
                "avg_definitions": n_given / len(rs),
                "questions_with_unneeded_definitions": sum(bool(set(r["definitions_given"]) - set(r["required_chunks"]))
                                                           for r in rs),
                "questions_missing_a_definition": sum(not set(r["required_chunks"]) <= set(r["definitions_given"])
                                                      for r in needs),
                "misses": {r["id"]: {"required": r["required_chunks"], "given": r["definitions_given"]} for r in rs
                           if set(r["required_chunks"]) != set(r["definitions_given"])}}

    def paired(set_name, sys_a, model_a, sys_b, model_b):
        a = {r["id"]: r for r in by[(set_name, sys_a, model_a)] if not r["missing"]}
        b = {r["id"]: r for r in by[(set_name, sys_b, model_b)] if not r["missing"]}
        ids = a.keys() & b.keys()
        c = lambda x, y: sum(a[i]["correct"] == x and b[i]["correct"] == y for i in ids)
        return {"n": len(ids), "both": c(True, True), "only_a": c(True, False), "only_b": c(False, True),
                "neither": c(False, False), "fixed": sorted(i for i in ids if not a[i]["correct"] and b[i]["correct"]),
                "broken": sorted(i for i in ids if a[i]["correct"] and not b[i]["correct"])}

    small, large = list(MODELS)
    gated_vs_ungated = {s: {m: paired(s, "doc_rag", m, "doc_rag_gated", m) for m in MODELS} for s in sets}
    small_vs_large = {s: {system: paired(s, system, small, system, large) for system in PARTS[part][s]} for s in sets}
    naive = {s: {m: {"wrong": sum(not r["correct"] for r in by[(s, "zero_shot", m)] if not r["missing"]),
                     "naive_match": sum(r["naive_match"] for r in by[(s, "zero_shot", m)])} for m in MODELS}
             for s in sets if any(r["required_chunks"] for r in by[(s, "zero_shot", first)])}
    results = {"part": part, "config": CONFIG, "n_questions": {s: len(qs) for s, qs in sets.items()},
               "accuracy": acc, "gating": gating, "gated_vs_ungated": gated_vs_ungated,
               "small_vs_large": small_vs_large, "zero_shot_naive": naive, "usage": usage,
               "n_missing": sum(r["missing"] for r in records)}
    if MODELS[large][0] == "cerebras":
        results["cerebras_cost_usd_estimate"] = usage[large]["total_tokens"] * CEREBRAS_USD_PER_TOKEN
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"stage3c_{part}_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fails = [r for r in records if not r["correct"] and not r["missing"]]
    pd.DataFrame(fails).drop(columns=["missing", "correct"], errors="ignore").to_csv(
        RESULTS_DIR / f"stage3c_{part}_failures.csv", index=False, encoding="utf-8")
    md = to_markdown(results, list(sets))
    (RESULTS_DIR / f"stage3c_{part}_results.md").write_text(md, encoding="utf-8")
    print(md)


def to_markdown(res, set_names):
    acc, models = res["accuracy"], list(MODELS)
    name = lambda m: f"{m} ({provider_of(m)})"
    title = {"dev": "Stage 3C dev-set results (tuning data: healthcare + retail)",
             "saas": "Stage 3C held-out results (SaaS, frozen settings, run once)"}[res["part"]]
    L = [f"# {title}", ""]
    if res["n_missing"]:
        L += [f"**Incomplete: {res['n_missing']} responses are not cached yet.** Accuracies cover answered "
              "questions only.", ""]
    g = res["config"]["doc_rag_gated"]
    L += [f"Models: {', '.join(name(m) for m in models)}; temperature 0, low reasoning effort. doc_rag: top-"
          f"{res['config']['doc_rag']['k']} chunks, {res['config']['doc_rag']['retriever']} retriever. doc_rag_gated: "
          f"{g['matching']}; retrieval fallback: {g['retrieval_fallback']}.", ""]
    sections = set_names + (["saas_all"] if "saas_all" in acc else [])
    titles = dict(SET_TITLES, saas_all="All 60 SaaS questions (glossary + plain)")
    for s in sections:
        L += [f"## {titles[s]}", "", "| System | " + " | ".join(name(m) for m in models) + " |",
              "|---|" + "---|" * len(models)]
        for system, per_model in acc[s].items():
            cells = []
            for m in models:
                o = per_model[m]["overall"]
                cells.append(f"**{pct(o['accuracy'])}** ({o['n_correct']}/{o['n']})"
                             + (f", {o['n_errors']} SQL errors" if o["n_errors"] else ""))
            L.append(f"| {system} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Accuracy by intent", "", "| Set | System | Model | " + " | ".join(INTENTS) + " |",
          "|---|---|---|" + "---|" * len(INTENTS)]
    for s in sections:
        for system, per_model in acc[s].items():
            for m in models:
                L.append(f"| {s} | {system} | {m} | " + " | ".join(pct(per_model[m]["by_intent"][i]["accuracy"])
                                                                   for i in INTENTS) + " |")
    L += ["", "## Which definitions reached the prompt", "",
          "Model-independent. *All required* and *exact* are over questions that need a definition; *precision* is "
          "the share of sent definitions that the question needed.", "",
          "| Set | System | All required | Exact set | Precision | Definitions per question | Questions given an unneeded definition | Questions missing a needed one |",
          "|---|---|---|---|---|---|---|---|"]
    for s, per in res["gating"].items():
        for system, v in per.items():
            L.append(f"| {s} | {system} | {pct(v['all_required'])} | {pct(v['exact'])} | {pct(v['precision'])} | "
                     f"{v['avg_definitions']:.2f} | {v['questions_with_unneeded_definitions']} | "
                     f"{v['questions_missing_a_definition']} |")
    misses = [(s, qid, v) for s, per in res["gating"].items() for qid, v in per.get("doc_rag_gated", {})
              .get("misses", {}).items()]
    if misses:
        L += ["", "doc_rag_gated questions whose definitions differ from the required set:", ""]
        L += [f"- {s} {qid}: required {', '.join(v['required']) or 'none'}; given {', '.join(v['given']) or 'none'}"
              for s, qid, v in misses]
    L += ["", "## doc_rag (ungated) vs doc_rag_gated, question by question", "",
          "| Set | Model | Both right | Only ungated right | Only gated right | Both wrong | Fixed by gating | Broken by gating |",
          "|---|---|---|---|---|---|---|---|"]
    for s, per in res["gated_vs_ungated"].items():
        for m, p in per.items():
            L.append(f"| {s} | {m} | {p['both']} | {p['only_a']} | {p['only_b']} | {p['neither']} | "
                     f"{', '.join(p['fixed']) or '–'} | {', '.join(p['broken']) or '–'} |")
    small, large = models
    L += ["", f"## Small vs. large ({small} vs. {large}), question by question", "",
          "| Set | System | Both right | Only small | Only large | Both wrong |", "|---|---|---|---|---|---|"]
    for s, per in res["small_vs_large"].items():
        for system, p in per.items():
            L.append(f"| {s} | {system} | {p['both']} | {p['only_a']} | {p['only_b']} | {p['neither']} |")
    if res["zero_shot_naive"]:
        L += ["", "## Zero-shot wrong answers that took the everyday reading (matched `naive_sql`)", "",
              "| Set | Model | Wrong | Matched the naive reading |", "|---|---|---|---|"]
        for s, per in res["zero_shot_naive"].items():
            for m, v in per.items():
                L.append(f"| {s} | {name(m)} | {v['wrong']} | {v['naive_match']} |")
    f = lambda x, d=2: "n/a" if x is None else f"{x:.{d}f}"
    L += ["", "## Calls and tokens behind this part", "",
          "Every cached call this part's prompts use (including ones reused from Stage 3B). Latency is not comparable "
          "across providers.", "",
          "| Model | Provider | Calls | Total tokens | Cached prompt tokens | Client median | Prompt tokens/call | Completion tokens/call |",
          "|---|---|---|---|---|---|---|---|"]
    for m, u in res["usage"].items():
        L.append(f"| {m} | {u['provider']} | {u['n_calls']} | {u['total_tokens']:,} | {u['cached_prompt_tokens']:,} | "
                 f"{f(u['latency_median_s'])}s | {f(u['avg_prompt_tokens'], 0)} | {f(u['avg_completion_tokens'], 0)} |")
    if "cerebras_cost_usd_estimate" in res:
        L += ["", f"Cerebras cost estimate for these {large} calls: **${res['cerebras_cost_usd_estimate']:.3f}** "
              "(at Stage 3B's observed $0.18 per 457K tokens)."]
    L += ["", f"Every wrong answer is in `stage3c_{res['part']}_failures.csv`.", ""]
    return "\n".join(L)


# --- BERT on SaaS ----------------------------------------------------------------------------------------
def bert():
    """The intent classifier on the held-out SaaS questions (a domain it never saw). No LLM calls."""
    saas = load_jsonl(BENCH_DIR / "saas_questions.jsonl")
    dev = load_jsonl(BENCH_DIR / "glossary_questions.jsonl") + load_jsonl(BENCH_DIR / "test_questions.jsonl")
    out = {}
    for name, qs in [("saas", saas), ("dev", dev)]:
        pred = predict_intents(qs)
        rows = [{"id": q["id"], "set": q.get("set", "dev"), "question": q["question"], "intent": q["intent"], "pred": p}
                for q, p in zip(qs, pred)]
        acc = lambda rs: sum(r["intent"] == r["pred"] for r in rs) / len(rs) if rs else None
        out[name] = {"n": len(rows), "accuracy": acc(rows),
                     "by_set": {s: acc([r for r in rows if r["set"] == s]) for s in sorted({r["set"] for r in rows})},
                     "by_intent": {i: acc([r for r in rows if r["intent"] == i]) for i in INTENTS},
                     "confusions": dict(Counter(f"{r['intent']} -> {r['pred']}" for r in rows if r["intent"] != r["pred"])),
                     "errors": [r for r in rows if r["intent"] != r["pred"]]}
    (RESULTS_DIR / "stage3c_bert_saas.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    s, d = out["saas"], out["dev"]
    L = ["# BERT intent accuracy on the held-out SaaS questions", "",
         "The fine-tuned intent classifier was never trained or tuned on SaaS data. Dev = the 60 glossary + 120 original "
         "healthcare/retail questions, for reference.", "",
         "| Questions | n | Accuracy | " + " | ".join(INTENTS) + " |", "|---|---|---|" + "---|" * len(INTENTS),
         f"| SaaS (all) | {s['n']} | **{pct(s['accuracy'])}** | " + " | ".join(pct(s["by_intent"][i]) for i in INTENTS) + " |"]
    L += [f"| SaaS {k} | {sum(q['set'] == k for q in saas)} | {pct(v)} |" + " |" * len(INTENTS)
          for k, v in s["by_set"].items()]
    L += [f"| Dev | {d['n']} | {pct(d['accuracy'])} | " + " | ".join(pct(d["by_intent"][i]) for i in INTENTS) + " |",
          "", "Confusions on SaaS (true -> predicted): " + ", ".join(f"{k} ({v})" for k, v in
                                                                 sorted(s["confusions"].items(), key=lambda x: -x[1])), "",
          "| ID | Question | True | Predicted |", "|---|---|---|---|"]
    L += [f"| {r['id']} | {r['question']} | {r['intent']} | {r['pred']} |" for r in s["errors"]]
    md = "\n".join(L) + "\n"
    (RESULTS_DIR / "stage3c_bert_saas.md").write_text(md, encoding="utf-8")
    print(md)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--part", required=True, choices=list(PARTS))
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=list(MODELS))
    r.add_argument("--part", required=True, choices=list(PARTS))
    r.add_argument("--wait", action="store_true", help="sleep through daily-quota limits instead of stopping")
    p = sub.add_parser("report")
    p.add_argument("--part", required=True, choices=list(PARTS))
    sub.add_parser("bert")
    args = ap.parse_args()
    if args.cmd == "estimate":
        estimate(args.part)
    elif args.cmd == "run":
        run(args.model, args.part, args.wait)
    elif args.cmd == "report":
        report(args.part)
    else:
        bert()


if __name__ == "__main__":
    main()
