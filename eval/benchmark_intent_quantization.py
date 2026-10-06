"""Memory and latency of the BERT intent model, loaded the way the app loads it: torch fp32, torch dynamic
int8 (INTENT_RUNTIME=torch, INTENT_QUANTIZE=none|int8), and the int8 ONNX export on onnxruntime
(INTENT_RUNTIME=onnx, what the app deploys). The ONNX variant never imports torch.

Each variant runs in its own subprocess so its memory is measured from a clean start:
  weights_mb   - size of the serialized state_dict (int8 Linear weights are packed, so this is the real footprint);
                 for ONNX, the size of model_int8.onnx
  rss_mb       - resident memory after loading (incl. torch + transformers), split into anonymous memory (the heap,
                 which a container limit cannot reclaim) and file-backed pages (e.g. memory-mapped weights)
  peak_rss_mb  - the process's peak resident memory, which includes the moment fp32 and int8 copies coexist
  latency      - one question at a time through the transformers pipeline, as the API calls it; median and p95
                 over the 150 hard-set questions, after 5 warm-up calls, with torch limited to one thread

Accuracy is measured separately by eval/evaluate_intent.py (rows bert and bert_int8). Run both on Linux (the
Docker image) so the quantized kernels match production. The container's CPU limit (docker run --cpus) scales
the latency numbers; on a fraction of a core they grow roughly in proportion.

Usage (from the repo root):  python eval/benchmark_intent_quantization.py [--model intent_model]
"""
import argparse
import io
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "eval" / "results"
HARD_PATH = ROOT_DIR / "eval" / "intent_hard_test.csv"


def _status_mb(field: str) -> float:
    """VmRSS / VmHWM from /proc (Linux only)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024
    raise RuntimeError(field)


VARIANTS = ["none", "int8", "onnx"]
NAMES = {"none": "fp32 (torch)", "int8": "int8 (torch, dynamic, Linear layers)", "onnx": "int8 ONNX (onnxruntime)"}


def measure(variant: str, model_path: str) -> dict:
    import pandas as pd

    if variant == "onnx":
        os.environ["INTENT_RUNTIME"] = "onnx"   # OnnxIntentClassifier already uses one thread
    else:
        import torch

        torch.set_num_threads(1)
        os.environ["INTENT_RUNTIME"] = "torch"
        os.environ["INTENT_QUANTIZE"] = variant
    sys.path.insert(0, str(ROOT_DIR / "src"))
    os.environ["INTENT_MODEL_PATH"] = model_path
    from intent import load_intent_classifier

    base = _status_mb("VmRSS")
    t = time.perf_counter()
    clf = load_intent_classifier()
    load_s = time.perf_counter() - t
    if clf is None:
        raise RuntimeError(f"could not load {model_path}")
    import gc
    gc.collect()
    after = {k: round(_status_mb(k), 1) for k in ("VmRSS", "RssAnon", "RssFile", "VmHWM")}

    questions = pd.read_csv(HARD_PATH)["text"].tolist()
    for q in questions[:5]:
        clf(q)
    times = []
    for q in questions:
        t = time.perf_counter()
        clf(q)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    # After inference every weight page has been touched, so this is the steady state.
    served = {k: round(_status_mb(k), 1) for k in ("VmRSS", "RssAnon", "RssFile", "VmHWM")}
    if variant == "onnx":
        weights_mb = round((Path(model_path) / "onnx" / "model_int8.onnx").stat().st_size / 2**20, 1)
    else:
        buf = io.BytesIO()   # serialized last, so this copy of the weights never shows up in the memory readings
        torch.save(clf.model.state_dict(), buf)
        weights_mb = round(len(buf.getvalue()) / 2**20, 1)
    return {"variant": variant, "weights_mb": weights_mb, "torch_imported": "torch" in sys.modules,
            "rss_before_load_mb": round(base, 1), "rss_mb": after["VmRSS"], "anon_mb": after["RssAnon"],
            "file_mb": after["RssFile"], "peak_rss_mb": served["VmHWM"], "load_s": round(load_s, 2),
            "served_rss_mb": served["VmRSS"], "served_anon_mb": served["RssAnon"], "served_file_mb": served["RssFile"],
            "latency_ms_median": round(statistics.median(times), 1),
            "latency_ms_p95": round(times[int(0.95 * (len(times) - 1))], 1), "n_queries": len(times)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT_DIR / "intent_model"))
    ap.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)   # internal: one subprocess
    ap.add_argument("--variants", default=",".join(VARIANTS), help="which to run, e.g. onnx (without torch installed)")
    args = ap.parse_args()
    if args.variant:
        print(json.dumps(measure(args.variant, args.model)))
        return

    rows = []
    for variant in args.variants.split(","):
        out = subprocess.run([sys.executable, __file__, "--model", args.model, "--variant", variant],
                             check=True, capture_output=True, text=True).stdout
        rows.append(json.loads(out.strip().splitlines()[-1]))
    cpus = os.cpu_count()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "intent_quantization.json"
    if out_json.exists() and args.variants != ",".join(VARIANTS):   # partial run: keep the other variants' rows
        old = {r["variant"]: r for r in json.loads(out_json.read_text(encoding="utf-8"))["rows"]}
        old.update({r["variant"]: r for r in rows})
        rows = [old[v] for v in VARIANTS if v in old]
    out_json.write_text(json.dumps({"rows": rows, "cpu_count": cpus}, indent=2), encoding="utf-8")
    L = ["# Intent model: fp32 vs dynamic int8", "",
         f"Measured in Docker (Linux), one inference thread, {cpus} visible CPUs. Accuracy is in "
         "`intent_results.md` (rows `bert`, `bert_int8`, `bert_onnx_int8`).", "",
         "| Variant | Weights | RSS after load (anon + file) | RSS after 155 queries (anon + file) | Peak RSS | Load time | Latency median | Latency p95 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        name = NAMES[r["variant"]]
        L.append(f"| {name} | {r['weights_mb']} MB | {r['rss_mb']} MB ({r['anon_mb']} + {r['file_mb']}) | "
                 f"{r['served_rss_mb']} MB ({r['served_anon_mb']} + {r['served_file_mb']}) | "
                 f"{r['peak_rss_mb']} MB | {r['load_s']} s | "
                 f"{r['latency_ms_median']} ms | {r['latency_ms_p95']} ms |")
    L += ["", "RSS before loading the model (Python and pandas, plus torch for the torch variants): "
          + ", ".join(f"{NAMES[r['variant']]} {r['rss_before_load_mb']} MB" for r in rows) + ".", ""]
    md = "\n".join(L)
    (RESULTS_DIR / "intent_quantization.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
