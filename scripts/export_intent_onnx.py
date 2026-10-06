"""Export the fine-tuned BERT intent model to ONNX with int8 weights, check it against torch, and optionally
upload it to the Hub.

The app's runtime then needs only onnxruntime + tokenizers (src/intent_onnx.py), not torch or transformers.
torch/transformers/onnx are used here at export time only (requirements-eval.txt).

Steps:
  1. export intent_model/ to ONNX (fp32), inputs input_ids / attention_mask / token_type_ids, dynamic batch
     and sequence length
  2. dynamic int8 quantization of the weights with onnxruntime.quantization
  3. verify on the intent val split (1,000 questions): the fp32 ONNX logits must match torch (argmax identical,
     max |diff| < 1e-3); the int8 model's predictions are compared with torch's and reported (its accuracy is
     judged on the hard set by eval/evaluate_intent.py)
  4. write <out>/model_int8.onnx, tokenizer.json and config.json (labels); with --upload, put them in the
     model repo's onnx/ folder

Usage (from the repo root):
  python scripts/export_intent_onnx.py                      # writes intent_model/onnx/
  python scripts/export_intent_onnx.py --upload dangkhoa241/nl2sql-intent-model
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "eval"))

OPSET = 17
INPUTS = ["input_ids", "attention_mask", "token_type_ids"]


def export_fp32(model, tokenizer, path: Path):
    import torch

    sample = tokenizer(["how many patients", "average billing by insurer"], padding=True, return_tensors="pt")
    args = tuple(sample[name] for name in INPUTS)
    dynamic = {name: {0: "batch", 1: "sequence"} for name in INPUTS}
    dynamic["logits"] = {0: "batch"}
    with torch.no_grad():
        torch.onnx.export(model, args, str(path), input_names=INPUTS, output_names=["logits"],
                          dynamic_axes=dynamic, opset_version=OPSET, dynamo=False)


def quantize_int8(src: Path, dst: Path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def onnx_logits(path: Path, encodings: list) -> np.ndarray:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return np.concatenate([session.run(["logits"], {k: e[k] for k in INPUTS})[0] for e in encodings])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT_DIR / "intent_model"))
    ap.add_argument("--out", default=None, help="output folder (default <model>/onnx)")
    ap.add_argument("--upload", metavar="REPO_ID", help="upload to this Hub model repo's onnx/ folder")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from evaluate_intent import BATCH_SIZE, MAX_LENGTH, load_split

    model_dir = Path(args.model)
    out = Path(args.out) if args.out else model_dir / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "model_fp32.onnx"
        export_fp32(model, tokenizer, fp32)
        int8 = out / "model_int8.onnx"
        quantize_int8(fp32, int8)

        _, val_df, _ = load_split()
        texts = val_df["text"].tolist()
        encodings = [{k: v.astype(np.int64) for k, v in tokenizer(texts[i:i + BATCH_SIZE], truncation=True,
                                                                     padding=True, max_length=MAX_LENGTH,
                                                                     return_tensors="np").items()}
                     for i in range(0, len(texts), BATCH_SIZE)]
        with torch.no_grad():
            ref = np.concatenate([model(**{k: torch.from_numpy(e[k]) for k in INPUTS}).logits.numpy()
                                  for e in encodings])
        got32 = onnx_logits(fp32, encodings)
        got8 = onnx_logits(int8, encodings)
        fp32_mb = fp32.stat().st_size / 2**20

    diff = float(np.abs(got32 - ref).max())
    same32 = float((got32.argmax(1) == ref.argmax(1)).mean())
    same8 = float((got8.argmax(1) == ref.argmax(1)).mean())
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    acc = lambda logits: float((np.array(labels)[logits.argmax(1)] == val_df["label"].to_numpy()).mean())
    report = {"n_val": len(texts), "fp32_onnx_max_abs_logit_diff": diff, "fp32_onnx_argmax_agreement": same32,
              "int8_onnx_argmax_agreement": same8, "val_accuracy": {"torch_fp32": acc(ref), "onnx_fp32": acc(got32),
                                                                   "onnx_int8": acc(got8)},
              "size_mb": {"onnx_fp32": round(fp32_mb, 1), "onnx_int8": round(int8.stat().st_size / 2**20, 1)}}
    print(json.dumps(report, indent=2))
    if same32 < 1.0 or diff > 1e-3:
        sys.exit("The fp32 ONNX export doesn't match torch; not writing or uploading it.")

    shutil.copy2(model_dir / "tokenizer.json", out / "tokenizer.json")
    (out / "config.json").write_text(json.dumps({
        "labels": labels, "max_length": MAX_LENGTH, "do_lower_case": True, "opset": OPSET,
        "quantization": "onnxruntime dynamic int8 (weights)", "inputs": INPUTS, "export_check": report,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    if args.upload:
        from huggingface_hub import HfApi

        info = HfApi().upload_folder(repo_id=args.upload, repo_type="model", folder_path=str(out),
                                     path_in_repo="onnx", commit_message="Add the ONNX int8 export")
        print(f"Uploaded to https://huggingface.co/{args.upload}/tree/main/onnx ({info})")


if __name__ == "__main__":
    main()
