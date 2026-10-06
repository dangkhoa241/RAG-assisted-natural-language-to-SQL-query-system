"""BERT intent classification with onnxruntime + tokenizers only (no torch, no transformers).

Loads the int8 ONNX export made by scripts/export_intent_onnx.py from a local folder or a Hugging Face Hub
model repo, looking in its onnx/ subfolder: model_int8.onnx, tokenizer.json, config.json (the labels).
Calling the classifier mirrors the transformers text-classification pipeline the app used before:
classifier("question") -> [{"label": ..., "score": ...}].
"""
import json
from pathlib import Path

import numpy as np

ONNX_SUBFOLDER = "onnx"
FILES = ("model_int8.onnx", "tokenizer.json", "config.json")


def _resolve(path: str) -> Path:
    """The folder holding the ONNX files: <path>/onnx locally, or the Hub repo's onnx/ folder (downloaded once
    to the Hugging Face cache)."""
    local = Path(path)
    if local.is_dir():
        folder = local / ONNX_SUBFOLDER if (local / ONNX_SUBFOLDER).is_dir() else local
        missing = [f for f in FILES if not (folder / f).is_file()]
        if missing:
            raise FileNotFoundError(f"{folder} is missing {', '.join(missing)} (run scripts/export_intent_onnx.py)")
        return folder
    from huggingface_hub import hf_hub_download

    files = [hf_hub_download(path, f"{ONNX_SUBFOLDER}/{f}") for f in FILES]
    return Path(files[0]).parent


class OnnxIntentClassifier:
    def __init__(self, path: str, threads: int = 1):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        folder = _resolve(path)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        self.labels = config["labels"]
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        # The file keeps the training-time "pad everything to 64" setting; single questions need no padding.
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(config.get("max_length", 64))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads   # the free host has a fraction of one CPU
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False      # smaller, steadier memory; inputs here are tiny anyway
        self.session = ort.InferenceSession(str(folder / "model_int8.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def logits(self, texts: list) -> np.ndarray:
        """Batch inference; pads each batch to its longest question."""
        encs = self.tokenizer.encode_batch(list(texts))
        width = max(len(e.ids) for e in encs)
        feed = {}
        for name, attr in (("input_ids", "ids"), ("attention_mask", "attention_mask"), ("token_type_ids", "type_ids")):
            if name in self.input_names:
                feed[name] = np.array([getattr(e, attr) + [0] * (width - len(e.ids)) for e in encs], dtype=np.int64)
        return self.session.run(["logits"], feed)[0]

    def predict(self, texts: list) -> list:
        return [self.labels[i] for i in self.logits(texts).argmax(axis=1)]

    def __call__(self, question: str) -> list:
        z = self.logits([question])[0]
        p = np.exp(z - z.max())
        p /= p.sum()
        best = int(p.argmax())
        return [{"label": self.labels[best], "score": float(p[best])}]
