"""The intent classifier's runtimes: onnxruntime (production) and the choice between it and torch."""
from pathlib import Path

import pytest

import intent
from intent_onnx import OnnxIntentClassifier

ONNX_DIR = Path(__file__).resolve().parent.parent / "intent_model" / "onnx"


def test_runtime_choice_is_validated(monkeypatch):
    monkeypatch.delenv("INTENT_RUNTIME", raising=False)
    assert intent.intent_runtime() == "onnx"
    monkeypatch.setenv("INTENT_RUNTIME", "torch")
    assert intent.intent_runtime() == "torch"
    monkeypatch.setenv("INTENT_RUNTIME", "tensorflow")
    with pytest.raises(ValueError):
        intent.intent_runtime()


def test_missing_onnx_files_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_intent_onnx"):
        OnnxIntentClassifier(str(tmp_path))


@pytest.mark.skipif(not ONNX_DIR.is_dir(), reason="needs intent_model/onnx (scripts/export_intent_onnx.py)")
def test_onnx_classifier_matches_the_pipeline_interface():
    clf = OnnxIntentClassifier(str(ONNX_DIR.parent))
    out = clf("how many patients are older than 80")
    assert len(out) == 1 and out[0]["label"] == "count" and 0.5 < out[0]["score"] <= 1.0
    assert clf.predict(["average billing amount by insurer", "how has monthly revenue changed over time", "show all orders"]) \
        == ["aggregate", "trend", "filter"]
