"""Download the intent model's ONNX files (the onnx/ folder of INTENT_MODEL_PATH) into the Hugging Face cache.

Run at build time (render.yaml's buildCommand) so the app starts without downloading the model. Does nothing
if INTENT_MODEL_PATH is unset or a local folder, and never fails the build: the app downloads the model at
startup instead, or uses keyword intents.
"""
import os
import sys
from pathlib import Path


def main():
    repo = os.environ.get("INTENT_MODEL_PATH", "").strip()
    if not repo or Path(repo).exists():
        print("prefetch: INTENT_MODEL_PATH is unset or local; nothing to download")
        return
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo, allow_patterns=["onnx/*"])
        print(f"prefetch: {repo} onnx/ -> {path}")
    except Exception as e:   # the build shouldn't fail because the Hub is briefly unreachable
        print(f"prefetch: skipped ({type(e).__name__}: {e}); the app will download the model at startup",
              file=sys.stderr)


if __name__ == "__main__":
    main()
