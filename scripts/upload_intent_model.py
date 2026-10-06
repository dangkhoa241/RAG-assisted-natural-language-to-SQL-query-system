"""Upload the trained BERT intent classifier (intent_model/) to a Hugging Face model repo.

The deployed app then loads it with INTENT_MODEL_PATH=<repo id>. Only the files inference needs are
uploaded (weights, config, tokenizer, eval results); training_args.bin is a pickle and is left out.

Log in first (`hf auth login`, or set HF_TOKEN to a write token), then from the repo root:
  python scripts/upload_intent_model.py --repo-id your-username/nl2sql-intent-model
  python scripts/upload_intent_model.py --repo-id your-username/nl2sql-intent-model --private
A private repo also needs an HF_TOKEN secret (read access) on the Space.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
REQUIRED = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
            "special_tokens_map.json"]
OPTIONAL = ["all_results.json", "eval_results.json", "train_results.json"]


def model_card(repo_id: str, config: dict, results: dict) -> str:
    labels = ", ".join(f"`{v}`" for _, v in sorted(config.get("id2label", {}).items(), key=lambda kv: int(kv[0])))
    acc = results.get("eval_accuracy")
    return f"""---
library_name: transformers
pipeline_tag: text-classification
base_model: bert-base-uncased
language: en
---

# NL→SQL intent classifier

Fine-tuned `bert-base-uncased` that labels a natural-language data question with the kind of SQL it needs:
{labels}. It routes questions in the
[RAG-assisted natural-language-to-SQL system](https://github.com/dangkhoa241/RAG-assisted-natural-language-to-SQL-query-system)
and picks the chart type.

- Training data: 1,000 of the 2,000 domain-neutral questions in the repo's `data/intent_dataset.csv`
  (400 per intent, 14 domains); the other 1,000 are the validation split.
- Validation accuracy: {f"{acc:.1%}" if isinstance(acc, (int, float)) else "see eval_results.json"}. That split is
  templated like the training data, so it says little; on 150 hand-written hard questions the model scores
  84.7%, and 75% on questions from an unseen SaaS domain (see the repo's README).

```python
from transformers import pipeline
clf = pipeline("text-classification", model="{repo_id}")
clf("average billing amount by insurance provider")
```
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="e.g. your-username/nl2sql-intent-model")
    ap.add_argument("--folder", default=str(ROOT_DIR / "intent_model"))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi

    folder = Path(args.folder)
    missing = [f for f in REQUIRED if not (folder / f).is_file()]
    if missing:
        sys.exit(f"{folder} is missing {', '.join(missing)}. Train the model first (src/model_training.ipynb).")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    results = json.loads((folder / "eval_results.json").read_text(encoding="utf-8")) \
        if (folder / "eval_results.json").is_file() else {}
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(folder / f))
           for f in REQUIRED + OPTIONAL if (folder / f).is_file()]
    ops.append(CommitOperationAdd(path_in_repo="README.md",
                                  path_or_fileobj=model_card(args.repo_id, config, results).encode("utf-8")))
    info = api.create_commit(args.repo_id, operations=ops, repo_type="model",
                             commit_message="Upload the NL->SQL intent classifier")
    print(f"Uploaded {len(ops)} files to https://huggingface.co/{args.repo_id} ({info.oid[:7]})")
    print(f"Set INTENT_MODEL_PATH={args.repo_id} on the Space.")


if __name__ == "__main__":
    main()
