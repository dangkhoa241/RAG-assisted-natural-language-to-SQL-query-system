"""Evaluate the intent classifier against simple baselines on the original val split and a harder test set.

Models:
  keyword  - guess_intent_by_keywords from src/intent.py (the app's fallback)
  tfidf_lr - TF-IDF (1-2 grams) + LogisticRegression, trained on the same train split as BERT
  bert     - the fine-tuned model in intent_model/ (CPU, batched inference)

Test sets:
  val            - the notebook's 50% stratified val split (same seed/test_size/stratify)
  hard           - eval/intent_hard_test.csv (all 150 rows)
  in_domain_hard - hard rows from training domains with messy/indirect phrasing
  unseen_domain  - hard rows from domains not present in training

Usage (from the repo root):  python eval/evaluate_intent.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "intent_dataset.csv"
HARD_PATH = ROOT_DIR / "eval" / "intent_hard_test.csv"
MODEL_DIR = ROOT_DIR / "intent_model"
RESULTS_DIR = ROOT_DIR / "eval" / "results"

# Must match src/model_training.ipynb
TEST_SIZE = 0.50
RANDOM_SEED = 42
MAX_LENGTH = 64
BATCH_SIZE = 32

sys.path.insert(0, str(ROOT_DIR / "src"))
from intent import guess_intent_by_keywords  # noqa: E402


def load_split():
    """Rebuild the notebook's train/val split exactly."""
    df = pd.read_csv(DATA_PATH)[["text", "label"]].dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(str)
    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    df["label_id"] = df["label"].map(label2id)
    train_df, val_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["label_id"]
    )
    return train_df, val_df, labels


def predict_bert(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to("cpu").eval()
    id2label = model.config.id2label
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            enc = tokenizer(
                texts[i:i + BATCH_SIZE], truncation=True, padding=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            ids = model(**enc).logits.argmax(dim=-1).tolist()
            preds.extend(id2label[j] for j in ids)
    return preds


def score(y_true, y_pred, labels):
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }


def to_markdown(results, labels, model_names, set_names):
    lines = ["# Intent classifier evaluation", ""]
    lines += ["## Accuracy / macro-F1", ""]
    header = "| Model | " + " | ".join(f"{s} (n={results['sets'][s]['n']})" for s in set_names) + " |"
    lines += [header, "|---|" + "---|" * len(set_names)]
    for m in model_names:
        cells = [
            f"{results['sets'][s]['models'][m]['accuracy']:.3f} / {results['sets'][s]['models'][m]['macro_f1']:.3f}"
            for s in set_names
        ]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "Cells are accuracy / macro-F1.", ""]

    lines += ["## Per-class F1 on the hard set", ""]
    lines += ["| Model | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for m in model_names:
        pc = results["hard_per_class_f1"][m]
        lines.append(f"| {m} | " + " | ".join(f"{pc[l]:.2f}" for l in labels) + " |")
    lines.append("")

    lines += ["## Confusion matrices on the hard set (rows = true, cols = predicted)", ""]
    for m in model_names:
        cm = results["hard_confusion_matrix"][m]
        lines += [f"**{m}**", "", "| true \\ pred | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for l, row in zip(labels, cm):
            lines.append(f"| {l} | " + " | ".join(str(v) for v in row) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    if not MODEL_DIR.exists():
        sys.exit(f"Trained model not found at {MODEL_DIR}. Run src/model_training.ipynb first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, labels = load_split()
    hard_df = pd.read_csv(HARD_PATH)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Hard: {len(hard_df)}")

    tfidf_lr = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
        LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
    )
    tfidf_lr.fit(train_df["text"], train_df["label"])

    models = {
        "keyword": lambda texts: [guess_intent_by_keywords(t) for t in texts],
        "tfidf_lr": lambda texts: list(tfidf_lr.predict(texts)),
        "bert": predict_bert,
    }
    model_names = list(models)

    val_texts = val_df["text"].tolist()
    hard_texts = hard_df["text"].tolist()
    val_preds = {m: f(val_texts) for m, f in models.items()}
    hard_preds = {m: f(hard_texts) for m, f in models.items()}
    for m in model_names:
        hard_df[f"pred_{m}"] = hard_preds[m]

    subsets = {
        "val": (val_df["label"].tolist(), val_preds),
        "hard": (hard_df["label"].tolist(), hard_preds),
    }
    for split in ["in_domain_hard", "unseen_domain"]:
        mask = (hard_df["split_type"] == split).to_numpy()
        subsets[split] = (
            hard_df.loc[mask, "label"].tolist(),
            {m: list(np.array(p)[mask]) for m, p in hard_preds.items()},
        )

    results = {
        "labels": labels,
        "train_size": len(train_df),
        "sets": {},
        "hard_per_class_f1": {},
        "hard_confusion_matrix": {},
    }
    for name, (y_true, preds) in subsets.items():
        results["sets"][name] = {
            "n": len(y_true),
            "models": {m: score(y_true, preds[m], labels) for m in model_names},
        }

    y_hard = hard_df["label"].tolist()
    for m in model_names:
        f1s = f1_score(y_hard, hard_preds[m], labels=labels, average=None, zero_division=0)
        results["hard_per_class_f1"][m] = dict(zip(labels, map(float, f1s)))
        results["hard_confusion_matrix"][m] = confusion_matrix(y_hard, hard_preds[m], labels=labels).tolist()

    with open(RESULTS_DIR / "intent_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md = to_markdown(results, labels, model_names, list(subsets))
    (RESULTS_DIR / "intent_results.md").write_text(md + "\n", encoding="utf-8")

    pred_cols = [f"pred_{m}" for m in model_names]
    wrong = hard_df[(hard_df[pred_cols].ne(hard_df["label"], axis=0)).any(axis=1)]
    wrong.rename(columns={"label": "true_label"}).to_csv(
        RESULTS_DIR / "misclassified.csv", index=False, encoding="utf-8"
    )

    print(md)
    print(f"\n{len(wrong)} hard-set rows misclassified by at least one model -> {RESULTS_DIR / 'misclassified.csv'}")


if __name__ == "__main__":
    main()
