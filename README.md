# RAG-Assisted Natural Language to SQL Query System

A natural-language data assistant for any tabular CSV dataset. A fine-tuned BERT intent classifier
routes each question. Retrieval-augmented text-to-SQL then writes the query, using a Groq-hosted LLM
grounded in schema and example context retrieved with local embeddings.

> **Status:** 🚧 The RAG text-to-SQL pipeline and the new React UI are **in progress**. The code in this
> repo is still the v1 system: BERT intent classifier, rule-based SQL generator and Streamlit UI,
> documented below.

**🔗 v1 live demo:** [ml-assisted-natural-language-to-sql-query-system.streamlit.app](https://ml-assisted-natural-language-to-sql-query-system.streamlit.app/)

**v1 pipeline:** User question → Intent classification → SQL generation → Query execution → Table → Chart

---

## 📜 Project history

This repo continues the original project, with its full commit history:
[dangkhoa241/LLMs-powered-natural-language-query-system-for-healthcare](https://github.com/dangkhoa241/LLMs-powered-natural-language-query-system-for-healthcare).
v2 adds retrieval-augmented SQL generation and a React front end. Earlier work is preserved in the history.

---

## 🚀 v1: Enhanced Version

The current code is an enhanced version of the original project.

### What changed

* **Generalized beyond healthcare** — the original app only worked against `healthcare_dataset.csv`, with
  column names, category values, and keyword lists (gender, blood type, insurance provider, admission type,
  etc.) hardcoded directly into the SQL-generation logic. The app now inspects whatever CSV is uploaded at
  runtime and derives its column types, categories, and values from the data itself, so the same code works
  on a sales dataset, an HR dataset, a student dataset, and so on, unmodified.
* **Retrained the intent classifier on a domain-neutral dataset** — `intent_dataset.csv` previously contained
  only healthcare-phrased questions. It now contains 2,000 examples (400 per intent) spanning 14 domains
  (retail, education, HR, finance, sports, restaurants, real estate, IoT, library, logistics, social media,
  manufacturing, ecommerce, healthcare) so the BERT classifier generalizes instead of overfitting to one
  domain's vocabulary.
* **Graceful degradation without a trained model** — previously the app would hard-stop if `intent_model/`
  wasn't present. It now falls back to a lightweight keyword-based intent guesser so it's usable immediately,
  while still preferring the trained BERT classifier when available.
* **More robust filter/aggregate/trend logic** — numeric comparisons now correctly handle phrasing like
  "under age 40" (a word between the comparison term and the number), column matching uses exact word-token
  matching instead of substring matching (avoiding false positives like "orders" matching a column named
  `Order ID`), and trend queries now aggregate the metric actually asked about instead of always returning a
  raw row count.
* **Cleaner dependencies** — removed an invalid/non-existent package (`sqlite3-binary`) from
  `requirements.txt`; `sqlite3` is part of the Python standard library.

### What was evaluated and intentionally left out

* An LLM-API-driven SQL generator and two small local open-source text-to-SQL models were both evaluated.
  Both produced unreliable SQL for aggregate/compare/trend queries (missing `GROUP BY`, dropped aggregate
  functions, wrong comparison operators). The final design instead uses fully self-contained, rule-based SQL
  generation driven by the uploaded schema — no external API calls and no model download required to run it.

The system contains two major components:

### 1. **Intent Classification Model**

* A custom fine-tuned intent classifier trained using `intent_dataset.csv`
* Built on top of a pretrained **BERT (`bert-base-uncased`)** Transformer model
* Trained on domain-neutral example questions (retail, education, HR, finance, sports, healthcare,
  and more) so it generalizes to whatever dataset is uploaded, rather than one specific domain
* Produces intent categories: *filter, count, aggregate, compare, trend*
* Stored in the `intent_model/` folder after training
* Used to decide which type of SQL query to generate and which chart to show
* Optional: if `intent_model/` hasn't been trained yet, the app falls back to a lightweight
  keyword-based intent guesser so it still works out of the box

### 2. **Schema-Aware SQL Generator**

* Hand-built, rule-based NL→SQL engine — no external LLM API calls
* Reads the uploaded CSV's actual columns, data types, and values at runtime, then:
  * matches mentioned category values against the real distinct values in each column
  * matches numeric comparisons ("over 90000", "under age 40") to the right numeric column
  * detects date-like columns and grouping columns from the data itself
* Because everything is derived from the uploaded schema instead of hardcoded column names,
  the same logic works on a healthcare dataset, a sales dataset, an HR dataset, etc.

### 3. **Streamlit Web Application (`app.py`)**

* Interactive UI for uploading a CSV and entering queries
* Displays:

  * 🧠 Detected intent + generated SQL
  * 📄 Clean results table
  * 📈 Automatically generated chart based on intent (bar / pie / line)
  * 📝 Basic insights (highest / lowest values)
* Fully end-to-end: from CSV upload + text input → visualization

---

## 📂 What This Project Contains

```text
data/
  healthcare_dataset.csv   # Example dataset (one of many CSVs the app can load)
  intent_dataset.csv       # Domain-neutral training data for the intent classifier

intent_model/              # Saved fine-tuned BERT intent classification model (after training)

eval/
  intent_hard_test.csv      # 150 hand-written hard test questions (see "Evaluation" below)
  evaluate_intent.py         # Compares BERT against keyword and TF-IDF baselines
  results/                    # Generated metrics, comparison table, misclassified examples

src/
  app.py                    # Streamlit entry point — thin orchestrator wiring the pieces together
  data_context.py            # CSV loading, type inference, SQLite table setup
  intent.py                   # BERT intent classifier + keyword-based fallback
  sql_builder.py               # Schema-aware, rule-based NL -> SQL generation
  visualization.py              # Chart rendering + insights
  model_training.ipynb           # Notebook for training the intent model

logs/                       # Training logs
requirements.txt           # Dependencies
```

---

## 🧠 Model Training (Intent Classifier)

The intent classification model is built by fine-tuning a **BERT-based Transformer (`bert-base-uncased`)** on a labeled, domain-neutral intent dataset.

### Base Model

* **Pretrained Model:** `bert-base-uncased`
* **Architecture:** Bidirectional Transformer Encoder
* **Tokenizer:** WordPiece tokenizer (uncased)
* **Framework:** PyTorch + Hugging Face Transformers

### Training Dataset

* Source file: `data/intent_dataset.csv`
* 2,000 examples (400 per intent) spanning 14 domains — retail, education, HR, finance, sports,
  restaurants, real estate, IoT, library, logistics, social media, manufacturing, ecommerce, and
  healthcare — so the classifier isn't tied to healthcare phrasing
* Each sample contains:

  * A natural language question
  * A corresponding intent label (filter, count, aggregate, compare, or trend)

### Training Objective

The model is fine-tuned for a **multi-class text classification task** to learn how to map user questions to the correct analytical intent. The trained intent model directly controls:

* The structure of generated SQL queries
* The selection of visualization types (bar, line, pie, etc.)

### Training Pipeline

* Input: Natural language questions across many domains
* Tokenization: BERT tokenizer
* Model: `AutoModelForSequenceClassification`
* Loss Function: Cross-entropy loss
* Optimizer: AdamW
* Output: Trained intent classification model saved to `intent_model/`

### Purpose in the System

The fine-tuned BERT intent model enables:

* Accurate understanding of user analytical goals
* Reduced ambiguity before SQL generation
* Automatic and correct chart type selection

---

## 📊 Evaluation

### Why a hard test set

The notebook's validation split (50% of `intent_dataset.csv`) comes from the same templated, synthetic
distribution as the training data. On it, BERT scores 100%, but a TF-IDF + logistic regression baseline
also scores 99.9%. That split can't tell the models apart, so it doesn't justify using BERT.

`eval/intent_hard_test.csv` contains 150 hand-written questions (30 per intent):

* **`in_domain_hard` (75):** domains that appear in training, phrased the way people actually type. The
  questions mostly avoid trigger words like "how many", "average", "compare" and "trend", and they include
  typos, lowercase text without punctuation, multi-clause questions and indirect phrasing ("are men or
  women billed more", "which month did we sell the most").
* **`unseen_domain` (75):** domains that don't appear in training: airlines, hotels, agriculture, gaming,
  energy, insurance claims, car rental and telecom.

No hard-set question is an exact or near duplicate of a training question (every difflib ratio is ≤ 0.75).
Labels follow what each intent does in `sql_builder.py`:

| Intent | What the SQL returns |
|---|---|
| filter | matching rows |
| count | one row count |
| aggregate | a numeric metric per group |
| compare | a metric for a few named groups |
| trend | a metric over time buckets |

### Results

Run `python eval/evaluate_intent.py`. You need `requirements-train.txt` installed and a trained
`intent_model/`. The script rebuilds the notebook's exact train/val split and trains the
TF-IDF baseline on the same 1,000 training rows that BERT used. Full output, including confusion
matrices, is in `eval/results/`.

| Model | val (n=1000) | hard (n=150) | in_domain_hard (n=75) | unseen_domain (n=75) |
|---|---|---|---|---|
| Keyword fallback | 0.657 / 0.669 | 0.240 / 0.136 | 0.240 / 0.133 | 0.240 / 0.139 |
| TF-IDF (1-2 gram) + LogReg | 0.999 / 0.999 | 0.693 / 0.689 | 0.667 / 0.662 | 0.720 / 0.718 |
| BERT (fine-tuned) | 1.000 / 1.000 | 0.847 / 0.841 | 0.853 / 0.837 | 0.840 / 0.844 |

*Cells are accuracy / macro-F1.*

Per-class F1 on the hard set:

| Model | aggregate | compare | count | filter | trend |
|---|---|---|---|---|---|
| Keyword fallback | 0.10 | 0.00 | 0.21 | 0.38 | 0.00 |
| TF-IDF + LogReg | 0.69 | 0.77 | 0.55 | 0.67 | 0.77 |
| BERT | 0.90 | 0.95 | 0.64 | 0.80 | 0.92 |

### What this shows

* **BERT is clearly better than the baselines once phrasing stops being templated.** On the hard set it
  beats TF-IDF by about 15 accuracy points (0.85 vs 0.69). The gap is largest on aggregate, compare and
  trend, where BERT recognizes indirect phrasing such as "what do houses go for in each neighborhood",
  "is fedex faster than ups" and "has the price of jet fuel been rising lately". TF-IDF has no n-gram
  match for these.
* **The val split overstates every model.** BERT drops from 100% to about 85%, and TF-IDF drops from 99.9%
  to about 69%. Treat the val score as a sanity check, not an accuracy estimate.
* **Count is BERT's weak spot (F1 0.64, recall 15/30).** If a question doesn't contain "how many" or
  "count", BERT usually predicts filter ("vacant rooms tonight, just the number", "employees on parental
  leave right now - just a number please"). Typos in the trigger phrase ("how manny", "hw mny") break it
  too. This is the most useful thing to fix in the training data.
* **New domains don't hurt BERT much here** (0.84 unseen vs 0.85 in-domain). With 75 rows per subset,
  though, the 95% confidence interval is about ±8 points, so this difference isn't meaningful. The
  `unseen_domain` subset tests vocabulary shift; its sentence structure is about as hard as the in-domain
  subset's.
* **The keyword fallback's 24% is partly by design.** Most hard-set questions deliberately avoid its
  trigger words, so most of them fall through to "filter". The result does show how brittle the fallback
  is on real phrasing, but it's a lower bound and shouldn't be read as a typical accuracy.
* **Caveats:** the hard set is small (150 rows), one person wrote and labeled it, and that person knew the
  keyword list when writing it. A few labels are judgment calls. For example, "which month did we sell the
  most" is labeled trend because it groups by time.

---

## 🔍 What Makes This Project Unique

* Combines **ML-based intent classification + a self-built, schema-aware SQL generator**
* Works on **any uploaded CSV**, not a single fixed dataset — columns, types, and values are
  discovered at runtime
* Full *intent-aware* NL→SQL system
* Automatic **chart selection** based on predicted intent
* No external LLM API calls — everything runs locally
* End-to-end **Streamlit application** included
* Reproducible model training notebook

---

## 🧠 Example Workflow

**User uploads `healthcare_dataset.csv` and asks:**

> “Show emergency cases under age 40 with billing less than 12,000”

**System:**

1. Intent model → **filter**
2. SQL generator → matches "emergency" to the `Admission Type` column's real values, "under age 40"
   to the `Age` column, "billing less than 12,000" to the `Billing Amount` column
3. Executor → runs the generated SQL against the uploaded data
4. Result → filtered table (charts are generated for aggregate/compare/trend queries)

The same pipeline works unmodified if the user instead uploads a sales, HR, or student dataset —
only the column names and values found in the CSV change, not the code.

---

## 🖥️ Running the Application

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Training the intent classifier (optional — the app falls back to keyword-based intent detection
without it) needs a few extra dependencies, kept in a separate file so the deployed app doesn't
have to install them:

```bash
pip install -r requirements-train.txt
jupyter notebook src/model_training.ipynb
```

---

## ☁️ Deployment

The app is deployable as-is on [Streamlit Community Cloud](https://streamlit.io/cloud) (free,
connects directly to a GitHub repo):

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo,
   branch `main`, and main file path `src/app.py`.
3. Deploy. `intent_model/` is gitignored (it's a ~400MB trained model, not meant for git), so the
   deployed app runs on the keyword-based intent fallback out of the box — no extra setup required.

To have the deployed app use the actual trained BERT classifier instead of the fallback:

1. Push the contents of `intent_model/` to a model repo on the
   [Hugging Face Hub](https://huggingface.co/new) (e.g. `your-username/intent-model`).
2. In the Streamlit Cloud app's settings, add a secret/environment variable
   `INTENT_MODEL_PATH=your-username/intent-model`.
3. Redeploy — `src/intent.py` reads that variable and loads the model from the Hub instead of the
   local `intent_model/` folder.

