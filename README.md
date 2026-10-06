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

This repo is v2 of the project. Its full commit history carries over from the earlier versions:

1. **Original class project:** a healthcare-only NL-to-SQL system:
   [dangkhoa241/LLMs-powered-natural-language-query-system-for-healthcare](https://github.com/dangkhoa241/LLMs-powered-natural-language-query-system-for-healthcare)
2. **v1:** generalized to any CSV, with a Streamlit demo:
   [dangkhoa241/ML-assisted-natural-language-to-SQL-query-system](https://github.com/dangkhoa241/ML-assisted-natural-language-to-SQL-query-system)

v2 adds retrieval-augmented SQL generation and a React front end.

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
  v2 revisits this with retrieval-augmented prompting and a proper benchmark. See
  "Retrieval-Augmented SQL" below.

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
  retail_sales.csv         # Seeded synthetic retail dataset (second SQL-benchmark dataset)
  intent_dataset.csv       # Domain-neutral training data for the intent classifier

intent_model/              # Saved fine-tuned BERT intent classification model (after training)

eval/
  intent_hard_test.csv      # 150 hand-written hard test questions (see "Evaluation" below)
  evaluate_intent.py         # Compares BERT against keyword and TF-IDF baselines
  evaluate_sql.py            # Text-to-SQL benchmark: rule-based vs zero-shot LLM vs RAG
  evaluate_doc_retrieval.py  # Glossary retrieval: dense vs BM25 vs hybrid
  evaluate_stage3.py         # Stage 3: glossary RAG and model-size comparison (20b on Groq, 120b on Cerebras)
  sql_metrics.py             # Execution-accuracy result-set comparison
  sql_benchmark/
    test_questions.jsonl     # 120 benchmark questions with gold SQL (healthcare + retail)
    glossary_questions.jsonl # 60 questions that need a business definition (Stage 3)
    build_glossary_questions.py # Builds and checks the glossary benchmark
    REVIEW.md                # 15 glossary questions written out for human review
    example_bank.jsonl       # 150 question -> SQL retrieval examples on 5 other schemas
    build_example_bank.py    # Builds the bank and executes every example's SQL
    check_benchmark.py       # Gold-query and bank/test leakage checks
    make_retail_dataset.py   # Generates data/retail_sales.csv (seeded)
  results/                    # Generated metrics, tables, failures, cached LLM responses

src/
  app.py                    # Streamlit entry point — thin orchestrator wiring the pieces together
  data_context.py            # CSV loading, type inference, SQLite table setup
  intent.py                   # BERT intent classifier + keyword-based fallback
  sql_builder.py               # Schema-aware, rule-based NL -> SQL generation
  rag_sql.py                   # Retrieval-augmented NL -> SQL (Groq or Cerebras LLM + FAISS retrieval)
  doc_retrieval.py             # Business-glossary retrieval (dense, BM25, hybrid)
  sql_safety.py                # Read-only, single-SELECT, timeout + row-cap guardrails
  visualization.py              # Chart rendering + insights
  model_training.ipynb           # Notebook for training the intent model

tests/                      # pytest: SQL safety checks + result-set comparison
logs/                       # Training logs
requirements.txt           # Runtime dependencies
requirements-train.txt     # Training / evaluation / test dependencies
.env.example               # Template for .env (GROQ_API_KEY; CEREBRAS_API_KEY for the Stage 3 benchmark)
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

## 🔎 Retrieval-Augmented SQL

v1's README says LLM-generated SQL was unreliable for aggregate, compare and trend queries. This stage
tests whether that's still true, and whether retrieval fixes it, by giving an LLM retrieved
question→SQL examples plus the relevant schema. `src/app.py` doesn't use this module yet; app
integration is the next stage.

### Architecture

```text
question ──► embed (all-MiniLM-L6-v2) ──► FAISS top-k ──► k example question→SQL pairs ─┐
dataset  ──► build_schema_context(): columns, types, categorical values, min/max ───────┤
                                                                                       ▼
                              Groq LLM (openai/gpt-oss-120b, temperature 0) ──► SQL text
                                                                                       ▼
            validate_sql(): one SELECT/WITH statement, no write or admin keywords
                     │ rejected, or LLM call failed                    │ ok
                     ▼                                                 ▼
          rule-based sql_builder (v1)      read-only DB copy + 5 s timeout + 1,000-row cap
```

* **`src/rag_sql.py`:** `build_schema_context`, `retrieve_examples`, and `generate_sql(question,
  dataset, mode)` with `mode` set to `llm_zero_shot` (schema only) or `llm_rag` (schema plus k
  examples). The model name, embedding model and default k are config constants at the top of the
  file. The API key comes from `GROQ_API_KEY` in `.env`; copy `.env.example`.
* **`src/sql_safety.py`:** three independent layers.
  1. A static check that allows a single `SELECT`/`WITH` statement and rejects `;`-chained
     statements, `PRAGMA`, `ATTACH`, DDL and DML. Keywords inside string literals, quoted
     identifiers and comments are ignored.
  2. Execution on a private in-memory copy of the database with `PRAGMA query_only` and an
     authorizer that only permits reads.
  3. A wall-clock timeout and a row cap.
* **Example bank (`eval/sql_benchmark/example_bank.jsonl`):** 150 question→SQL pairs on 5 schemas
  that don't appear in the benchmark: HR, school, logistics, hotel and energy. The bank teaches SQL
  patterns but can't leak benchmark answers. Every bank query is executed when the bank is built.

### Design choices

* **FAISS over Chroma.** The bank is a 150-row, version-controlled JSONL file, and FAISS covers what
  it needs:
  * exact search with `IndexFlatIP` over unit vectors, which is cosine similarity;
  * one pip wheel;
  * no server, no SQLite-backed store, no telemetry, and no extra persistence layer to keep in sync.

  The index is a rebuildable artifact, cached in `.rag_cache/` and keyed by the bank's hash.
  Chroma's metadata filtering and incremental upserts would only pay off with a large, changing
  example store.
* **`all-MiniLM-L6-v2` for embeddings.** It's small, fast on CPU and free. Its similarity scores are
  spread out enough for the leakage check (cosine > 0.9) to be meaningful.
* **`openai/gpt-oss-120b` on Groq.** It's the strongest general model on Groq's current list. It
  runs with low reasoning effort to keep latency and token use down.
* **The safety failure mode is fallback.** If the LLM call fails or its SQL is rejected, the app
  still answers, using the v1 rule-based generator.

### Benchmark

* **Data:** 120 questions with gold SQL, 60 on `healthcare_dataset.csv` and 60 on a seeded synthetic
  `retail_sales.csv` (1,000 orders), with 12 per intent per dataset. Every gold query was executed.
  Filters return 3–71 rows, and top-N questions have no ties at the LIMIT boundary.
* **Leakage check:** `eval/sql_benchmark/check_benchmark.py` compares every bank question with every
  test question. It caught one exact duplicate, which was then reworded. After that, the maximum
  embedding cosine is **0.80** ("average billing amount by year" vs "average bill by year") and the
  maximum difflib ratio is **0.82**, both under the 0.9 and 0.85 limits.
* **Metric: execution accuracy.** The generated SQL's result set must equal the gold result set.
  * Rows are compared as a multiset, or in order if the question asks for a ranking.
  * Floats are rounded to 2 decimals.
  * Column names and order are ignored. Extra predicted columns are allowed, but every gold column
    must be matched.
  * The comparison logic is in `eval/sql_metrics.py` and covered by `tests/`.
* **Run it** with `python eval/evaluate_sql.py`. Every LLM response is cached in
  `eval/results/llm_cache.jsonl`, keyed by model and prompt hash, so re-runs make no API calls.
  * A run from an empty cache needs about 480 calls (~400K tokens), which is more than Groq's free
    tier allows in a day (200K tokens). The script stops cleanly at the quota and resumes from the
    cache.
  * The script pins `PYTHONHASHSEED` because of a rule-based bug described below.

### Results

| System | Overall | Healthcare | Retail | SQL errors | Rejected by safety | Fallbacks |
|---|---|---|---|---|---|---|
| rule_based (BERT intent, as in the app today) | **30.8%** (37/120) | 36.7% | 25.0% | 0 | 0 | 0 |
| rule_based_gold_intent (true intent) | **32.5%** (39/120) | 38.3% | 26.7% | 0 | 0 | 0 |
| llm_zero_shot (schema only) | **99.2%** (119/120) | 98.3% | 100.0% | 0 | 0 | 0 |
| llm_rag (schema + 3 examples) | **93.3%** (112/120) | 96.7% | 90.0% | 1 | 1 | 1 |

Accuracy by intent:

| System | filter | count | aggregate | compare | trend |
|---|---|---|---|---|---|
| rule_based | 21% | 38% | 54% | 17% | 25% |
| rule_based_gold_intent | 21% | 38% | 54% | 17% | 33% |
| llm_zero_shot | 100% | 100% | 100% | 96% | 100% |
| llm_rag | 92% | 100% | 88% | 88% | 100% |

Ablation over the number of retrieved examples (k=0 sends exactly the zero-shot prompt):

| k | Overall | filter | count | aggregate | compare | trend | SQL errors |
|---|---|---|---|---|---|---|---|
| 0 | **99.2%** | 100% | 100% | 100% | 96% | 100% | 0 |
| 1 | **80.0%** | 92% | 96% | 58% | 83% | 71% | 2 |
| 3 | **93.3%** | 92% | 100% | 88% | 88% | 100% | 1 |
| 5 | **95.8%** | 96% | 100% | 92% | 96% | 96% | 0 |

Full results: `eval/results/sql_results.md` / `.json`. Every question that any system got wrong is
listed in `eval/results/sql_failures.csv`.

### What this shows

**The hypothesis was wrong for this model and benchmark.** A current LLM given a good schema context
(types, actual categorical values, date format and ranges) doesn't make the errors v1 ran into.
Zero-shot got every `GROUP BY`, aggregate and operator right, with no SQL errors. Retrieval had
nothing left to fix: it never turned a wrong zero-shot answer into a right one, and it broke 7 that
zero-shot got right.

**Why retrieval hurt.** I traced every RAG failure back to its retrieved examples and raw response.
There were two causes:

1. **Retrieval matches on topic, not SQL shape.** "Average order revenue by region" retrieved hotel
   *revenue* examples (similarity 0.41–0.50) instead of an "average X by Y" pattern.
2. **The examples conflict with the target schema.** Every bank example queries a table also named
   `data`, but with different columns. With low reasoning effort the model sometimes resolves the
   conflict badly:
   * it copies an example outright (`SELECT Cancelled, AVG(Total Price) ...`, a hotel query, for a
     retail question);
   * it substitutes an example's filter value (`Category = 'Electronics'` instead of
     `Customer Segment = 'Corporate'`);
   * it gives up (`SELECT 1`, `SELECT * FROM data LIMIT 0`, or `LIMIT 3` with the WHERE clause
     dropped).

   This second cause is partly a prompt-design flaw on my side. Fixing it is the obvious next step
   (see below).

**The ablation fits that explanation.** k=1 is worst (80%) because a single off-topic example
dominates the prompt. More examples dilute any one bad example: k=3 scores 93% and k=5 scores 96%.
None of them beats k=0.

**The rule-based generator is far weaker than its v1 demos suggested,** and the intent model isn't
the bottleneck: BERT predicts 88% of benchmark intents correctly, and giving it the true intent only
adds 2 points. The failures are in SQL construction:

* **Filters are dropped or wrong.**
  * "Over 80" is ignored when the word "age" isn't nearby.
  * "Older than 65" and "between 30 and 40" aren't recognized at all.
  * "Under 2000" becomes a filter for the *year* 2000.
* **Compare questions group by the wrong column.** Without a "by X" phrase, it groups by the
  lowest-cardinality column. "Cigna vs Aetna average billing" is grouped by Gender.
* **The wrong aggregate is chosen.** "Highest bill" doesn't match the column name "Billing Amount",
  so it falls back to `COUNT(*)`.
* **Some questions are out of reach entirely:** top-N, `COUNT(DISTINCT)`, lengths of stay, and
  free-text values such as a customer name.
* **The date column is nondeterministic.** `sql_builder` picks it with `next(iter(set))`, so on the
  healthcare data it's *Date of Admission* in some processes and *Discharge Date* in others. Total
  accuracy moves between 27.5% and 32.5% depending on Python's hash seed. The numbers above use a
  pinned seed. The bug wasn't fixed in this stage; Stage 3 fixed it.

**Remaining LLM misses are mostly judgment calls.** Zero-shot's one miss ("normal vs abnormal test
results") also returned the third group, "Inconclusive". One RAG miss answered "which sells more,
yoga mats or water bottles" with only the winning row. The metric counts both as wrong.

**Caveats:**

* The benchmark is small (n=120; the 95% confidence interval on 99.2% is about 95–100%).
* It's single-table, and I wrote both the questions and the gold SQL.
* The LLM prompt states the same output conventions the gold SQL follows: `SELECT *` for listings,
  one row per compared group, and `strftime` buckets. This is a best case for the LLM systems. The
  rule-based system doesn't share those conventions, though the column-flexible metric reduces the
  difference.
* One RAG output was an empty response, counted under "Rejected by safety"; the fallback answered
  that question correctly.

**What this means for the next stage:**

* Use `llm_zero_shot` as the primary generator, with the rule-based path as the safety fallback.
* Keep RAG off by default until it's fixed. Candidate fixes:
  * give each example its own table name and schema line, so it can't be confused with the user's table;
  * retrieve by SQL pattern rather than topic, for example by masking domain nouns before embedding,
    or by filtering the bank by predicted intent;
  * add a minimum-similarity threshold, below which no examples are sent.
* RAG is most likely to help on harder schemas, such as multi-table joins, unusual column semantics
  or domain-specific definitions. This benchmark doesn't test those.

### Stage 3: business definitions and model size

Stage 2 found that the schema alone was enough context for the original 120 questions. Stage 3 asks
two questions:

1. Does retrieval help when the schema *isn't* enough, because the question uses a company-specific
   business term?
2. Can a smaller model plus retrieval match a larger model?

#### What was added

* **Business glossaries** (`docs/glossary/healthcare.md`, `retail.md`), with 26 definitions each.
  Several definitions are deliberately different from the everyday reading:
  * a *senior patient* is 67 or older, not 65;
  * *government payers* are Medicare **and** UnitedHealthcare;
  * the hospital's fiscal year runs July–June and the retailer's runs February–January;
  * a *returning customer* ordered in both 2023 and 2024, and the term has nothing to do with returns.

  The glossaries also contain near-miss distractors: *senior care tier* (80+) next to *senior
  patient*, *premium room* next to *premium insurer*, and *net sales* next to *net revenue*.
* **A 60-question glossary benchmark** (`eval/sql_benchmark/glossary_questions.jsonl`): 30 questions
  per dataset and 12 per intent. Each question uses at least one term without restating it. Each comes
  with the IDs of the definitions it needs, gold SQL, and a `naive_sql` for the plausible everyday
  reading. Every naive query was checked to give a different answer from the gold. 15 questions are
  written out for human review in `eval/sql_benchmark/REVIEW.md`.
* **Glossary retrieval** (`src/doc_retrieval.py`), using dense embeddings, BM25, or both fused with
  reciprocal rank fusion. The hybrid retriever was best: recall@3 **93.3%**, and 90% of questions had
  every needed definition in the top 3 (`eval/results/doc_retrieval_results.md`).
* **The Stage 2 example-retrieval fixes.** Each example now shows its own table name and schema, the
  bank is filtered by predicted intent, and examples below cosine 0.35 aren't sent. This system is
  called `example_rag_fixed`.
* **The `sql_builder` hash-seed bug** described above is fixed. The date column is now chosen
  deterministically.

#### Setup

* **Systems.** On the glossary set:
  * `zero_shot`: schema only;
  * `doc_rag`: schema plus the top-3 retrieved definitions;
  * `oracle_doc`: schema plus exactly the needed definitions, which is an upper bound for `doc_rag`.

  On the original 120 questions: `zero_shot`, `example_rag_fixed`, and `doc_rag`. Running `doc_rag`
  there checks whether irrelevant definitions do harm.
* **Models.** `gpt-oss-20b` runs on **Groq**, and `gpt-oss-120b` runs on **Cerebras**. Both use the
  same prompts, temperature 0, low reasoning effort and a 1,024-token completion cap.
  * The 120b moved to Cerebras because Groq's free-tier daily token cap would have spread its 516
    calls over several days. Cerebras finished them in one afternoon.
  * Every 120b number in this section comes from Cerebras, and every 20b number comes from Groq. The
    Stage 2 tables above are Groq results.
  * Cerebras reproduces Stage 2's 120b zero-shot result exactly: 99.2%, with the same single miss.
* **Running it.** `python eval/evaluate_stage3.py run --model gpt-oss-20b|gpt-oss-120b --wait` makes
  the calls, and `python eval/evaluate_stage3.py report` scores everything from the caches.
  * The provider is part of the cache key, and each model has its own cache file in `eval/results/`.
  * Cerebras calls are throttled to 5 per minute. The run also hit a ~300 requests/hour limit once
    and slept through it.

#### Results

Glossary set (60 questions that need a business definition):

| System | gpt-oss-20b (Groq) | gpt-oss-120b (Cerebras) |
|---|---|---|
| zero_shot (schema only) | **10.0%** (6/60) | **10.0%** (6/60) |
| doc_rag (top-3 retrieved definitions) | **86.7%** (52/60) | **88.3%** (53/60) |
| oracle_doc (exactly the needed definitions) | **95.0%** (57/60) | **95.0%** (57/60) |

Original set (the 120 Stage 2 questions):

| System | gpt-oss-20b (Groq) | gpt-oss-120b (Cerebras) |
|---|---|---|
| zero_shot | **98.3%** (118/120) | **99.2%** (119/120) |
| example_rag_fixed (Stage 2's llm_rag: 93.3%) | **97.5%** (117/120) | **100.0%** (120/120) |
| doc_rag (glossary definitions, none needed) | **94.2%** (113/120) | **95.0%** (114/120) |

Small vs. large, question by question. "Only 20b" counts questions the 20b got right and the 120b
got wrong.

| Set | System | Both right | Only 20b | Only 120b | Both wrong |
|---|---|---|---|---|---|
| glossary | zero_shot | 4 | 2 | 2 | 52 |
| glossary | doc_rag | 49 | 3 | 4 | 4 |
| glossary | oracle_doc | 55 | 2 | 2 | 1 |
| original | zero_shot | 118 | 0 | 1 | 1 |
| original | example_rag_fixed | 117 | 0 | 3 | 0 |
| original | doc_rag | 111 | 2 | 3 | 4 |

Full tables (by dataset and intent, failure breakdowns, latency and tokens) are in
`eval/results/stage3_results.md` / `.json`. Every wrong answer is in `eval/results/stage3_failures.csv`,
with the retrieved chunk or example IDs.

#### What this shows

**Business definitions are knowledge the model doesn't have, and model size doesn't supply it.**
Both models score 10% on the glossary set without definitions. The model six times larger is no
better: each model got 2 questions right that the other missed. Many wrong answers are exactly the everyday reading the benchmark was built to catch (14 of the 20b's
and 19 of the 120b's answers match `naive_sql`):

* "senior patients" became `Age >= 65`;
* "government payers" became Medicare only;
* "fiscal year" became the calendar year;
* "returning customers" became `Returned = 'Yes'`.

**Retrieved definitions close almost all of that gap.** Accuracy goes from 10% to 87–88% with
retrieved definitions and to 95% with the exact ones. The small model plus retrieval (86.7%) beats
the large model without it (10.0%) by 77 points.

**On this benchmark, context matters far more than model size.** Given the same context, the two
models are never more than 2.5 points apart (3 of 120 questions), and they tie on `oracle_doc`. At low reasoning effort, the 20b is a reasonable choice whenever
the right context is in the prompt.

**The remaining `doc_rag` gap is mostly retrieval.**

* Both models miss the same 4 questions, and in each of them a needed definition wasn't retrieved.
* A retrieved distractor can make things worse than retrieving nothing. For "high-cost admissions of
  senior patients", the retriever returned *senior care tier* (80+) instead of *senior patient*
  (67+), and both models wrote `Age >= 80`.
* Where the definitions were all present, the remaining misses come from SQL details, especially
  fiscal-year arithmetic (see below).

**The Stage 2 example-retrieval fixes worked.** Example RAG went from 93.3% (Stage 2) to 97.5–100%.
It no longer hurts the 120b, and its 3 extra correct answers are within noise. It still doesn't
reliably beat zero-shot: the 20b drops 0.8 points.

**Irrelevant definitions cost a few points.** `doc_rag` on the original set (no definitions needed)
scores 95.0% for the 120b (99.2% zero-shot) and 94.2% for the 20b (98.3% zero-shot). Both models
applied definitions the question didn't use, despite the prompt telling them not to:

* *billable days* (+1 day) was applied to plain "length of stay" questions;
* the *AOV* rule (non-returned orders only) was applied to "average order revenue".

The AOV case is arguably a conflict between the glossary and the Stage 2 gold SQL rather than a
model error. Either way, the app should only add definitions whose term actually appears in the
question, for example with a retrieval-score or term-match gate, rather than always sending the
top 3.

**Notable failures:**

| Question | System | What went wrong |
|---|---|---|
| number of admissions per fiscal year | 20b oracle_doc | The `CASE` returns text in one branch and an integer in the other, so SQLite puts `'2021'` and `2021` in separate groups |
| net revenue per fiscal year | 120b oracle_doc | The same text/integer mix in its fiscal-year label, although the definition was right |
| how many returning customers are there? | 120b doc_rag + oracle_doc | `strftime('%Y','`Order Date`')`: quoting the column name turns it into a string literal, so every year is NULL |
| patients on Lipitor with abnormal test results admitted in 2023 | 120b doc_rag | It wrote `'Lipicon'`, a value that appears nowhere in the data. Zero-shot spelled it correctly |
| hard goods vs soft goods: average revenue per order | 120b zero_shot | It guessed the hidden definition exactly, then selected `Revenue` from a subquery that didn't include it |
| how many bulk buyers are there? | 20b doc_rag | `COUNT(DISTINCT ...)` with `GROUP BY` returns a 1 for each buyer instead of one count |
| compare CSAT between B2B and consumer orders | 120b doc_rag + oracle_doc | It returned the three segments separately instead of merging Corporate and Small Business into B2B |
| show patients with premium insurers and flagged results... | both, doc_rag | *Premium insurer* wasn't retrieved, so both models listed all five insurers (*premium room* was retrieved instead) |

**Latency and cost.**

* The models ran on different providers, so their latencies can't be compared: the difference is as
  much about hardware as model size. Median client latency was 0.32 s for the 120b on Cerebras and
  0.40 s for the 20b on Groq.
* The 120b's 516 calls used 457K tokens. Cerebras served 324K of them from its prompt cache, so only
  133K uncached tokens counted against the daily limit. The full run cost $0.18 in Cerebras
  credits, about $0.00035 per query.
* The 20b's 516 calls used 452K tokens on Groq. Once the free tier's daily token cap was reached,
  the run slowed to about one call every 5–7 minutes.

**Caveats:**

* The glossary set is small (n=60), and the 95% confidence interval on 87% is roughly ±9 points.
  The model-size differences are well inside that.
* I wrote the glossaries, the questions and the gold SQL. The definitions are deliberately
  counterintuitive so that the model can't already know them. Real company glossaries are often
  closer to everyday usage, so the zero-shot score here is a worst case.
* Each system was run once, at temperature 0. The two models are served by different providers,
  so any serving differences, such as numerics or kernels, are confounded with model size.

**Next stage:** integrate into the app with `zero_shot` as the default generator and glossary
retrieval behind a relevance gate, using the 20b if latency or cost matters.

### Held-out domain (SaaS)

Stage 3B left two problems:

1. Definitions a question doesn't need still cost about 4 points.
2. Retrieval sometimes returned a near-miss distractor instead of the needed definition.

Stage 3C fixes both with **term gating**: a definition is sent only if its term, or one of its listed
aliases, appears in the question. The fix was tuned on healthcare and retail only, then tested once on
a third domain that played no part in any decision.

#### Method

* **Held-out data.** A seeded SaaS subscriptions table (`data/saas_subscriptions.csv`: 1,200
  accounts, 487 of them cancelled) and a 29-term glossary (`docs/glossary/saas.md`).
  * The glossary fixes a reporting as-of date of 2025-06-30, so terms like *active account* and
    *recently churned* give a deterministic answer.
  * Its rules are non-obvious. For example, *churned* means cancelled more than 14 days after
    signup; earlier cancellations are *trial drop-offs*. *ARR* counts only annual-billing accounts.
  * 5 terms are near-miss distractors: ACV vs. ARR, dormant vs. active, recent signup vs. new logo,
    enterprise-scale vs. strategic, and passive vs. detractor.
  * The glossary and its aliases were written before any question.
* **60 held-out questions** (`eval/sql_benchmark/saas_questions.jsonl`):
  * 40 need a definition (8 per intent). Each has gold SQL and a naive everyday reading that gives a
    different answer.
  * 20 plain questions need none. Several sit next to a glossary term ("average MRR by plan",
    "number of cancellations per year") to test whether unneeded definitions get applied.
  * 15 are written out for review in `REVIEW_SAAS.md`.
* **Tuning on the dev set only.**
  * Aliases were added to the healthcare and retail glossaries without changing any definition
    text, so every Stage 3B prompt stayed byte-identical and cached.
  * On dev, gating sent exactly the required definitions for all 60 glossary questions and none for
    the 120 original questions. That is optimistic by construction, because I wrote those aliases
    while looking at the dev questions.
  * Because gating missed nothing on dev, no retrieval fallback was added.
  * The settings were frozen in `eval/stage3c_config.json` before any SaaS call
    (`eval/evaluate_stage3c.py`).
* **Same setup as Stage 3B:** gpt-oss-20b on Groq, gpt-oss-120b on Cerebras, temperature 0, low
  reasoning effort.

#### Results

Dev set (tuning data):

| System | Glossary 20b | Glossary 120b | Original 20b | Original 120b |
|---|---|---|---|---|
| doc_rag (ungated top-3) | 86.7% | 88.3% | 94.2% | 95.0% |
| **doc_rag_gated** | **95.0%** | **91.7%** | **98.3%** | **99.2%** |
| oracle_doc / zero_shot | 95.0% (oracle) | 95.0% (oracle) | 98.3% (zero-shot) | 99.2% (zero-shot) |

**Held-out SaaS set** (frozen settings, run once):

| System | Glossary (40): 20b | Glossary (40): 120b | Plain (20): 20b | Plain (20): 120b | All 60: 20b | All 60: 120b |
|---|---|---|---|---|---|---|
| zero_shot | 0.0% | 0.0% | 95.0% | 100.0% | 31.7% | 33.3% |
| doc_rag (ungated top-3) | 67.5% | 77.5% | 100.0% | 100.0% | 78.3% | 85.0% |
| **doc_rag_gated** | **80.0%** | **95.0%** | **95.0%** | **100.0%** | **85.0%** | **96.7%** |
| oracle_doc | 85.0% | 97.5% | n/a | n/a | n/a | n/a |

Which definitions reached the prompt (this doesn't depend on the model):

| SaaS set | System | Has every needed definition | Exact set | Precision | Questions given an unneeded definition |
|---|---|---|---|---|---|
| glossary (40) | doc_rag | 75.0% | 0% | 31.7% | 40 |
| glossary (40) | doc_rag_gated | **97.5%** | **95.0%** | **98.0%** | 1 |
| plain (20) | doc_rag | n/a | n/a | 0% | 20 |
| plain (20) | doc_rag_gated | n/a | n/a | n/a | **0** |

Small vs. large on the SaaS glossary set, question by question:

| System | Both right | Only 20b | Only 120b | Both wrong |
|---|---|---|---|---|
| zero_shot | 0 | 0 | 0 | 40 |
| doc_rag | 25 | 2 | 6 | 7 |
| doc_rag_gated | 31 | 1 | 7 | 1 |
| oracle_doc | 33 | 1 | 6 | 0 |

Full results are in `eval/results/stage3c_saas_results.md` / `.json`, and every wrong answer is in
`stage3c_saas_failures.csv`.

#### What this shows

**Gating held up on the unseen domain.**
* For the 120b, it raised the glossary set from 77.5% (ungated) to 95.0%, within one question of
  the oracle (97.5%). The 20b went from 67.5% to 80.0%.
* On the plain questions, gating sent no definitions at all, so those prompts were exactly the
  zero-shot ones.
* Ungated `doc_rag` gave every question three definitions. On the glossary set, only 31.7% of those
  were ones the question needed, and a quarter of the questions were missing a needed one.
* Overall, gating scored 85.0% (20b) and 96.7% (120b) on all 60 questions, against 78.3% and 85.0%
  ungated. **All of that gain came from the glossary questions.**

**The "unneeded definitions" problem didn't reproduce on SaaS.** On dev, sending definitions to
questions that didn't need them cost about 4 points. On the SaaS plain questions, ungated `doc_rag`
scored 100% for both models, even with three irrelevant definitions in every prompt. The SaaS
definitions mostly name distinct concepts and don't redefine plain words; the dev case was *billable
days* being applied to "length of stay". So gating made no difference on the plain set. For the
20b it was one question worse: sp_p03 is a miss that zero-shot makes too, and the extra context
happened to fix it. The protection gating gave on dev's original set is real, but on this held-out
domain it was never needed.

**Without definitions, both models scored 0 of 40.** That's lower than the dev domains (10%),
because the SaaS rules are less guessable. Typical wrong answers:
* "active accounts" became `Cancel Date IS NULL`, ignoring the 30-day login rule;
* "churned in fiscal year 2024" became calendar 2024 and included trial drop-offs;
* "enterprise-scale" became `Plan = 'Enterprise'`.

**Unlike on dev, the larger model now leads clearly.** Both models got exactly the same definitions,
yet the 120b scored 95.0% gated and 97.5% with the oracle definitions, against 80.0% and 85.0% for
the 20b. Under gating, 7 questions were right only for the 120b and 1 only for the 20b. Most of the
20b's extra misses are SQL mistakes rather than misreadings of a definition: SQLite integer division
and missed filters (see below). The SaaS definitions involve more arithmetic (ratios, percentages,
date differences) than the dev ones.

**The one gating miss was an alias gap, not a matching bug.** In "list **over-provisioned Pro
accounts** in APAC", the plan name sits inside the term "over-provisioned account", and the
single-word alias was spelled "overprovisioned". Neither matched, so both models guessed. The
120b's guess, `Seats Used > Seats`, is the opposite of the definition.

*Post-hoc, not part of the frozen result:* with "over-provisioned" added as an alias, that question
gets exactly the oracle prompt, which both models already answer correctly. Gated accuracy would
then be 82.5% (20b) and 97.5% (120b). The frozen numbers above stay the reported ones.

**Notable failures:**

| Question | System | What went wrong |
|---|---|---|
| seat utilization by region | 20b gated + oracle | `SUM(Seats Used) / SUM(Seats) * 100` in SQLite is integer division, so every region gets 0 |
| net promoter score by industry / by signup year | 20b gated + oracle | The same integer division in `SUM(...) / COUNT(NPS)`. The definition was applied correctly |
| do paid-acquisition accounts have a higher logo churn rate? | 20b gated | It filtered to cancelled accounts first, so both groups show a 100% churn rate |
| compare the number of detractors and passives | 120b, all three systems with definitions | `CASE` without an `ELSE` leaves promoters (9–10) as a third, unlabeled `NULL` group |
| compare ARR between EMEA and North America | 20b gated + oracle | ARR was right, but it returned all four regions |
| total MRR of at-risk accounts by plan | 120b doc_rag | *At-risk* wasn't retrieved, so the model used **cancelled** accounts: the opposite of the definition |
| count the detractors in EMEA | 120b doc_rag | Only the *passive* distractor was retrieved, and the model fell back to the textbook detractor rule (≤ 6) |
| trial drop-offs per fiscal year | 20b gated + oracle | The fiscal-year label mixes text and integers again (as in Stage 3B), splitting groups |

**BERT intent accuracy on SaaS** (a domain the classifier never saw) was **75.0%**, against 86.7%
on dev. Trend questions phrased "X by signup year" are mostly read as aggregate or count (16.7%
correct). This doesn't affect the LLM systems above, which don't use the predicted intent for
glossary retrieval.

**Cost.**
* The 120b's 165 calls on Cerebras used 137K tokens, of which 94K were served from the prompt cache.
  That's about **$0.05** at Stage 3B's observed rate.
* The 20b's 165 calls used 138K tokens on Groq, and the run was again paced by the free tier's daily
  token cap.

**Caveats:**

* The held-out set is small: 40 glossary questions plus 20 plain ones, so one question is 2.5
  points on the glossary set.
* One author wrote the SaaS data, glossary, aliases, questions and gold SQL. The aliases were fixed
  before the questions were written, but my phrasing habits still carry over to both.
* Gating can only send what the alias lists cover. A real glossary would need curated aliases, or a
  retrieval fallback for questions that match nothing.
* Each system was run once, at temperature 0. The model comparison is confounded with the provider,
  as in Stage 3B.

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

