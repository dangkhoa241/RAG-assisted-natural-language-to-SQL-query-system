# Stage 3 results: glossary RAG and model size

Models: `openai/gpt-oss-20b` on Groq, `gpt-oss-120b` on Cerebras (reasoning effort low, temperature 0). Every response for a model comes from the one provider listed. doc_rag: top-3 glossary chunks, hybrid retriever. example_rag_fixed: up to 3 intent-filtered examples with cosine >= 0.35.

## Glossary set (60 questions that need business definitions)

| System | gpt-oss-20b (Groq) overall | gpt-oss-20b healthcare | gpt-oss-20b retail | gpt-oss-120b (Cerebras) overall | gpt-oss-120b healthcare | gpt-oss-120b retail |
|---|---|---|---|---|---|---|
| zero_shot | **10.0%** (6/60) | 6.7% | 13.3% | **10.0%** (6/60) | 6.7% | 13.3% |
| doc_rag | **86.7%** (52/60) | 80.0% | 93.3% | **88.3%** (53/60) | 86.7% | 90.0% |
| oracle_doc | **95.0%** (57/60) | 93.3% | 96.7% | **95.0%** (57/60) | 100.0% | 90.0% |

## Original set (120 Stage 2 questions)

| System | gpt-oss-20b (Groq) overall | gpt-oss-20b healthcare | gpt-oss-20b retail | gpt-oss-120b (Cerebras) overall | gpt-oss-120b healthcare | gpt-oss-120b retail |
|---|---|---|---|---|---|---|
| zero_shot | **98.3%** (118/120) | 98.3% | 98.3% | **99.2%** (119/120) | 98.3% | 100.0% |
| example_rag_fixed | **97.5%** (117/120) | 100.0% | 95.0% | **100.0%** (120/120) | 100.0% | 100.0% |
| doc_rag | **94.2%** (113/120) | 93.3% | 95.0% | **95.0%** (114/120) | 95.0% | 95.0% |

## Glossary set by intent

| System | Model | filter | count | aggregate | compare | trend |
|---|---|---|---|---|---|---|
| zero_shot | gpt-oss-20b (Groq) | 8.3% | 8.3% | 8.3% | 8.3% | 16.7% |
| zero_shot | gpt-oss-120b (Cerebras) | 8.3% | 16.7% | 0.0% | 8.3% | 16.7% |
| doc_rag | gpt-oss-20b (Groq) | 83.3% | 83.3% | 91.7% | 91.7% | 83.3% |
| doc_rag | gpt-oss-120b (Cerebras) | 83.3% | 83.3% | 91.7% | 83.3% | 100.0% |
| oracle_doc | gpt-oss-20b (Groq) | 100.0% | 100.0% | 100.0% | 100.0% | 75.0% |
| oracle_doc | gpt-oss-120b (Cerebras) | 100.0% | 91.7% | 100.0% | 91.7% | 91.7% |

## Headline: small model + RAG vs. large model zero-shot

| Set | gpt-oss-20b (Groq) + doc_rag | gpt-oss-20b + oracle_doc | gpt-oss-120b (Cerebras) zero_shot | gpt-oss-120b + doc_rag |
|---|---|---|---|---|
| glossary | 86.7% | 95.0% | 10.0% | 88.3% |
| original | 94.2% | n/a | 99.2% | 95.0% |

## Small vs. large, question by question (gpt-oss-20b vs. gpt-oss-120b)

Only questions both models answered are counted. "Only small" means the small model was right and the large model wrong.

| Set | System | n | gpt-oss-20b | gpt-oss-120b | Both right | Only small | Only large | Both wrong |
|---|---|---|---|---|---|---|---|---|
| glossary | zero_shot | 60 | 10.0% | 10.0% | 4 | 2 | 2 | 52 |
| glossary | doc_rag | 60 | 86.7% | 88.3% | 49 | 3 | 4 | 4 |
| glossary | oracle_doc | 60 | 95.0% | 95.0% | 55 | 2 | 2 | 1 |
| original | zero_shot | 120 | 98.3% | 99.2% | 118 | 0 | 1 | 1 |
| original | example_rag_fixed | 120 | 97.5% | 100.0% | 117 | 0 | 3 | 0 |
| original | doc_rag | 120 | 94.2% | 95.0% | 111 | 2 | 3 | 4 |

## Zero-shot on the glossary set: wrong answers that took the literal reading

A naive match means the wrong answer returned exactly what the hand-written literal reading of the term returns (`naive_sql`), e.g. treating "senior patient" as age >= 65 instead of the glossary's cut-off.

| Model | Wrong answers | Matched the naive reading |
|---|---|---|
| gpt-oss-20b (Groq) | 54 | 14 |
| gpt-oss-120b (Cerebras) | 54 | 19 |

## doc_rag failures on the glossary set

A retrieval miss means at least one required chunk was not in the top 3; a generation miss means all required chunks were in the prompt and the SQL was still wrong.

| Model | Retrieval misses | Generation misses |
|---|---|---|
| gpt-oss-20b (Groq) | 4 | 4 |
| gpt-oss-120b (Cerebras) | 4 | 3 |

## Latency and tokens per call

Client latency is the wall time of each successful request (rate-limit waits and throttling excluded); server time is the provider's reported `total_time`. **The two models ran on different providers, so latency is not comparable between them**: it measures Groq vs. Cerebras hardware as much as model size.

| Model | Provider | Calls | Client median | Client p95 | Server median | Server p95 | Prompt tokens | Completion tokens (reasoning) |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | Groq | 516 | 0.40s | 0.91s | 0.15s | 0.54s | 793 | 83 (38) |
| gpt-oss-120b | Cerebras | 516 | 0.32s | 1.06s | 0.07s | 0.53s | 793 | 93 (48) |

## Notes

- Errors, safety rejections and fallbacks per system are in `stage3_results.json`; every wrong answer is in `stage3_failures.csv` (with retrieved chunk IDs and whether it matched the naive-guess SQL).
- Answers that were right except for leaving out a gold label column count as wrong (see tests/test_result_compare.py); there were 1 such answers.
