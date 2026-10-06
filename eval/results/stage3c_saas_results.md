# Stage 3C held-out results (SaaS, frozen settings, run once)

Models: gpt-oss-20b (Groq), gpt-oss-120b (Cerebras); temperature 0, low reasoning effort. doc_rag: top-3 chunks, hybrid retriever. doc_rag_gated: whole-word phrase, case-insensitive, hyphens as spaces, plural -s ignored, FY<year> matches FY; a match inside a longer match for another chunk is dropped; retrieval fallback: None.

## SaaS glossary set (40 held-out questions that need a definition)

| System | gpt-oss-20b (Groq) | gpt-oss-120b (Cerebras) |
|---|---|---|
| zero_shot | **0.0%** (0/40), 1 SQL errors | **0.0%** (0/40) |
| doc_rag | **67.5%** (27/40), 1 SQL errors | **77.5%** (31/40) |
| doc_rag_gated | **80.0%** (32/40) | **95.0%** (38/40) |
| oracle_doc | **85.0%** (34/40) | **97.5%** (39/40) |

## SaaS plain set (20 held-out questions, no definitions needed)

| System | gpt-oss-20b (Groq) | gpt-oss-120b (Cerebras) |
|---|---|---|
| zero_shot | **95.0%** (19/20) | **100.0%** (20/20) |
| doc_rag | **100.0%** (20/20) | **100.0%** (20/20) |
| doc_rag_gated | **95.0%** (19/20) | **100.0%** (20/20) |

## All 60 SaaS questions (glossary + plain)

| System | gpt-oss-20b (Groq) | gpt-oss-120b (Cerebras) |
|---|---|---|
| zero_shot | **31.7%** (19/60), 1 SQL errors | **33.3%** (20/60) |
| doc_rag | **78.3%** (47/60), 1 SQL errors | **85.0%** (51/60) |
| doc_rag_gated | **85.0%** (51/60) | **96.7%** (58/60) |

## Accuracy by intent

| Set | System | Model | filter | count | aggregate | compare | trend |
|---|---|---|---|---|---|---|---|
| saas_glossary | zero_shot | gpt-oss-20b | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| saas_glossary | zero_shot | gpt-oss-120b | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| saas_glossary | doc_rag | gpt-oss-20b | 100.0% | 50.0% | 62.5% | 50.0% | 75.0% |
| saas_glossary | doc_rag | gpt-oss-120b | 100.0% | 62.5% | 62.5% | 62.5% | 100.0% |
| saas_glossary | doc_rag_gated | gpt-oss-20b | 75.0% | 100.0% | 75.0% | 75.0% | 75.0% |
| saas_glossary | doc_rag_gated | gpt-oss-120b | 87.5% | 100.0% | 100.0% | 87.5% | 100.0% |
| saas_glossary | oracle_doc | gpt-oss-20b | 100.0% | 100.0% | 75.0% | 75.0% | 75.0% |
| saas_glossary | oracle_doc | gpt-oss-120b | 100.0% | 100.0% | 100.0% | 87.5% | 100.0% |
| saas_plain | zero_shot | gpt-oss-20b | 100.0% | 100.0% | 100.0% | 75.0% | 100.0% |
| saas_plain | zero_shot | gpt-oss-120b | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| saas_plain | doc_rag | gpt-oss-20b | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| saas_plain | doc_rag | gpt-oss-120b | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| saas_plain | doc_rag_gated | gpt-oss-20b | 100.0% | 100.0% | 100.0% | 75.0% | 100.0% |
| saas_plain | doc_rag_gated | gpt-oss-120b | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| saas_all | zero_shot | gpt-oss-20b | 33.3% | 33.3% | 33.3% | 25.0% | 33.3% |
| saas_all | zero_shot | gpt-oss-120b | 33.3% | 33.3% | 33.3% | 33.3% | 33.3% |
| saas_all | doc_rag | gpt-oss-20b | 100.0% | 66.7% | 75.0% | 66.7% | 83.3% |
| saas_all | doc_rag | gpt-oss-120b | 100.0% | 75.0% | 75.0% | 75.0% | 100.0% |
| saas_all | doc_rag_gated | gpt-oss-20b | 83.3% | 100.0% | 83.3% | 75.0% | 83.3% |
| saas_all | doc_rag_gated | gpt-oss-120b | 91.7% | 100.0% | 100.0% | 91.7% | 100.0% |

## Which definitions reached the prompt

Model-independent. *All required* and *exact* are over questions that need a definition; *precision* is the share of sent definitions that the question needed.

| Set | System | All required | Exact set | Precision | Definitions per question | Questions given an unneeded definition | Questions missing a needed one |
|---|---|---|---|---|---|---|---|
| saas_glossary | doc_rag | 75.0% | 0.0% | 31.7% | 3.00 | 40 | 10 |
| saas_glossary | doc_rag_gated | 97.5% | 95.0% | 98.0% | 1.25 | 1 | 1 |
| saas_plain | doc_rag | n/a | n/a | 0.0% | 3.00 | 20 | 0 |
| saas_plain | doc_rag_gated | n/a | n/a | n/a | 0.00 | 0 | 0 |

doc_rag_gated questions whose definitions differ from the required set:

- saas_glossary sg_f03: required ss_recently_churned; given ss_churned, ss_recently_churned
- saas_glossary sg_f04: required ss_overprovisioned; given none

## doc_rag (ungated) vs doc_rag_gated, question by question

| Set | Model | Both right | Only ungated right | Only gated right | Both wrong | Fixed by gating | Broken by gating |
|---|---|---|---|---|---|---|---|
| saas_glossary | gpt-oss-20b | 22 | 5 | 10 | 3 | sg_a07, sg_a08, sg_c02, sg_c03, sg_c04, sg_c05, sg_p01, sg_p02, sg_t02, sg_t06 | sg_a06, sg_f04, sg_f08, sg_t07, sg_t08 |
| saas_glossary | gpt-oss-120b | 30 | 1 | 8 | 1 | sg_a05, sg_a07, sg_a08, sg_c03, sg_c04, sg_c05, sg_p02, sg_p04 | sg_f04 |
| saas_plain | gpt-oss-20b | 19 | 1 | 0 | 0 | – | sp_p03 |
| saas_plain | gpt-oss-120b | 20 | 0 | 0 | 0 | – | – |

## Small vs. large (gpt-oss-20b vs. gpt-oss-120b), question by question

| Set | System | Both right | Only small | Only large | Both wrong |
|---|---|---|---|---|---|
| saas_glossary | zero_shot | 0 | 0 | 0 | 40 |
| saas_glossary | doc_rag | 25 | 2 | 6 | 7 |
| saas_glossary | doc_rag_gated | 31 | 1 | 7 | 1 |
| saas_glossary | oracle_doc | 33 | 1 | 6 | 0 |
| saas_plain | zero_shot | 19 | 0 | 1 | 0 |
| saas_plain | doc_rag | 20 | 0 | 0 | 0 |
| saas_plain | doc_rag_gated | 19 | 0 | 1 | 0 |

## Zero-shot wrong answers that took the everyday reading (matched `naive_sql`)

| Set | Model | Wrong | Matched the naive reading |
|---|---|---|---|
| saas_glossary | gpt-oss-20b (Groq) | 40 | 12 |
| saas_glossary | gpt-oss-120b (Cerebras) | 40 | 10 |

## Calls and tokens behind this part

Every cached call this part's prompts use (including ones reused from Stage 3B). Latency is not comparable across providers.

| Model | Provider | Calls | Total tokens | Cached prompt tokens | Client median | Prompt tokens/call | Completion tokens/call |
|---|---|---|---|---|---|---|---|
| gpt-oss-20b | Groq | 165 | 137,808 | 13,312 | 0.44s | 722 | 113 |
| gpt-oss-120b | Cerebras | 165 | 136,845 | 94,464 | 0.30s | 722 | 107 |

Cerebras cost estimate for these gpt-oss-120b calls: **$0.054** (at Stage 3B's observed $0.18 per 457K tokens).

Every wrong answer is in `stage3c_saas_failures.csv`.
