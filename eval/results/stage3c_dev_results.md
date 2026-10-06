# Stage 3C dev-set results (tuning data: healthcare + retail)

Models: gpt-oss-20b (Groq), gpt-oss-120b (Cerebras); temperature 0, low reasoning effort. doc_rag: top-3 chunks, hybrid retriever. doc_rag_gated: whole-word phrase, case-insensitive, hyphens as spaces, plural -s ignored, FY<year> matches FY; a match inside a longer match for another chunk is dropped; retrieval fallback: None.

## Dev glossary set (60 healthcare + retail questions that need a definition)

| System | gpt-oss-20b (Groq) | gpt-oss-120b (Cerebras) |
|---|---|---|
| zero_shot | **10.0%** (6/60) | **10.0%** (6/60), 1 SQL errors |
| doc_rag | **86.7%** (52/60) | **88.3%** (53/60) |
| doc_rag_gated | **95.0%** (57/60), 1 SQL errors | **91.7%** (55/60) |
| oracle_doc | **95.0%** (57/60), 1 SQL errors | **95.0%** (57/60) |

## Dev original set (120 Stage 2 questions, no definitions needed)

| System | gpt-oss-20b (Groq) | gpt-oss-120b (Cerebras) |
|---|---|---|
| zero_shot | **98.3%** (118/120) | **99.2%** (119/120) |
| doc_rag | **94.2%** (113/120) | **95.0%** (114/120) |
| doc_rag_gated | **98.3%** (118/120) | **99.2%** (119/120) |

## Accuracy by intent

| Set | System | Model | filter | count | aggregate | compare | trend |
|---|---|---|---|---|---|---|---|
| glossary | zero_shot | gpt-oss-20b | 8.3% | 8.3% | 8.3% | 8.3% | 16.7% |
| glossary | zero_shot | gpt-oss-120b | 8.3% | 16.7% | 0.0% | 8.3% | 16.7% |
| glossary | doc_rag | gpt-oss-20b | 83.3% | 83.3% | 91.7% | 91.7% | 83.3% |
| glossary | doc_rag | gpt-oss-120b | 83.3% | 83.3% | 91.7% | 83.3% | 100.0% |
| glossary | doc_rag_gated | gpt-oss-20b | 100.0% | 100.0% | 100.0% | 100.0% | 75.0% |
| glossary | doc_rag_gated | gpt-oss-120b | 91.7% | 91.7% | 100.0% | 91.7% | 83.3% |
| glossary | oracle_doc | gpt-oss-20b | 100.0% | 100.0% | 100.0% | 100.0% | 75.0% |
| glossary | oracle_doc | gpt-oss-120b | 100.0% | 91.7% | 100.0% | 91.7% | 91.7% |
| original | zero_shot | gpt-oss-20b | 100.0% | 100.0% | 100.0% | 91.7% | 100.0% |
| original | zero_shot | gpt-oss-120b | 100.0% | 100.0% | 100.0% | 95.8% | 100.0% |
| original | doc_rag | gpt-oss-20b | 95.8% | 95.8% | 91.7% | 87.5% | 100.0% |
| original | doc_rag | gpt-oss-120b | 91.7% | 100.0% | 95.8% | 91.7% | 95.8% |
| original | doc_rag_gated | gpt-oss-20b | 100.0% | 100.0% | 100.0% | 91.7% | 100.0% |
| original | doc_rag_gated | gpt-oss-120b | 100.0% | 100.0% | 100.0% | 95.8% | 100.0% |

## Which definitions reached the prompt

Model-independent. *All required* and *exact* are over questions that need a definition; *precision* is the share of sent definitions that the question needed.

| Set | System | All required | Exact set | Precision | Definitions per question | Questions given an unneeded definition | Questions missing a needed one |
|---|---|---|---|---|---|---|---|
| glossary | doc_rag | 90.0% | 0.0% | 45.0% | 3.00 | 60 | 6 |
| glossary | doc_rag_gated | 100.0% | 100.0% | 100.0% | 1.45 | 0 | 0 |
| original | doc_rag | n/a | n/a | 0.0% | 3.00 | 120 | 0 |
| original | doc_rag_gated | n/a | n/a | n/a | 0.00 | 0 | 0 |

## doc_rag (ungated) vs doc_rag_gated, question by question

| Set | Model | Both right | Only ungated right | Only gated right | Both wrong | Fixed by gating | Broken by gating |
|---|---|---|---|---|---|---|---|
| glossary | gpt-oss-20b | 51 | 1 | 6 | 2 | hg_a05, hg_c03, hg_f02, hg_f03, rg_c06, rg_p03 | rg_t02 |
| glossary | gpt-oss-120b | 50 | 3 | 5 | 2 | hg_a05, hg_c03, hg_f02, hg_f03, rg_p02 | rg_f06, rg_t01, rg_t02 |
| original | gpt-oss-20b | 113 | 0 | 5 | 2 | h_a07, h_c10, h_f07, r_a03, r_p10 | – |
| original | gpt-oss-120b | 113 | 1 | 6 | 0 | h_a07, h_f03, h_f07, r_p02, r_p10, r_t10 | h_p06 |

## Small vs. large (gpt-oss-20b vs. gpt-oss-120b), question by question

| Set | System | Both right | Only small | Only large | Both wrong |
|---|---|---|---|---|---|
| glossary | zero_shot | 4 | 2 | 2 | 52 |
| glossary | doc_rag | 49 | 3 | 4 | 4 |
| glossary | doc_rag_gated | 53 | 4 | 2 | 1 |
| glossary | oracle_doc | 55 | 2 | 2 | 1 |
| original | zero_shot | 118 | 0 | 1 | 1 |
| original | doc_rag | 111 | 2 | 3 | 4 |
| original | doc_rag_gated | 118 | 0 | 1 | 1 |

## Zero-shot wrong answers that took the everyday reading (matched `naive_sql`)

| Set | Model | Wrong | Matched the naive reading |
|---|---|---|---|
| glossary | gpt-oss-20b (Groq) | 54 | 14 |
| glossary | gpt-oss-120b (Cerebras) | 54 | 19 |

## Calls and tokens behind this part

Every cached call this part's prompts use (including ones reused from Stage 3B). Latency is not comparable across providers.

| Model | Provider | Calls | Total tokens | Cached prompt tokens | Client median | Prompt tokens/call | Completion tokens/call |
|---|---|---|---|---|---|---|---|
| gpt-oss-20b | Groq | 430 | 352,405 | 79,104 | 0.40s | 731 | 88 |
| gpt-oss-120b | Cerebras | 430 | 356,668 | 262,144 | 0.32s | 731 | 98 |

Cerebras cost estimate for these gpt-oss-120b calls: **$0.140** (at Stage 3B's observed $0.18 per 457K tokens).

Every wrong answer is in `stage3c_dev_failures.csv`.
