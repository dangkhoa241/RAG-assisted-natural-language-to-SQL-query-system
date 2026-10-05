# Text-to-SQL benchmark results

Model: `openai/gpt-oss-120b` (reasoning effort low, temperature 0). Embeddings: `sentence-transformers/all-MiniLM-L6-v2`. llm_rag uses k=3. 120 questions.

## Execution accuracy

| System | Overall | Healthcare | Retail | SQL errors | Rejected by safety | Fallbacks |
|---|---|---|---|---|---|---|
| rule_based | **30.8%** (37/120) | 36.7% | 25.0% | 0 (0.0%) | 0 | 0 |
| rule_based_gold_intent | **32.5%** (39/120) | 38.3% | 26.7% | 0 (0.0%) | 0 | 0 |
| llm_zero_shot | **99.2%** (119/120) | 98.3% | 100.0% | 0 (0.0%) | 0 | 0 |
| llm_rag | **93.3%** (112/120) | 96.7% | 90.0% | 1 (0.8%) | 1 | 1 |

## Accuracy by intent

| System | filter | count | aggregate | compare | trend |
|---|---|---|---|---|---|
| rule_based | 21% | 38% | 54% | 17% | 25% |
| rule_based_gold_intent | 21% | 38% | 54% | 17% | 33% |
| llm_zero_shot | 100% | 100% | 100% | 96% | 100% |
| llm_rag | 92% | 100% | 88% | 88% | 100% |

## llm_rag ablation: number of retrieved examples

| k | Overall | filter | count | aggregate | compare | trend | SQL errors |
|---|---|---|---|---|---|---|---|
| 0 | **99.2%** | 100% | 100% | 100% | 96% | 100% | 0 |
| 1 | **80.0%** | 92% | 96% | 58% | 83% | 71% | 2 |
| 3 | **93.3%** | 92% | 100% | 88% | 88% | 100% | 1 |
| 5 | **95.8%** | 96% | 100% | 92% | 96% | 96% | 0 |

k=0 sends exactly the llm_zero_shot prompt, so its row matches llm_zero_shot.

