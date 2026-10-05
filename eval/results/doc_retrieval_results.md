# Glossary retrieval results

60 glossary questions; each dataset's glossary has 26 (healthcare) / 26 (retail) chunks, including distractors. Dense = `sentence-transformers/all-MiniLM-L6-v2`, BM25 = rank_bm25 BM25Okapi, hybrid = reciprocal rank fusion (k=60).

recall@k = share of required chunks in the top k; all@k = share of questions with every required chunk in the top k.

| Retriever | recall@1 | recall@3 | recall@5 | all@3 | all@5 | MRR | recall@3 healthcare | recall@3 retail |
|---|---|---|---|---|---|---|---|---|
| dense | 63.3% | **90.8%** | 95.8% | 85.0% | 93.3% | 0.893 | 90.0% | 91.7% |
| bm25 | 50.8% | **89.2%** | 95.8% | 83.3% | 93.3% | 0.821 | 85.0% | 93.3% |
| hybrid | 63.3% | **93.3%** | 100.0% | 90.0% | 100.0% | 0.894 | 91.7% | 95.0% |

Best retriever by all@3: **hybrid** (used for doc_rag with k=3).

## Questions with a required chunk missing from the top 3

| Question | Retriever | Required | Retrieved top 3 |
|---|---|---|---|
| hg_f02 | dense | hc_high_cost, hc_senior_patient | hc_high_cost_condition, hc_high_cost, hc_cost_overrun |
| hg_f03 | dense | hc_premium_insurer, hc_flagged_result | hc_flagged_claim, hc_flagged_result, hc_high_cost_condition |
| hg_c03 | dense | hc_government_payer, hc_fiscal_year | hc_fiscal_year, hc_high_cost, hc_fiscal_quarter |
| hg_a05 | dense | hc_premium_insurer | hc_high_cost_condition, hc_cost_overrun, hc_government_payer |
| hg_p05 | dense | hc_senior_patient, hc_young_adult | hc_senior_care_tier, hc_senior_patient, hc_high_risk_patient |
| rg_c02 | dense | rt_b2b_order, rt_core_market | rt_b2b_order, rt_high_value_order, rt_bulk_buyer |
| rg_a01 | dense | rt_net_revenue | rt_core_market, rt_net_sales, rt_gross_sales |
| rg_p05 | dense | rt_mature_shopper, rt_average_order_value | rt_mature_shopper, rt_high_value_order, rt_high_value_customer |
| rg_t02 | dense | rt_fiscal_year, rt_net_revenue | rt_fiscal_year, rt_net_sales, rt_gross_sales |
| hg_f02 | bm25 | hc_high_cost, hc_senior_patient | hc_senior_care_tier, hc_high_cost_condition, hc_high_cost |
| hg_f03 | bm25 | hc_premium_insurer, hc_flagged_result | hc_senior_care_tier, hc_flagged_claim, hc_premium_room |
| hg_c03 | bm25 | hc_government_payer, hc_fiscal_year | hc_fiscal_year, hc_flu_season, hc_fiscal_quarter |
| hg_c04 | bm25 | hc_rare_blood_type, hc_icu_room | hc_rare_blood_type, hc_senior_care_tier, hc_unplanned_admission |
| hg_a05 | bm25 | hc_premium_insurer | hc_cost_overrun, hc_high_cost_condition, hc_premium_room |
| hg_p05 | bm25 | hc_senior_patient, hc_young_adult | hc_senior_care_tier, hc_premium_room, hc_young_adult |
| hg_t06 | bm25 | hc_cost_overrun, hc_fiscal_year | hc_fiscal_quarter, hc_fiscal_year, hc_billable_days |
| rg_f02 | bm25 | rt_deep_discount, rt_bulk_order | rt_deep_discount, rt_markdown, rt_bulk_buyer |
| rg_p06 | bm25 | rt_detractor | rt_core_market, rt_fiscal_year, rt_markdown |
| rg_t02 | bm25 | rt_fiscal_year, rt_net_revenue | rt_fiscal_year, rt_fiscal_quarter, rt_average_order_value |
| hg_f02 | hybrid | hc_high_cost, hc_senior_patient | hc_high_cost_condition, hc_senior_care_tier, hc_high_cost |
| hg_f03 | hybrid | hc_premium_insurer, hc_flagged_result | hc_flagged_claim, hc_flagged_result, hc_premium_room |
| hg_c03 | hybrid | hc_government_payer, hc_fiscal_year | hc_fiscal_year, hc_flu_season, hc_fiscal_quarter |
| hg_a05 | hybrid | hc_premium_insurer | hc_high_cost_condition, hc_cost_overrun, hc_premium_room |
| rg_p06 | hybrid | rt_detractor | rt_core_market, rt_returning_customer, rt_return_rate |
| rg_t02 | hybrid | rt_fiscal_year, rt_net_revenue | rt_fiscal_year, rt_net_sales, rt_fiscal_quarter |
