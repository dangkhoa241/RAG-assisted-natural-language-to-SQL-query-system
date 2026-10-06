# BERT intent accuracy on the held-out SaaS questions

The fine-tuned intent classifier was never trained or tuned on SaaS data. Dev = the 60 glossary + 120 original healthcare/retail questions, for reference.

| Questions | n | Accuracy | filter | count | aggregate | compare | trend |
|---|---|---|---|---|---|---|---|
| SaaS (all) | 60 | **75.0%** | 91.7% | 100.0% | 91.7% | 75.0% | 16.7% |
| SaaS glossary | 40 | 75.0% | | | | | |
| SaaS plain | 20 | 75.0% | | | | | |
| Dev | 180 | 86.7% | 91.7% | 100.0% | 97.2% | 91.7% | 52.8% |

Confusions on SaaS (true -> predicted): trend -> aggregate (6), trend -> count (4), compare -> count (2), filter -> count (1), aggregate -> count (1), compare -> aggregate (1)

| ID | Question | True | Predicted |
|---|---|---|---|
| sg_f01 | list the strategic accounts in EMEA | filter | count |
| sg_a01 | what is our ARR? | aggregate | count |
| sg_p02 | high-touch vs self-serve accounts: average MRR | compare | aggregate |
| sg_p06 | Starter vs Pro: how many expansion-ready accounts does each have? | compare | count |
| sg_t02 | number of signups per fiscal year | trend | count |
| sg_t03 | monthly number of churned accounts in 2024 | trend | count |
| sg_t04 | ARR by signup year | trend | aggregate |
| sg_t05 | logo churn rate by signup year | trend | aggregate |
| sg_t06 | average tenure by signup year | trend | aggregate |
| sg_t07 | how has the net promoter score changed by signup year? | trend | aggregate |
| sp_p02 | EMEA vs APAC: number of accounts | compare | count |
| sp_t01 | number of signups per month in 2024 | trend | count |
| sp_t02 | number of cancellations per year | trend | count |
| sp_t03 | total MRR by signup year | trend | aggregate |
| sp_t04 | average seats per account by signup year | trend | aggregate |
