# SaaS held-out benchmark: review sample

15 of the 40 glossary questions in `saas_questions.jsonl` (3 per intent), for a human check that
(a) the question uses a business term without restating it, (b) the gold SQL implements the glossary
definition, and (c) the naive SQL is a plausible everyday reading that gives a different answer.
The 20 plain questions (no definition needed) are listed at the end.

The data is `data/saas_subscriptions.csv` (`make_saas_dataset.py`, seed 11); the glossary is
`docs/glossary/saas.md`, with the reporting as-of date 2025-06-30. Regenerate with
`python eval/sql_benchmark/build_saas_questions.py`.

## Filter

### `sg_f01`: "list the strategic accounts in EMEA"

**Required definitions**

- `ss_strategic_account` **Strategic account**: An account on the Enterprise `Plan` with an `MRR` of at least 5,000.

**Gold SQL**

```sql
SELECT * FROM data WHERE (`Plan` = 'Enterprise' AND `MRR` >= 5000) AND `Region` = 'EMEA'
```

**16 rows**; first 3:

| Company Name | Plan | MRR | Region |
|---|---|---|---|
| Crescent Energy | Enterprise | 5259.54 | EMEA |
| Cedar Solutions | Enterprise | 13879.22 | EMEA |
| Quartz Clinics | Enterprise | 11252.45 | EMEA |

**Naive SQL**

```sql
SELECT * FROM data WHERE `Plan` = 'Enterprise' AND `Region` = 'EMEA'
```

**39 rows**; first 3:

| Company Name | Plan | Region |
|---|---|---|
| Red Group | Enterprise | EMEA |
| Crescent Energy | Enterprise | EMEA |
| Cedar Solutions | Enterprise | EMEA |

### `sg_f03`: "show the recently churned accounts in North America"

**Required definitions**

- `ss_recently_churned` **Recently churned**: A churned account (cancelled more than 14 days after signup) whose `Cancel Date` is on or after 2025-04-01, the start of the last quarter before the reporting date.

**Gold SQL**

```sql
SELECT * FROM data WHERE ((`Cancel Date` IS NOT NULL AND (julianday(`Cancel Date`) - julianday(`Signup Date`)) > 14) AND `Cancel Date` >= '2025-04-01') AND `Region` = 'North America'
```

**22 rows**; first 3:

| Company Name | Signup Date | Cancel Date | Region |
|---|---|---|---|
| Falcon Dynamics | 2021-04-30 | 2025-05-03 | North America |
| Echo Logistics | 2021-10-01 | 2025-06-08 | North America |
| Harbor Health | 2022-06-27 | 2025-05-10 | North America |

**Naive SQL**

```sql
SELECT * FROM data WHERE `Cancel Date` >= '2025-04-01' AND `Region` = 'North America'
```

**24 rows**; first 3:

| Company Name | Cancel Date | Region |
|---|---|---|
| Falcon Dynamics | 2025-05-03 | North America |
| Echo Logistics | 2025-06-08 | North America |
| Harbor Health | 2025-05-10 | North America |

### `sg_f08`: "show support-heavy SMB accounts"

**Required definitions**

- `ss_support_heavy` **Support-heavy account**: An account with at least 5 `Support Tickets` and more than one ticket per five seats (`Support Tickets` > 0.2 × `Seats`).
- `ss_smb_account` **SMB account**: An account with fewer than 20 `Seats`, regardless of `Plan` (a Pro account with 12 seats is SMB).

**Gold SQL**

```sql
SELECT * FROM data WHERE (`Support Tickets` >= 5 AND `Support Tickets` > 0.2 * `Seats`) AND (`Seats` < 20)
```

**81 rows**; first 3:

| Company Name | Seats | Support Tickets |
|---|---|---|
| Iron Group | 10 | 5 |
| Stone Finance | 19 | 5 |
| Bright Systems | 17 | 5 |

**Naive SQL**

```sql
SELECT * FROM data WHERE `Support Tickets` >= 5 AND `Plan` = 'Starter'
```

**7 rows**; first 3:

| Company Name | Plan | Support Tickets |
|---|---|---|
| Stone Media | Starter | 5 |
| Coral Retail | Starter | 5 |
| Clear Foods | Starter | 6 |

## Count

### `sg_c01`: "how many active accounts do we have?"

**Required definitions**

- `ss_active_account` **Active account**: An account with no `Cancel Date` whose `Last Login Date` is on or after 2025-05-31 (a login within 30 days of the reporting date 2025-06-30). Subscribed accounts that have not logged in since then are not active.

**Gold SQL**

```sql
SELECT COUNT(*) FROM data WHERE (`Cancel Date` IS NULL AND `Last Login Date` >= '2025-05-31')
```

**1 row**; first 1:

| COUNT(*) |
|---|
| 554 |

**Naive SQL**

```sql
SELECT COUNT(*) FROM data WHERE `Cancel Date` IS NULL
```

**1 row**; first 1:

| COUNT(*) |
|---|
| 713 |

### `sg_c03`: "how many accounts churned in fiscal year 2024?"

**Required definitions**

- `ss_churned` **Churned account**: An account whose `Cancel Date` is more than 14 days after its `Signup Date`. Cancellations within the first 14 days are trial drop-offs, not churn.
- `ss_fiscal_year` **Fiscal year**: The fiscal year runs from November 1 to October 31 and is named after the calendar year in which it ends, based on the date being reported (usually `Signup Date`). FY2024 = 2023-11-01 through 2024-10-31.

**Gold SQL**

```sql
SELECT COUNT(*) FROM data WHERE (`Cancel Date` IS NOT NULL AND (julianday(`Cancel Date`) - julianday(`Signup Date`)) > 14) AND (CAST(strftime('%Y', `Cancel Date`) AS INTEGER) + (CAST(strftime('%m', `Cancel Date`) AS INTEGER) >= 11)) = 2024
```

**1 row**; first 1:

| COUNT(*) |
|---|
| 120 |

**Naive SQL**

```sql
SELECT COUNT(*) FROM data WHERE `Cancel Date` IS NOT NULL AND strftime('%Y', `Cancel Date`) = '2024'
```

**1 row**; first 1:

| COUNT(*) |
|---|
| 157 |

### `sg_c07`: "how many recent signups are still subscribed?"

**Required definitions**

- `ss_recent_signup` **Recent signup**: Any account with a `Signup Date` on or after 2025-06-01, including trial drop-offs. Used by the onboarding team; sales reporting uses new logos instead.

**Gold SQL**

```sql
SELECT COUNT(*) FROM data WHERE (`Signup Date` >= '2025-06-01') AND `Cancel Date` IS NULL
```

**1 row**; first 1:

| COUNT(*) |
|---|
| 23 |

**Naive SQL**

```sql
SELECT COUNT(*) FROM data WHERE `Signup Date` >= '2025-04-01' AND `Cancel Date` IS NULL
```

**1 row**; first 1:

| COUNT(*) |
|---|
| 70 |

## Aggregate

### `sg_a01`: "what is our ARR?"

**Required definitions**

- `ss_arr` **ARR**: Annual recurring revenue counts only accounts on the Annual `Billing Cycle` that have no `Cancel Date`: ARR = SUM(`MRR`) × 12 over those accounts. Monthly-billed accounts are excluded from ARR.

**Gold SQL**

```sql
SELECT 12 * SUM(`MRR`) FROM data WHERE (`Billing Cycle` = 'Annual' AND `Cancel Date` IS NULL)
```

**1 row**; first 1:

| 12 * SUM(`MRR`) |
|---|
| 10685535.84 |

**Naive SQL**

```sql
SELECT 12 * SUM(`MRR`) FROM data WHERE `Cancel Date` IS NULL
```

**1 row**; first 1:

| 12 * SUM(`MRR`) |
|---|
| 13531687.44 |

### `sg_a03`: "seat utilization by region"

**Required definitions**

- `ss_seat_utilization` **Seat utilization**: SUM(`Seats Used`) / SUM(`Seats`) × 100 over the accounts in scope: a pooled percentage, not the average of each account's own ratio.

**Gold SQL**

```sql
SELECT `Region`, 100.0 * SUM(`Seats Used`) / SUM(`Seats`) FROM data GROUP BY `Region`
```

**4 rows**; first 3:

| Region | 100.0 * SUM(`Seats Used`) / SUM(`Seats`) |
|---|---|
| APAC | 58.7 |
| EMEA | 72.58 |
| LATAM | 66.44 |

**Naive SQL**

```sql
SELECT `Region`, AVG(100.0 * `Seats Used` / `Seats`) FROM data GROUP BY `Region`
```

**4 rows**; first 3:

| Region | AVG(100.0 * `Seats Used` / `Seats`) |
|---|---|
| APAC | 66.78 |
| EMEA | 68.29 |
| LATAM | 66.17 |

### `sg_a06`: "net promoter score by industry"

**Required definitions**

- `ss_net_promoter_score` **Net promoter score**: Computed over accounts with a non-empty `NPS`: 100 × (share scoring 9 or 10) − 100 × (share scoring 0–5).

**Gold SQL**

```sql
SELECT `Industry`, 100.0 * (SUM(`NPS` >= 9) - SUM(`NPS` <= 5)) / COUNT(`NPS`) FROM data GROUP BY `Industry`
```

**8 rows**; first 3:

| Industry | 100.0 * (SUM(`NPS` >= 9) - SUM(`NPS` <= 5)) / COUNT(`NPS`) |
|---|---|
| Education | 10.84 |
| Financial Services | 17.78 |
| Healthcare | 25.25 |

**Naive SQL**

```sql
SELECT `Industry`, 100.0 * (SUM(`NPS` >= 9) - SUM(`NPS` <= 6)) / COUNT(`NPS`) FROM data GROUP BY `Industry`
```

**8 rows**; first 3:

| Industry | 100.0 * (SUM(`NPS` >= 9) - SUM(`NPS` <= 6)) / COUNT(`NPS`) |
|---|---|
| Education | -2.41 |
| Financial Services | 7.78 |
| Healthcare | 18.18 |

## Compare

### `sg_p01`: "compare the logo churn rate of monthly and annual billing"

**Required definitions**

- `ss_logo_churn_rate` **Logo churn rate**: 100 × (number of churned accounts) / (number of accounts that are not trial drop-offs). A churned account cancelled more than 14 days after signup; a trial drop-off cancelled within 14 days. Counts accounts, not MRR.

**Gold SQL**

```sql
SELECT `Billing Cycle`, 100.0 * SUM(CASE WHEN (`Cancel Date` IS NOT NULL AND (julianday(`Cancel Date`) - julianday(`Signup Date`)) > 14) THEN 1 ELSE 0 END) / SUM(CASE WHEN (`Cancel Date` IS NULL OR (julianday(`Cancel Date`) - julianday(`Signup Date`)) > 14) THEN 1 ELSE 0 END) FROM data GROUP BY `Billing Cycle`
```

**2 rows**; first 2:

| Billing Cycle | 100.0 * SUM(CASE WHEN (`Cancel Date` IS NOT NULL AND (julianday(`Cancel Date`) - julianday(`Signup Date`)) > 14) THEN 1 ELSE 0 END) / SUM(CASE WHEN (`Cancel Date` IS NULL OR (julianday(`Cancel Date`) - julianday(`Signup Date`)) > 14) THEN 1 ELSE 0 END) |
|---|---|
| Annual | 25.11 |
| Monthly | 42.28 |

**Naive SQL**

```sql
SELECT `Billing Cycle`, 100.0 * AVG(`Cancel Date` IS NOT NULL) FROM data GROUP BY `Billing Cycle`
```

**2 rows**; first 2:

| Billing Cycle | 100.0 * AVG(`Cancel Date` IS NOT NULL) |
|---|---|
| Annual | 29.0 |
| Monthly | 47.83 |

### `sg_p02`: "high-touch vs self-serve accounts: average MRR"

**Required definitions**

- `ss_high_touch` **High-touch account**: `Acquisition Channel` is Outbound Sales, Partner, or Event (event leads are worked by the sales team).
- `ss_self_serve` **Self-serve account**: `Acquisition Channel` is Organic Search, Paid Search, or Referral: the account signed up without a salesperson.

**Gold SQL**

```sql
SELECT AVG(`MRR`) FROM data GROUP BY `Acquisition Channel` IN ('Outbound Sales', 'Partner', 'Event')
```

**2 rows**; first 2:

| AVG(`MRR`) |
|---|
| 566.18 |
| 2287.59 |

**Naive SQL**

```sql
SELECT AVG(`MRR`) FROM data GROUP BY `Acquisition Channel` = 'Outbound Sales'
```

**2 rows**; first 2:

| AVG(`MRR`) |
|---|
| 812.05 |
| 3456.47 |

### `sg_p07`: "compare the number of detractors and passives"

**Required definitions**

- `ss_detractor` **Detractor**: An account whose `NPS` is 5 or lower. Unlike the textbook rule, a score of 6 counts as passive. Accounts with an empty `NPS` did not respond and are neither.
- `ss_passive` **Passive respondent**: An account whose `NPS` is 6, 7 or 8. Passives count in neither side of the net promoter score.

**Gold SQL**

```sql
SELECT COUNT(*) FROM data WHERE `NPS` <= 8 GROUP BY `NPS` <= 5
```

**2 rows**; first 2:

| COUNT(*) |
|---|
| 359 |
| 162 |

**Naive SQL**

```sql
SELECT COUNT(*) FROM data WHERE `NPS` <= 8 GROUP BY `NPS` <= 6
```

**2 rows**; first 2:

| COUNT(*) |
|---|
| 284 |
| 237 |

## Trend

### `sg_t02`: "number of signups per fiscal year"

**Required definitions**

- `ss_fiscal_year` **Fiscal year**: The fiscal year runs from November 1 to October 31 and is named after the calendar year in which it ends, based on the date being reported (usually `Signup Date`). FY2024 = 2023-11-01 through 2024-10-31.

**Gold SQL**

```sql
SELECT COUNT(*) FROM data GROUP BY (CAST(strftime('%Y', `Signup Date`) AS INTEGER) + (CAST(strftime('%m', `Signup Date`) AS INTEGER) >= 11)) ORDER BY (CAST(strftime('%Y', `Signup Date`) AS INTEGER) + (CAST(strftime('%m', `Signup Date`) AS INTEGER) >= 11))
```

**5 rows**; first 3:

| COUNT(*) |
|---|
| 150 |
| 228 |
| 272 |

**Naive SQL**

```sql
SELECT COUNT(*) FROM data GROUP BY strftime('%Y', `Signup Date`) ORDER BY strftime('%Y', `Signup Date`)
```

**5 rows**; first 3:

| COUNT(*) |
|---|
| 191 |
| 231 |
| 277 |

### `sg_t05`: "logo churn rate by signup year"

**Required definitions**

- `ss_logo_churn_rate` **Logo churn rate**: 100 × (number of churned accounts) / (number of accounts that are not trial drop-offs). A churned account cancelled more than 14 days after signup; a trial drop-off cancelled within 14 days. Counts accounts, not MRR.

**Gold SQL**

```sql
SELECT strftime('%Y', `Signup Date`), 100.0 * SUM(CASE WHEN (`Cancel Date` IS NOT NULL AND (julianday(`Cancel Date`) - julianday(`Signup Date`)) > 14) THEN 1 ELSE 0 END) / SUM(CASE WHEN (`Cancel Date` IS NULL OR (julianday(`Cancel Date`) - julianday(`Signup Date`)) > 14) THEN 1 ELSE 0 END) FROM data GROUP BY 1 ORDER BY 1
```

**5 rows**; first 3:

| strftime('%Y', `Signup Date`) | 100.0 * SUM(CASE WHEN (`Cancel Date` IS NOT NULL AND (julianday(`Cancel Date`) - julianday(`Signup Date`)) > 14) THEN 1 ELSE 0 END) / SUM(CASE WHEN (`Cancel Date` IS NULL OR (julianday(`Cancel Date`) - julianday(`Signup Date`)) > 14) THEN 1 ELSE 0 END) |
|---|---|
| 2021 | 58.1 |
| 2022 | 45.75 |
| 2023 | 39.61 |

**Naive SQL**

```sql
SELECT strftime('%Y', `Signup Date`), 100.0 * AVG(`Cancel Date` IS NOT NULL) FROM data GROUP BY 1 ORDER BY 1
```

**5 rows**; first 3:

| strftime('%Y', `Signup Date`) | 100.0 * AVG(`Cancel Date` IS NOT NULL) |
|---|---|
| 2021 | 60.73 |
| 2022 | 50.22 |
| 2023 | 44.4 |

### `sg_t08`: "trial drop-offs per fiscal year"

**Required definitions**

- `ss_trial_dropoff` **Trial drop-off**: An account whose `Cancel Date` is no more than 14 days after its `Signup Date` (cancel date minus signup date ≤ 14 days). These are failed trials and never count as churn.
- `ss_fiscal_year` **Fiscal year**: The fiscal year runs from November 1 to October 31 and is named after the calendar year in which it ends, based on the date being reported (usually `Signup Date`). FY2024 = 2023-11-01 through 2024-10-31.

**Gold SQL**

```sql
SELECT COUNT(*) FROM data WHERE (`Cancel Date` IS NOT NULL AND (julianday(`Cancel Date`) - julianday(`Signup Date`)) <= 14) GROUP BY (CAST(strftime('%Y', `Signup Date`) AS INTEGER) + (CAST(strftime('%m', `Signup Date`) AS INTEGER) >= 11)) ORDER BY (CAST(strftime('%Y', `Signup Date`) AS INTEGER) + (CAST(strftime('%m', `Signup Date`) AS INTEGER) >= 11))
```

**5 rows**; first 3:

| COUNT(*) |
|---|
| 9 |
| 17 |
| 25 |

**Naive SQL**

```sql
SELECT COUNT(*) FROM data WHERE `Cancel Date` IS NOT NULL AND (julianday(`Cancel Date`) - julianday(`Signup Date`)) <= 30 GROUP BY strftime('%Y', `Signup Date`) ORDER BY strftime('%Y', `Signup Date`)
```

**5 rows**; first 3:

| COUNT(*) |
|---|
| 15 |
| 21 |
| 30 |

## Plain questions (no definition needed)

| ID | Intent | Question | Gold SQL | Gold rows |
|---|---|---|---|---|
| sp_f01 | filter | show Enterprise accounts in LATAM | `SELECT * FROM data WHERE `Plan` = 'Enterprise' AND `Region` = 'LATAM'` | 14 |
| sp_f02 | filter | list accounts with more than 500 seats | `SELECT * FROM data WHERE `Seats` > 500` | 7 |
| sp_f03 | filter | show accounts that signed up in March 2025 | `SELECT * FROM data WHERE strftime('%Y-%m', `Signup Date`) = '2025-03'` | 35 |
| sp_f04 | filter | list Pro accounts in the education industry that gave an NPS of 10 | `SELECT * FROM data WHERE `Plan` = 'Pro' AND `Industry` = 'Education' AND `NPS` = 10` | 4 |
| sp_c01 | count | how many accounts signed up in 2023? | `SELECT COUNT(*) FROM data WHERE strftime('%Y', `Signup Date`) = '2023'` | 1 |
| sp_c02 | count | how many accounts have a cancel date? | `SELECT COUNT(*) FROM data WHERE `Cancel Date` IS NOT NULL` | 1 |
| sp_c03 | count | how many accounts came from referrals? | `SELECT COUNT(*) FROM data WHERE `Acquisition Channel` = 'Referral'` | 1 |
| sp_c04 | count | how many accounts never answered the NPS survey? | `SELECT COUNT(*) FROM data WHERE `NPS` IS NULL` | 1 |
| sp_a01 | aggregate | average MRR by plan | `SELECT `Plan`, AVG(`MRR`) FROM data GROUP BY `Plan`` | 3 |
| sp_a02 | aggregate | total seats by region | `SELECT `Region`, SUM(`Seats`) FROM data GROUP BY `Region`` | 4 |
| sp_a03 | aggregate | average number of support tickets per industry | `SELECT `Industry`, AVG(`Support Tickets`) FROM data GROUP BY `Industry`` | 8 |
| sp_a04 | aggregate | what is the total MRR of accounts on annual billing? | `SELECT SUM(`MRR`) FROM data WHERE `Billing Cycle` = 'Annual'` | 1 |
| sp_p01 | compare | compare the average MRR of monthly and annual billing | `SELECT `Billing Cycle`, AVG(`MRR`) FROM data GROUP BY `Billing Cycle`` | 2 |
| sp_p02 | compare | EMEA vs APAC: number of accounts | `SELECT `Region`, COUNT(*) FROM data WHERE `Region` IN ('EMEA', 'APAC') GROUP BY `Region`` | 2 |
| sp_p03 | compare | do Enterprise accounts file more support tickets than Pro accounts on average? | `SELECT `Plan`, AVG(`Support Tickets`) FROM data WHERE `Plan` IN ('Enterprise', 'Pro') GROUP BY `Plan`` | 2 |
| sp_p04 | compare | compare the average NPS of software and healthcare accounts | `SELECT `Industry`, AVG(`NPS`) FROM data WHERE `Industry` IN ('Software', 'Healthcare') GROUP BY `Industry`` | 2 |
| sp_t01 | trend | number of signups per month in 2024 | `SELECT strftime('%Y-%m', `Signup Date`), COUNT(*) FROM data WHERE strftime('%Y', `Signup Date`) = '2024' GROUP BY 1 ORDER BY 1` | 12 |
| sp_t02 | trend | number of cancellations per year | `SELECT strftime('%Y', `Cancel Date`), COUNT(*) FROM data WHERE `Cancel Date` IS NOT NULL GROUP BY 1 ORDER BY 1` | 5 |
| sp_t03 | trend | total MRR by signup year | `SELECT strftime('%Y', `Signup Date`), SUM(`MRR`) FROM data GROUP BY 1 ORDER BY 1` | 5 |
| sp_t04 | trend | average seats per account by signup year | `SELECT strftime('%Y', `Signup Date`), AVG(`Seats`) FROM data GROUP BY 1 ORDER BY 1` | 5 |
