# Glossary benchmark: review sample

15 of the 60 questions in `glossary_questions.jsonl` (3 per intent, both datasets), for a human check that
(a) the question uses a business term without restating it, (b) the gold SQL implements the glossary
definition, and (c) a plausible-but-wrong reading of the term (the naive SQL) gives a different answer.

Regenerate the full set and its checks with `python eval/sql_benchmark/build_glossary_questions.py`.
Result-shape conventions: compare/trend questions over derived groups (e.g. premium vs standard insurers,
fiscal years) return only the metric column, so any group label the model chooses still matches;
fiscal-year trends are compared in period order.

## Filter

### 1. `hg_f01` (healthcare): "show long-stay admissions in the ICU"

**Required definitions**

- `hc_long_stay` **Long-stay admission**: An admission whose `Discharge Date` is at least 21 days after its `Date of Admission`. Long-stay admissions are reviewed by the utilization committee.
- `hc_icu_room` **ICU room**: `Room Number` from 400 to 449 inclusive. These rooms are the intensive care unit.

**Gold SQL**

```sql
SELECT * FROM data WHERE (julianday(`Discharge Date`) - julianday(`Date of Admission`)) >= 21 AND `Room Number` BETWEEN 400 AND 449
```

**42 rows**; first 3 (selected columns):

| Name | Discharge Date | Date of Admission | Room Number |
|---|---|---|---|
| Dylan Mcknight | 2019-07-13 | 2019-06-14 | 437 |
| Amanda Stein DVM | 2023-11-25 | 2023-10-28 | 404 |
| Linda Wheeler | 2020-06-04 | 2020-05-07 | 405 |

**Naive guess SQL**

```sql
SELECT * FROM data WHERE (julianday(`Discharge Date`) - julianday(`Date of Admission`)) > 14 AND `Room Number` >= 400
```

**145 rows**; first 3 (selected columns):

| Name | Discharge Date | Date of Admission | Room Number |
|---|---|---|---|
| Mrs. Brandy Flowers | 2021-08-02 | 2021-07-09 | 477 |
| Christina Williams | 2021-12-14 | 2021-11-29 | 444 |
| William Page | 2021-08-14 | 2021-07-29 | 492 |

### 2. `rg_f01` (retail): "list the names of our high-value customers"

**Required definitions**

- `rt_high_value_customer` **High-value customer**: A customer whose lifetime revenue (the sum of `Revenue` over all of their orders, returned or not) is greater than 1,750.

**Gold SQL**

```sql
SELECT `Customer Name` FROM data GROUP BY `Customer Name` HAVING SUM(`Revenue`) > 1750
```

**34 rows**; first 3 (selected columns):

| Customer Name |
|---|
| Aria Davis |
| Aria Garcia |
| Aria Patel |

**Naive guess SQL**

```sql
SELECT `Customer Name` FROM data GROUP BY `Customer Name` HAVING SUM(`Revenue`) > 1000
```

**135 rows**; first 3 (selected columns):

| Customer Name |
|---|
| Aria Chen |
| Aria Davis |
| Aria Garcia |

### 3. `hg_f04` (healthcare): "show the cost overruns among cancer patients"

**Required definitions**

- `hc_cost_overrun` **Cost overrun**: An admission billed at more than 1.5 times the average `Billing Amount` of all admissions with the same `Medical Condition`.

**Gold SQL**

```sql
SELECT * FROM data WHERE `Medical Condition` = 'Cancer' AND `Billing Amount` > 1.5 * (SELECT AVG(x.`Billing Amount`) FROM data x WHERE x.`Medical Condition` = data.`Medical Condition`)
```

**38 rows**; first 3 (selected columns):

| Name | Medical Condition | Billing Amount |
|---|---|---|
| Amy Roberts | Cancer | 40,325.07 |
| Rachael Davidson | Cancer | 41,295.40 |
| Cynthia Stanton | Cancer | 44,935.27 |

**Naive guess SQL**

```sql
SELECT * FROM data WHERE `Medical Condition` = 'Cancer' AND `Billing Amount` > (SELECT AVG(`Billing Amount`) FROM data)
```

**84 rows**; first 3 (selected columns):

| Name | Medical Condition | Billing Amount |
|---|---|---|
| Amy Roberts | Cancer | 40,325.07 |
| John Griffin | Cancer | 25,948.51 |
| Rachael Davidson | Cancer | 41,295.40 |

## Count

### 4. `rg_c01` (retail): "how many repeat customers do we have?"

**Required definitions**

- `rt_repeat_customer` **Repeat customer**: A customer with 3 or more orders. Customers with exactly 2 orders are not yet counted as repeat.

**Gold SQL**

```sql
SELECT COUNT(*) FROM (SELECT `Customer Name` FROM data GROUP BY `Customer Name` HAVING COUNT(*) >= 3)
```

**1 row**:

| COUNT(*) |
|---|
| 194 |

**Naive guess SQL**

```sql
SELECT COUNT(*) FROM (SELECT `Customer Name` FROM data GROUP BY `Customer Name` HAVING COUNT(*) >= 2)
```

**1 row**:

| COUNT(*) |
|---|
| 282 |

### 5. `hg_c03` (healthcare): "how many admissions came from government payers in fiscal year 2021?"

**Required definitions**

- `hc_government_payer` **Government payer**: `Insurance Provider` is Medicare or UnitedHealthcare. UnitedHealthcare counts because the hospital's managed-Medicaid contract is administered through it.
- `hc_fiscal_year` **Fiscal year**: The hospital fiscal year runs from July 1 to June 30 and is named after the calendar year in which it ends, based on `Date of Admission`. FY2022 = admissions from 2021-07-01 through 2022-06-30.

**Gold SQL**

```sql
SELECT COUNT(*) FROM data WHERE `Insurance Provider` IN ('Medicare', 'UnitedHealthcare') AND (CAST(strftime('%Y', `Date of Admission`) AS INTEGER) + (CAST(strftime('%m', `Date of Admission`) AS INTEGER) >= 7)) = 2021
```

**1 row**:

| COUNT(*) |
|---|
| 78 |

**Naive guess SQL**

```sql
SELECT COUNT(*) FROM data WHERE `Insurance Provider` = 'Medicare' AND strftime('%Y', `Date of Admission`) = '2021'
```

**1 row**:

| COUNT(*) |
|---|
| 47 |

### 6. `rg_c06` (retail): "how many bulk buyers are there?"

**Required definitions**

- `rt_bulk_buyer` **Bulk buyer**: A customer whose total `Quantity` across all orders is 25 or more. Different from a bulk order.

**Gold SQL**

```sql
SELECT COUNT(*) FROM (SELECT `Customer Name` FROM data GROUP BY `Customer Name` HAVING SUM(`Quantity`) >= 25)
```

**1 row**:

| COUNT(*) |
|---|
| 26 |

**Naive guess SQL**

```sql
SELECT COUNT(DISTINCT `Customer Name`) FROM data WHERE `Quantity` >= 6
```

**1 row**:

| COUNT(DISTINCT `Customer Name`) |
|---|
| 254 |

## Aggregate

### 7. `hg_a01` (healthcare): "what is the average daily rate for emergency admissions?"

**Required definitions**

- `hc_daily_rate` **Daily rate**: `Billing Amount` divided by the admission's billable days (discharge minus admission, plus one). "Average daily rate" means the mean of the per-admission daily rates, not total billing divided by total days.

**Gold SQL**

```sql
SELECT AVG(`Billing Amount` / ((julianday(`Discharge Date`) - julianday(`Date of Admission`)) + 1)) FROM data WHERE `Admission Type` = 'Emergency'
```

**1 row**:

| AVG(`Billing Amount` / ((julianday(`Discharge Date`) - julianday(`Date of Admission`)) + 1)) |
|---|
| 2,344.07 |

**Naive guess SQL**

```sql
SELECT AVG(`Billing Amount` / (julianday(`Discharge Date`) - julianday(`Date of Admission`))) FROM data WHERE `Admission Type` = 'Emergency'
```

**1 row**:

| AVG(`Billing Amount` / (julianday(`Discharge Date`) - julianday(`Date of Admission`))) |
|---|
| 3,002.00 |

### 8. `rg_a01` (retail): "what is the net revenue for each region?"

**Required definitions**

- `rt_net_revenue` **Net revenue**: `Revenue` from orders that were not returned (`Returned` = 'No'). Returned orders contribute zero.

**Gold SQL**

```sql
SELECT `Region`, SUM(`Revenue`) FROM data WHERE `Returned` = 'No' GROUP BY `Region`
```

**4 rows**:

| Region | SUM(`Revenue`) |
|---|---|
| East | 80,310.12 |
| North | 55,617.10 |
| South | 79,380.87 |
| West | 71,758.69 |

**Naive guess SQL**

```sql
SELECT `Region`, SUM(`Revenue`) FROM data GROUP BY `Region`
```

**4 rows**:

| Region | SUM(`Revenue`) |
|---|---|
| East | 91,285.99 |
| North | 64,417.04 |
| South | 86,975.13 |
| West | 79,619.17 |

### 9. `rg_a04` (retail): "what is the CSAT for each product category?"

**Required definitions**

- `rt_csat` **Customer satisfaction score (CSAT)**: The percentage of orders with `Rating` of 4 or 5: 100 × (orders rated ≥ 4) / (all orders).

**Gold SQL**

```sql
SELECT `Category`, 100.0 * AVG(`Rating` >= 4) FROM data GROUP BY `Category`
```

**5 rows**:

| Category | 100.0 * AVG(`Rating` >= 4) |
|---|---|
| Beauty | 75.00 |
| Clothing | 67.40 |
| Electronics | 64.71 |
| Home & Kitchen | 65.56 |
| Sports | 65.45 |

**Naive guess SQL**

```sql
SELECT `Category`, AVG(`Rating`) FROM data GROUP BY `Category`
```

**5 rows**:

| Category | AVG(`Rating`) |
|---|---|
| Beauty | 3.93 |
| Clothing | 3.77 |
| Electronics | 3.77 |
| Home & Kitchen | 3.82 |
| Sports | 3.70 |

## Compare

### 10. `hg_p01` (healthcare): "compare the average billing of premium and standard insurers"

**Required definitions**

- `hc_premium_insurer` **Premium insurer**: `Insurance Provider` is Aetna or Cigna. All other providers are standard insurers.

**Gold SQL**

```sql
SELECT AVG(`Billing Amount`) FROM data GROUP BY `Insurance Provider` IN ('Aetna', 'Cigna')
```

**2 rows**:

| AVG(`Billing Amount`) |
|---|
| 25,266.05 |
| 25,678.61 |

**Naive guess SQL**

```sql
SELECT AVG(`Billing Amount`) FROM data GROUP BY `Insurance Provider` != 'Medicare'
```

**2 rows**:

| AVG(`Billing Amount`) |
|---|
| 24,818.21 |
| 25,590.32 |

### 11. `rg_p01` (retail): "compare the return rate of online and in-store orders"

**Required definitions**

- `rt_return_rate` **Return rate**: Revenue-weighted: the `Revenue` of returned orders (`Returned` = 'Yes') divided by total `Revenue`, times 100. It is NOT the share of orders that were returned.

**Gold SQL**

```sql
SELECT `Channel`, 100.0 * SUM(CASE WHEN `Returned` = 'Yes' THEN `Revenue` ELSE 0 END) / SUM(`Revenue`) FROM data GROUP BY `Channel`
```

**2 rows**:

| Channel | 100.0 * SUM(CASE WHEN `Returned` = 'Yes' THEN `Revenue` ELSE 0 END) / SUM(`Revenue`) |
|---|---|
| In-Store | 10.76 |
| Online | 11.02 |

**Naive guess SQL**

```sql
SELECT `Channel`, 100.0 * AVG(`Returned` = 'Yes') FROM data GROUP BY `Channel`
```

**2 rows**:

| Channel | 100.0 * AVG(`Returned` = 'Yes') |
|---|---|
| In-Store | 9.73 |
| Online | 9.83 |

### 12. `hg_p04` (healthcare): "compare the number of admissions in the ICU and in the step-down unit"

**Required definitions**

- `hc_icu_room` **ICU room**: `Room Number` from 400 to 449 inclusive. These rooms are the intensive care unit.
- `hc_step_down_unit` **Step-down unit**: `Room Number` from 450 to 500 inclusive: the step-down (intermediate care) unit next to the ICU.

**Gold SQL**

```sql
SELECT COUNT(*) FROM data WHERE `Room Number` BETWEEN 400 AND 500 GROUP BY `Room Number` BETWEEN 400 AND 449
```

**2 rows**:

| COUNT(*) |
|---|
| 134 |
| 122 |

**Naive guess SQL**

```sql
SELECT COUNT(*) FROM data WHERE `Room Number` >= 300 GROUP BY `Room Number` >= 400
```

**2 rows**:

| COUNT(*) |
|---|
| 223 |
| 256 |

## Trend

### 13. `rg_t01` (retail): "revenue by fiscal quarter in fiscal year 2023"

**Required definitions**

- `rt_fiscal_quarter` **Fiscal quarter**: Quarters of the February–January fiscal year, by `Order Date` month: Q1 = February–April, Q2 = May–July, Q3 = August–October, Q4 = November–January.
- `rt_fiscal_year` **Fiscal year**: The fiscal year runs from February 1 to January 31 and is named after the calendar year in which it starts, based on `Order Date`. FY2023 = orders from 2023-02-01 through 2024-01-31.

**Gold SQL**

```sql
SELECT SUM(`Revenue`) FROM data WHERE (CAST(strftime('%Y', `Order Date`) AS INTEGER) - (CAST(strftime('%m', `Order Date`) AS INTEGER) = 1)) = 2023 GROUP BY (((CAST(strftime('%m', `Order Date`) AS INTEGER) + 10) % 12) / 3 + 1) ORDER BY (((CAST(strftime('%m', `Order Date`) AS INTEGER) + 10) % 12) / 3 + 1)
```

**4 rows**:

| SUM(`Revenue`) |
|---|
| 32,935.90 |
| 36,057.17 |
| 35,201.74 |
| 49,484.49 |

**Naive guess SQL**

```sql
SELECT SUM(`Revenue`) FROM data WHERE CAST(strftime('%Y', `Order Date`) AS INTEGER) = 2023 GROUP BY (CAST(strftime('%m', `Order Date`) AS INTEGER) - 1) / 3 ORDER BY (CAST(strftime('%m', `Order Date`) AS INTEGER) - 1) / 3
```

**4 rows**:

| SUM(`Revenue`) |
|---|
| 34,758.54 |
| 34,725.59 |
| 30,651.73 |
| 53,564.91 |

### 14. `hg_t01` (healthcare): "number of admissions per fiscal year"

**Required definitions**

- `hc_fiscal_year` **Fiscal year**: The hospital fiscal year runs from July 1 to June 30 and is named after the calendar year in which it ends, based on `Date of Admission`. FY2022 = admissions from 2021-07-01 through 2022-06-30.

**Gold SQL**

```sql
SELECT COUNT(*) FROM data GROUP BY (CAST(strftime('%Y', `Date of Admission`) AS INTEGER) + (CAST(strftime('%m', `Date of Admission`) AS INTEGER) >= 7)) ORDER BY (CAST(strftime('%Y', `Date of Admission`) AS INTEGER) + (CAST(strftime('%m', `Date of Admission`) AS INTEGER) >= 7))
```

**6 rows**:

| COUNT(*) |
|---|
| 130 |
| 202 |
| 199 |
| 211 |
| 184 |
| 73 |

**Naive guess SQL**

```sql
SELECT COUNT(*) FROM data GROUP BY CAST(strftime('%Y', `Date of Admission`) AS INTEGER) ORDER BY CAST(strftime('%Y', `Date of Admission`) AS INTEGER)
```

**6 rows**:

| COUNT(*) |
|---|
| 34 |
| 195 |
| 196 |
| 221 |
| 195 |
| 158 |

### 15. `rg_t04` (retail): "holiday-season revenue by year"

**Required definitions**

- `rt_holiday_season` **Holiday season**: Orders with an `Order Date` from November 15 through December 31 (inclusive) of any year.

**Gold SQL**

```sql
SELECT strftime('%Y', `Order Date`), SUM(`Revenue`) FROM data WHERE strftime('%m-%d', `Order Date`) >= '11-15' GROUP BY 1 ORDER BY 1
```

**2 rows**:

| strftime('%Y', `Order Date`) | SUM(`Revenue`) |
|---|---|
| 2023 | 29,625.74 |
| 2024 | 34,424.59 |

**Naive guess SQL**

```sql
SELECT strftime('%Y', `Order Date`), SUM(`Revenue`) FROM data WHERE strftime('%m', `Order Date`) IN ('11', '12') GROUP BY 1 ORDER BY 1
```

**2 rows**:

| strftime('%Y', `Order Date`) | SUM(`Revenue`) |
|---|---|
| 2023 | 38,104.65 |
| 2024 | 44,623.63 |
