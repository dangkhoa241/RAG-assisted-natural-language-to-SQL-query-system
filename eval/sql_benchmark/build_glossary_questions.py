"""Build eval/sql_benchmark/glossary_questions.jsonl: questions that need business definitions.

Every question uses a term from docs/glossary/<dataset>.md without restating its definition, so it can
only be answered correctly by knowing the glossary. Each item stores:
  required_chunks - glossary chunk IDs needed to answer it (1-2)
  gold_sql        - the correct query
  naive_sql       - a plausible-but-wrong guess at the definition (evidence the glossary matters)

Checks (the script fails loudly if any doesn't hold):
  - every gold and naive query runs; gold is non-empty (filters return at most 200 rows, counts are > 0)
  - the naive result DIFFERS from the gold result under the benchmark's own comparison (results_match)
  - every required chunk ID exists in the glossary
  - 30 questions per dataset, 6 per intent

Result-shape conventions (results_match ignores column names and allows extra predicted columns):
  - compare/trend over derived groups (e.g. premium vs standard insurers, fiscal years) return only the
    metric, so any label the model invents still matches; fiscal trends are `ordered` by period.
  - groups over real columns (Region, Channel, calendar months) keep the real label.

Usage (from the repo root):  python eval/sql_benchmark/build_glossary_questions.py
"""
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "eval"))
warnings.filterwarnings("ignore", message="Could not infer format")

from data_context import load_dataset  # noqa: E402
from doc_retrieval import load_glossary  # noqa: E402
from sql_metrics import results_match  # noqa: E402
from sql_safety import run_safe_query  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "glossary_questions.jsonl"
DATASETS = {"healthcare": ROOT_DIR / "data" / "healthcare_dataset.csv", "retail": ROOT_DIR / "data" / "retail_sales.csv"}
INTENTS = ["filter", "count", "aggregate", "compare", "trend"]

# --- Healthcare expressions ----------------------------------------------------
LOS = "(julianday(`Discharge Date`) - julianday(`Date of Admission`))"
ADM_YEAR = "CAST(strftime('%Y', `Date of Admission`) AS INTEGER)"
ADM_MONTH = "CAST(strftime('%m', `Date of Admission`) AS INTEGER)"
HC_FY = f"({ADM_YEAR} + ({ADM_MONTH} >= 7))"                        # July-June, named by end year
HC_FQ = f"((({ADM_MONTH} + 5) % 12) / 3 + 1)"                       # Jul-Sep = 1 ... Apr-Jun = 4
ICU = "`Room Number` BETWEEN 400 AND 449"
STEP_DOWN = "`Room Number` BETWEEN 450 AND 500"
PREMIUM_INS = "`Insurance Provider` IN ('Aetna', 'Cigna')"
GOV_PAYER = "`Insurance Provider` IN ('Medicare', 'UnitedHealthcare')"
CHRONIC = "`Medical Condition` IN ('Diabetes', 'Hypertension', 'Obesity', 'Arthritis')"
UNPLANNED = "`Admission Type` IN ('Emergency', 'Urgent')"
FLAGGED = "`Test Results` IN ('Abnormal', 'Inconclusive')"
FORMULARY = "`Medication` IN ('Lipitor', 'Penicillin', 'Ibuprofen')"
RARE_BLOOD = "`Blood Type` IN ('AB-', 'B-', 'A-')"
HIGH_RISK = "`Age` >= 60 AND `Medical Condition` IN ('Cancer', 'Diabetes')"
FLU = f"{ADM_MONTH} IN (11, 12, 1, 2)"
OVERRUN = ("`Billing Amount` > 1.5 * (SELECT AVG(x.`Billing Amount`) FROM data x "
           "WHERE x.`Medical Condition` = data.`Medical Condition`)")

# --- Retail expressions --------------------------------------------------------
ORD_YEAR = "CAST(strftime('%Y', `Order Date`) AS INTEGER)"
ORD_MONTH = "CAST(strftime('%m', `Order Date`) AS INTEGER)"
RT_FY = f"({ORD_YEAR} - ({ORD_MONTH} = 1))"                          # Feb-Jan, named by start year
RT_FQ = f"((({ORD_MONTH} + 10) % 12) / 3 + 1)"                      # Feb-Apr = 1 ... Nov-Jan = 4
GROSS = "(`Unit Price` * `Quantity`)"
HOLIDAY = "strftime('%m-%d', `Order Date`) >= '11-15'"
HARD = "`Category` IN ('Electronics', 'Home & Kitchen', 'Sports')"
CORE = "`Region` IN ('East', 'South')"
B2B = "`Customer Segment` IN ('Corporate', 'Small Business')"
CASH_EQ = "`Payment Method` IN ('Cash', 'Debit Card')"
CSAT = "100.0 * AVG(`Rating` >= 4)"
RETURN_RATE = "100.0 * SUM(CASE WHEN `Returned` = 'Yes' THEN `Revenue` ELSE 0 END) / SUM(`Revenue`)"
NAIVE_RETURN_RATE = "100.0 * AVG(`Returned` = 'Yes')"

Q = []


def q(qid, dataset, intent, question, required, gold_sql, naive_sql, ordered=False):
    Q.append({"id": qid, "dataset": dataset, "intent": intent, "ordered": ordered, "question": question,
              "required_chunks": required, "gold_sql": " ".join(gold_sql.split()),
              "naive_sql": " ".join(naive_sql.split())})


H, R = "healthcare", "retail"

# ============================== Healthcare =====================================
# filter
q("hg_f01", H, "filter", "show long-stay admissions in the ICU", ["hc_long_stay", "hc_icu_room"],
  f"SELECT * FROM data WHERE {LOS} >= 21 AND {ICU}",
  f"SELECT * FROM data WHERE {LOS} > 14 AND `Room Number` >= 400")
q("hg_f02", H, "filter", "list high-cost admissions of senior patients", ["hc_high_cost", "hc_senior_patient"],
  "SELECT * FROM data WHERE `Billing Amount` > 42500 AND `Age` >= 67",
  "SELECT * FROM data WHERE `Billing Amount` > 40000 AND `Age` >= 65")
q("hg_f03", H, "filter", "show patients with premium insurers and flagged results who were admitted in 2023",
  ["hc_premium_insurer", "hc_flagged_result"],
  f"SELECT * FROM data WHERE {PREMIUM_INS} AND {FLAGGED} AND strftime('%Y', `Date of Admission`) = '2023'",
  "SELECT * FROM data WHERE `Insurance Provider` != 'Medicare' AND `Test Results` = 'Abnormal' "
  "AND strftime('%Y', `Date of Admission`) = '2023'")
q("hg_f04", H, "filter", "show the cost overruns among cancer patients", ["hc_cost_overrun"],
  f"SELECT * FROM data WHERE `Medical Condition` = 'Cancer' AND {OVERRUN}",
  "SELECT * FROM data WHERE `Medical Condition` = 'Cancer' AND `Billing Amount` > (SELECT AVG(`Billing Amount`) FROM data)")
q("hg_f05", H, "filter", "list short stays in the step-down unit", ["hc_short_stay", "hc_step_down_unit"],
  f"SELECT * FROM data WHERE {LOS} <= 3 AND {STEP_DOWN}",
  f"SELECT * FROM data WHERE {LOS} <= 2 AND `Room Number` >= 300 AND `Room Number` < 400")
q("hg_f06", H, "filter", "show flu-season admissions of high-risk patients in 2022",
  ["hc_flu_season", "hc_high_risk_patient"],
  f"SELECT * FROM data WHERE {FLU} AND {HIGH_RISK} AND strftime('%Y', `Date of Admission`) = '2022'",
  f"SELECT * FROM data WHERE {ADM_MONTH} IN (12, 1, 2) AND `Age` >= 65 "
  "AND `Medical Condition` IN ('Cancer', 'Diabetes', 'Hypertension') AND strftime('%Y', `Date of Admission`) = '2022'")
# count
q("hg_c01", H, "count", "how many unplanned admissions were for chronic conditions?",
  ["hc_unplanned_admission", "hc_chronic_condition"],
  f"SELECT COUNT(*) FROM data WHERE {UNPLANNED} AND {CHRONIC}",
  "SELECT COUNT(*) FROM data WHERE `Admission Type` = 'Emergency' "
  "AND `Medical Condition` IN ('Diabetes', 'Hypertension', 'Obesity', 'Arthritis', 'Asthma')")
q("hg_c02", H, "count", "how many admissions involved a non-formulary drug?", ["hc_formulary_drug"],
  f"SELECT COUNT(*) FROM data WHERE NOT ({FORMULARY})",
  "SELECT COUNT(*) FROM data WHERE `Medication` IN ('Aspirin', 'Ibuprofen', 'Paracetamol')")
q("hg_c03", H, "count", "how many admissions came from government payers in fiscal year 2021?",
  ["hc_government_payer", "hc_fiscal_year"],
  f"SELECT COUNT(*) FROM data WHERE {GOV_PAYER} AND {HC_FY} = 2021",
  "SELECT COUNT(*) FROM data WHERE `Insurance Provider` = 'Medicare' AND strftime('%Y', `Date of Admission`) = '2021'")
q("hg_c04", H, "count", "count the ICU admissions of patients with a rare blood type", ["hc_rare_blood_type", "hc_icu_room"],
  f"SELECT COUNT(*) FROM data WHERE {RARE_BLOOD} AND {ICU}",
  "SELECT COUNT(*) FROM data WHERE `Blood Type` = 'AB-' AND `Room Number` >= 400")
q("hg_c05", H, "count", "how many admissions of young adults had flagged test results?", ["hc_young_adult", "hc_flagged_result"],
  f"SELECT COUNT(*) FROM data WHERE `Age` BETWEEN 18 AND 34 AND {FLAGGED}",
  "SELECT COUNT(*) FROM data WHERE `Age` BETWEEN 18 AND 25 AND `Test Results` = 'Abnormal'")
q("hg_c06", H, "count", "how many flagged claims are there?", ["hc_flagged_claim"],
  "SELECT COUNT(*) FROM data WHERE `Billing Amount` < 2000",
  "SELECT COUNT(*) FROM data WHERE `Billing Amount` > 45000")
# aggregate
q("hg_a01", H, "aggregate", "what is the average daily rate for emergency admissions?", ["hc_daily_rate"],
  f"SELECT AVG(`Billing Amount` / ({LOS} + 1)) FROM data WHERE `Admission Type` = 'Emergency'",
  f"SELECT AVG(`Billing Amount` / {LOS}) FROM data WHERE `Admission Type` = 'Emergency'")
q("hg_a02", H, "aggregate", "average billable days by medical condition", ["hc_billable_days"],
  f"SELECT `Medical Condition`, AVG({LOS} + 1) FROM data GROUP BY `Medical Condition`",
  f"SELECT `Medical Condition`, AVG({LOS}) FROM data GROUP BY `Medical Condition`")
q("hg_a03", H, "aggregate", "total billing of high-cost admissions for each insurance provider", ["hc_high_cost"],
  "SELECT `Insurance Provider`, SUM(`Billing Amount`) FROM data WHERE `Billing Amount` > 42500 GROUP BY `Insurance Provider`",
  "SELECT `Insurance Provider`, SUM(`Billing Amount`) FROM data WHERE `Billing Amount` > 40000 GROUP BY `Insurance Provider`")
q("hg_a04", H, "aggregate", "average billing amount of senior care tier patients by gender", ["hc_senior_care_tier"],
  "SELECT `Gender`, AVG(`Billing Amount`) FROM data WHERE `Age` >= 80 GROUP BY `Gender`",
  "SELECT `Gender`, AVG(`Billing Amount`) FROM data WHERE `Age` >= 65 GROUP BY `Gender`")
q("hg_a05", H, "aggregate", "average billing per medical condition for patients with premium insurers",
  ["hc_premium_insurer"],
  f"SELECT `Medical Condition`, AVG(`Billing Amount`) FROM data WHERE {PREMIUM_INS} GROUP BY `Medical Condition`",
  "SELECT `Medical Condition`, AVG(`Billing Amount`) FROM data WHERE `Insurance Provider` != 'Medicare' GROUP BY `Medical Condition`")
q("hg_a06", H, "aggregate", "average age of high-risk patients for each admission type", ["hc_high_risk_patient"],
  f"SELECT `Admission Type`, AVG(`Age`) FROM data WHERE {HIGH_RISK} GROUP BY `Admission Type`",
  "SELECT `Admission Type`, AVG(`Age`) FROM data WHERE `Age` >= 65 "
  "AND `Medical Condition` IN ('Cancer', 'Diabetes', 'Hypertension') GROUP BY `Admission Type`")
# compare
q("hg_p01", H, "compare", "compare the average billing of premium and standard insurers", ["hc_premium_insurer"],
  f"SELECT AVG(`Billing Amount`) FROM data GROUP BY {PREMIUM_INS}",
  "SELECT AVG(`Billing Amount`) FROM data GROUP BY `Insurance Provider` != 'Medicare'")
q("hg_p02", H, "compare", "do long-stay admissions or short stays cost more on average?",
  ["hc_long_stay", "hc_short_stay"],
  f"SELECT AVG(`Billing Amount`) FROM data WHERE {LOS} >= 21 OR {LOS} <= 3 GROUP BY {LOS} >= 21",
  f"SELECT AVG(`Billing Amount`) FROM data WHERE {LOS} > 14 OR {LOS} <= 2 GROUP BY {LOS} > 14")
q("hg_p03", H, "compare", "formulary vs non-formulary drugs: which has more flagged results?",
  ["hc_formulary_drug", "hc_flagged_result"],
  f"SELECT COUNT(*) FROM data WHERE {FLAGGED} GROUP BY {FORMULARY}",
  "SELECT COUNT(*) FROM data WHERE `Test Results` = 'Abnormal' GROUP BY `Medication` IN ('Lipitor', 'Penicillin')")
q("hg_p04", H, "compare", "compare the number of admissions in the ICU and in the step-down unit",
  ["hc_icu_room", "hc_step_down_unit"],
  f"SELECT COUNT(*) FROM data WHERE `Room Number` BETWEEN 400 AND 500 GROUP BY {ICU}",
  "SELECT COUNT(*) FROM data WHERE `Room Number` >= 300 GROUP BY `Room Number` >= 400")
q("hg_p05", H, "compare", "senior patients vs young adults: who has the higher average bill?",
  ["hc_senior_patient", "hc_young_adult"],
  "SELECT AVG(`Billing Amount`) FROM data WHERE `Age` >= 67 OR `Age` BETWEEN 18 AND 34 GROUP BY `Age` >= 67",
  "SELECT AVG(`Billing Amount`) FROM data WHERE `Age` >= 65 OR `Age` BETWEEN 18 AND 25 GROUP BY `Age` >= 65")
q("hg_p06", H, "compare", "compare the number of flagged results for unplanned and planned admissions",
  ["hc_flagged_result", "hc_unplanned_admission"],
  f"SELECT COUNT(*) FROM data WHERE {FLAGGED} GROUP BY {UNPLANNED}",
  "SELECT COUNT(*) FROM data WHERE `Test Results` = 'Abnormal' GROUP BY `Admission Type` = 'Emergency'")
# trend
q("hg_t01", H, "trend", "number of admissions per fiscal year", ["hc_fiscal_year"],
  f"SELECT COUNT(*) FROM data GROUP BY {HC_FY} ORDER BY {HC_FY}",
  f"SELECT COUNT(*) FROM data GROUP BY {ADM_YEAR} ORDER BY {ADM_YEAR}", ordered=True)
q("hg_t02", H, "trend", "admissions by fiscal quarter in FY2022", ["hc_fiscal_quarter", "hc_fiscal_year"],
  f"SELECT COUNT(*) FROM data WHERE {HC_FY} = 2022 GROUP BY {HC_FQ} ORDER BY {HC_FQ}",
  f"SELECT COUNT(*) FROM data WHERE {ADM_YEAR} = 2022 GROUP BY ({ADM_MONTH} - 1) / 3 ORDER BY ({ADM_MONTH} - 1) / 3",
  ordered=True)
q("hg_t03", H, "trend", "monthly number of unplanned admissions in 2022", ["hc_unplanned_admission"],
  f"SELECT strftime('%Y-%m', `Date of Admission`), COUNT(*) FROM data WHERE {UNPLANNED} "
  "AND strftime('%Y', `Date of Admission`) = '2022' GROUP BY 1 ORDER BY 1",
  "SELECT strftime('%Y-%m', `Date of Admission`), COUNT(*) FROM data WHERE `Admission Type` = 'Emergency' "
  "AND strftime('%Y', `Date of Admission`) = '2022' GROUP BY 1 ORDER BY 1")
q("hg_t04", H, "trend", "average daily rate by year", ["hc_daily_rate"],
  f"SELECT strftime('%Y', `Date of Admission`), AVG(`Billing Amount` / ({LOS} + 1)) FROM data GROUP BY 1 ORDER BY 1",
  f"SELECT strftime('%Y', `Date of Admission`), AVG(`Billing Amount` / {LOS}) FROM data GROUP BY 1 ORDER BY 1")
q("hg_t05", H, "trend", "number of long-stay admissions per year", ["hc_long_stay"],
  f"SELECT strftime('%Y', `Date of Admission`), COUNT(*) FROM data WHERE {LOS} >= 21 GROUP BY 1 ORDER BY 1",
  f"SELECT strftime('%Y', `Date of Admission`), COUNT(*) FROM data WHERE {LOS} > 14 GROUP BY 1 ORDER BY 1")
q("hg_t06", H, "trend", "how has the number of cost overruns changed by fiscal year?",
  ["hc_cost_overrun", "hc_fiscal_year"],
  f"SELECT COUNT(*) FROM data WHERE {OVERRUN} GROUP BY {HC_FY} ORDER BY {HC_FY}",
  f"SELECT COUNT(*) FROM data WHERE `Billing Amount` > 1.5 * (SELECT AVG(`Billing Amount`) FROM data) "
  f"GROUP BY {ADM_YEAR} ORDER BY {ADM_YEAR}", ordered=True)

# ================================ Retail =======================================
# filter
q("rg_f01", R, "filter", "list the names of our high-value customers", ["rt_high_value_customer"],
  "SELECT `Customer Name` FROM data GROUP BY `Customer Name` HAVING SUM(`Revenue`) > 1750",
  "SELECT `Customer Name` FROM data GROUP BY `Customer Name` HAVING SUM(`Revenue`) > 1000")
q("rg_f02", R, "filter", "show deep-discount bulk orders in the West", ["rt_deep_discount", "rt_bulk_order"],
  "SELECT * FROM data WHERE `Discount` >= 0.15 AND `Quantity` >= 6 AND `Region` = 'West'",
  "SELECT * FROM data WHERE `Discount` >= 0.2 AND `Quantity` >= 5 AND `Region` = 'West'")
q("rg_f03", R, "filter", "show detractor orders paid with a cash-equivalent method", ["rt_detractor", "rt_cash_equivalent"],
  f"SELECT * FROM data WHERE `Rating` <= 2 AND {CASH_EQ}",
  "SELECT * FROM data WHERE `Rating` <= 3 AND `Payment Method` = 'Cash'")
q("rg_f04", R, "filter", "list premium-product orders placed by mature shoppers",
  ["rt_premium_product_order", "rt_mature_shopper"],
  "SELECT * FROM data WHERE `Unit Price` >= 120 AND `Customer Age` >= 55",
  "SELECT * FROM data WHERE `Unit Price` >= 100 AND `Customer Age` >= 65")
q("rg_f05", R, "filter", "show hard-goods orders from the 2024 holiday season", ["rt_hard_goods", "rt_holiday_season"],
  f"SELECT * FROM data WHERE {HARD} AND {HOLIDAY} AND strftime('%Y', `Order Date`) = '2024'",
  "SELECT * FROM data WHERE `Category` IN ('Electronics', 'Home & Kitchen') "
  "AND strftime('%m', `Order Date`) IN ('11', '12') AND strftime('%Y', `Order Date`) = '2024'")
q("rg_f06", R, "filter", "list the names of omnichannel customers who are also repeat customers",
  ["rt_omnichannel_customer", "rt_repeat_customer"],
  "SELECT `Customer Name` FROM data GROUP BY `Customer Name` "
  "HAVING SUM(`Channel` = 'Online') > 0 AND SUM(`Channel` = 'In-Store') > 0 AND COUNT(*) >= 3",
  "SELECT `Customer Name` FROM data GROUP BY `Customer Name` "
  "HAVING SUM(`Channel` = 'Online') > 0 AND SUM(`Channel` = 'In-Store') > 0 AND COUNT(*) >= 2")
# count
q("rg_c01", R, "count", "how many repeat customers do we have?", ["rt_repeat_customer"],
  "SELECT COUNT(*) FROM (SELECT `Customer Name` FROM data GROUP BY `Customer Name` HAVING COUNT(*) >= 3)",
  "SELECT COUNT(*) FROM (SELECT `Customer Name` FROM data GROUP BY `Customer Name` HAVING COUNT(*) >= 2)")
q("rg_c02", R, "count", "how many B2B orders were placed in core markets?", ["rt_b2b_order", "rt_core_market"],
  f"SELECT COUNT(*) FROM data WHERE {B2B} AND {CORE}",
  "SELECT COUNT(*) FROM data WHERE `Customer Segment` = 'Corporate' AND `Region` IN ('East', 'West')")
q("rg_c03", R, "count", "how many orders did we get in fiscal year 2023?", ["rt_fiscal_year"],
  f"SELECT COUNT(*) FROM data WHERE {RT_FY} = 2023",
  "SELECT COUNT(*) FROM data WHERE strftime('%Y', `Order Date`) = '2023'")
q("rg_c04", R, "count", "how many returning customers are there?", ["rt_returning_customer"],
  "SELECT COUNT(*) FROM (SELECT `Customer Name` FROM data GROUP BY `Customer Name` "
  "HAVING MIN(strftime('%Y', `Order Date`)) = '2023' AND MAX(strftime('%Y', `Order Date`)) = '2024')",
  "SELECT COUNT(DISTINCT `Customer Name`) FROM data WHERE `Returned` = 'Yes'")
q("rg_c05", R, "count", "how many high-value orders came through the online channel?", ["rt_high_value_order"],
  "SELECT COUNT(*) FROM data WHERE `Revenue` >= 1000 AND `Channel` = 'Online'",
  "SELECT COUNT(*) FROM data WHERE `Revenue` > 800 AND `Channel` = 'Online'")
q("rg_c06", R, "count", "how many bulk buyers are there?", ["rt_bulk_buyer"],
  "SELECT COUNT(*) FROM (SELECT `Customer Name` FROM data GROUP BY `Customer Name` HAVING SUM(`Quantity`) >= 25)",
  "SELECT COUNT(DISTINCT `Customer Name`) FROM data WHERE `Quantity` >= 6")
# aggregate
q("rg_a01", R, "aggregate", "what is the net revenue for each region?", ["rt_net_revenue"],
  "SELECT `Region`, SUM(`Revenue`) FROM data WHERE `Returned` = 'No' GROUP BY `Region`",
  "SELECT `Region`, SUM(`Revenue`) FROM data GROUP BY `Region`")
q("rg_a02", R, "aggregate", "average order value per channel", ["rt_average_order_value"],
  "SELECT `Channel`, AVG(`Revenue`) FROM data WHERE `Returned` = 'No' GROUP BY `Channel`",
  "SELECT `Channel`, AVG(`Revenue`) FROM data GROUP BY `Channel`")
q("rg_a03", R, "aggregate", "total markdown by category", ["rt_markdown"],
  f"SELECT `Category`, SUM({GROSS} - `Revenue`) FROM data GROUP BY `Category`",
  "SELECT `Category`, SUM(`Discount`) FROM data GROUP BY `Category`")
q("rg_a04", R, "aggregate", "what is the CSAT for each product category?", ["rt_csat"],
  f"SELECT `Category`, {CSAT} FROM data GROUP BY `Category`",
  "SELECT `Category`, AVG(`Rating`) FROM data GROUP BY `Category`")
q("rg_a05", R, "aggregate", "total revenue from premium customers", ["rt_premium_customer"],
  "SELECT SUM(`Revenue`) FROM data WHERE `Customer Name` IN "
  "(SELECT `Customer Name` FROM data GROUP BY `Customer Name` HAVING AVG(`Unit Price`) >= 100)",
  "SELECT SUM(`Revenue`) FROM data WHERE `Customer Segment` = 'Corporate'")
q("rg_a06", R, "aggregate", "net sales by payment method", ["rt_net_sales"],
  f"SELECT `Payment Method`, SUM({GROSS}) FROM data WHERE `Returned` = 'No' GROUP BY `Payment Method`",
  "SELECT `Payment Method`, SUM(`Revenue`) FROM data WHERE `Returned` = 'No' GROUP BY `Payment Method`")
# compare
q("rg_p01", R, "compare", "compare the return rate of online and in-store orders", ["rt_return_rate"],
  f"SELECT `Channel`, {RETURN_RATE} FROM data GROUP BY `Channel`",
  f"SELECT `Channel`, {NAIVE_RETURN_RATE} FROM data GROUP BY `Channel`")
q("rg_p02", R, "compare", "core markets vs expansion markets: total gross sales", ["rt_core_market", "rt_gross_sales"],
  f"SELECT SUM({GROSS}) FROM data GROUP BY {CORE}",
  "SELECT SUM(`Revenue`) FROM data GROUP BY `Region` IN ('East', 'West')")
q("rg_p03", R, "compare", "hard goods vs soft goods: average revenue per order", ["rt_hard_goods"],
  f"SELECT AVG(`Revenue`) FROM data GROUP BY {HARD}",
  "SELECT AVG(`Revenue`) FROM data GROUP BY `Category` IN ('Electronics', 'Home & Kitchen')")
q("rg_p04", R, "compare", "compare CSAT between B2B and consumer orders", ["rt_csat", "rt_b2b_order"],
  f"SELECT {CSAT} FROM data GROUP BY {B2B}",
  "SELECT AVG(`Rating`) FROM data GROUP BY `Customer Segment` = 'Corporate'")
q("rg_p05", R, "compare", "do mature shoppers or younger shoppers have a higher average order value?",
  ["rt_mature_shopper", "rt_average_order_value"],
  "SELECT AVG(`Revenue`) FROM data WHERE `Returned` = 'No' GROUP BY `Customer Age` >= 55",
  "SELECT AVG(`Revenue`) FROM data GROUP BY `Customer Age` >= 65")
q("rg_p06", R, "compare", "compare the number of detractors in the North and the West", ["rt_detractor"],
  "SELECT `Region`, COUNT(*) FROM data WHERE `Rating` <= 2 AND `Region` IN ('North', 'West') GROUP BY `Region`",
  "SELECT `Region`, COUNT(*) FROM data WHERE `Rating` <= 3 AND `Region` IN ('North', 'West') GROUP BY `Region`")
# trend
q("rg_t01", R, "trend", "revenue by fiscal quarter in fiscal year 2023", ["rt_fiscal_quarter", "rt_fiscal_year"],
  f"SELECT SUM(`Revenue`) FROM data WHERE {RT_FY} = 2023 GROUP BY {RT_FQ} ORDER BY {RT_FQ}",
  f"SELECT SUM(`Revenue`) FROM data WHERE {ORD_YEAR} = 2023 GROUP BY ({ORD_MONTH} - 1) / 3 ORDER BY ({ORD_MONTH} - 1) / 3",
  ordered=True)
q("rg_t02", R, "trend", "net revenue per fiscal year", ["rt_fiscal_year", "rt_net_revenue"],
  f"SELECT SUM(`Revenue`) FROM data WHERE `Returned` = 'No' GROUP BY {RT_FY} ORDER BY {RT_FY}",
  f"SELECT SUM(`Revenue`) FROM data GROUP BY {ORD_YEAR} ORDER BY {ORD_YEAR}", ordered=True)
q("rg_t03", R, "trend", "monthly number of deep-discount orders in 2024", ["rt_deep_discount"],
  "SELECT strftime('%Y-%m', `Order Date`), COUNT(*) FROM data WHERE `Discount` >= 0.15 "
  "AND strftime('%Y', `Order Date`) = '2024' GROUP BY 1 ORDER BY 1",
  "SELECT strftime('%Y-%m', `Order Date`), COUNT(*) FROM data WHERE `Discount` >= 0.2 "
  "AND strftime('%Y', `Order Date`) = '2024' GROUP BY 1 ORDER BY 1")
q("rg_t04", R, "trend", "holiday-season revenue by year", ["rt_holiday_season"],
  f"SELECT strftime('%Y', `Order Date`), SUM(`Revenue`) FROM data WHERE {HOLIDAY} GROUP BY 1 ORDER BY 1",
  "SELECT strftime('%Y', `Order Date`), SUM(`Revenue`) FROM data "
  "WHERE strftime('%m', `Order Date`) IN ('11', '12') GROUP BY 1 ORDER BY 1")
q("rg_t05", R, "trend", "how did the return rate change month by month in 2023?", ["rt_return_rate"],
  f"SELECT strftime('%Y-%m', `Order Date`), {RETURN_RATE} FROM data "
  "WHERE strftime('%Y', `Order Date`) = '2023' GROUP BY 1 ORDER BY 1",
  f"SELECT strftime('%Y-%m', `Order Date`), {NAIVE_RETURN_RATE} FROM data "
  "WHERE strftime('%Y', `Order Date`) = '2023' GROUP BY 1 ORDER BY 1")
q("rg_t06", R, "trend", "number of premium-product orders per month in 2023", ["rt_premium_product_order"],
  "SELECT strftime('%Y-%m', `Order Date`), COUNT(*) FROM data WHERE `Unit Price` >= 120 "
  "AND strftime('%Y', `Order Date`) = '2023' GROUP BY 1 ORDER BY 1",
  "SELECT strftime('%Y-%m', `Order Date`), COUNT(*) FROM data WHERE `Unit Price` >= 100 "
  "AND strftime('%Y', `Order Date`) = '2023' GROUP BY 1 ORDER BY 1")


def main():
    datasets = {name: load_dataset(path) for name, path in DATASETS.items()}
    chunk_ids = {name: {c["id"] for c in load_glossary(name)} for name in DATASETS}
    problems = []

    ids = [x["id"] for x in Q]
    if len(ids) != len(set(ids)):
        problems.append("duplicate question IDs")
    counts = Counter((x["dataset"], x["intent"]) for x in Q)
    for name in DATASETS:
        for intent in INTENTS:
            if counts[(name, intent)] != 6:
                problems.append(f"{name}/{intent}: {counts[(name, intent)]} questions (expected 6)")

    for x in Q:
        conn = datasets[x["dataset"]].conn
        if not 1 <= len(x["required_chunks"]) <= 2:
            problems.append(f"{x['id']}: needs {len(x['required_chunks'])} chunks")
        for cid in x["required_chunks"]:
            if cid not in chunk_ids[x["dataset"]]:
                problems.append(f"{x['id']}: {cid} is not in the {x['dataset']} glossary")
        try:
            gold = run_safe_query(x["gold_sql"], conn)
            naive = run_safe_query(x["naive_sql"], conn)
        except Exception as e:
            problems.append(f"{x['id']}: {type(e).__name__}: {e}")
            continue
        if gold.empty:
            problems.append(f"{x['id']}: empty gold result")
        if x["intent"] == "filter" and len(gold) > 200:
            problems.append(f"{x['id']}: {len(gold)} rows (> 200)")
        if x["intent"] == "count" and gold.iloc[0, 0] == 0:
            problems.append(f"{x['id']}: count is 0")
        if results_match(gold, naive, x["ordered"]):
            problems.append(f"{x['id']}: naive guess gives the same result as gold")
        x["gold_rows"] = len(gold)

    if problems:
        sys.exit("Glossary benchmark problems:\n  " + "\n  ".join(problems))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for x in Q:
            f.write(json.dumps({k: v for k, v in x.items() if k != "gold_rows"}) + "\n")
    print(f"Wrote {len(Q)} questions to {OUT_PATH}")
    for x in Q:
        print(f"  {x['id']}: {x['gold_rows']} gold rows")


if __name__ == "__main__":
    main()
