"""Build eval/sql_benchmark/saas_questions.jsonl, the Stage 3C held-out set, and REVIEW_SAAS.md.

60 questions on data/saas_subscriptions.csv, written after the SaaS glossary (docs/glossary/saas.md) was
fixed, and never used for tuning:
  set="glossary" (40, 8 per intent) - each uses a glossary term without restating it. Stores
      required_chunks, gold_sql and naive_sql (a plausible everyday reading of the term).
  set="plain" (20, 4 per intent)    - need no definition. Several sit next to a glossary term ("average
      MRR", "number of cancellations") to test whether unneeded definitions get applied.

Checks (the script fails loudly if any doesn't hold):
  - every gold and naive query runs; gold is non-empty (filters return 1-200 rows, counts are > 0)
  - every naive result DIFFERS from the gold result under the benchmark's comparison (results_match)
  - every required chunk ID exists in the SaaS glossary; 8 glossary + 4 plain questions per intent

Result-shape conventions are the same as the dev glossary set: compare/trend questions over derived groups
(e.g. high-touch vs self-serve, fiscal years) return only the metric, so any label the model picks still
matches; fiscal-year trends are compared in period order. Groups over real columns keep their labels.

Usage (from the repo root):  python eval/sql_benchmark/build_saas_questions.py
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

BENCH_DIR = Path(__file__).resolve().parent
OUT_PATH = BENCH_DIR / "saas_questions.jsonl"
REVIEW_PATH = BENCH_DIR / "REVIEW_SAAS.md"
DATA_PATH = ROOT_DIR / "data" / "saas_subscriptions.csv"
INTENTS = ["filter", "count", "aggregate", "compare", "trend"]
REVIEW_IDS = ["sg_f01", "sg_f03", "sg_f08", "sg_c01", "sg_c03", "sg_c07", "sg_a01", "sg_a03", "sg_a06",
              "sg_p01", "sg_p02", "sg_p07", "sg_t02", "sg_t05", "sg_t08"]

# --- SaaS expressions (reporting as-of date 2025-06-30) -----------------------------------------------
LIFE = "(julianday(`Cancel Date`) - julianday(`Signup Date`))"
TRIAL = f"(`Cancel Date` IS NOT NULL AND {LIFE} <= 14)"
CHURNED = f"(`Cancel Date` IS NOT NULL AND {LIFE} > 14)"
NOT_TRIAL = f"(`Cancel Date` IS NULL OR {LIFE} > 14)"
ACTIVE = "(`Cancel Date` IS NULL AND `Last Login Date` >= '2025-05-31')"
DORMANT = "(`Cancel Date` IS NULL AND `Last Login Date` < '2025-05-01')"
AT_RISK = "(`Cancel Date` IS NULL AND `Support Tickets` >= 8 AND `Last Login Date` < '2025-06-09')"
RECENT_CHURN = f"({CHURNED} AND `Cancel Date` >= '2025-04-01')"
NEW_LOGO = f"(`Signup Date` >= '2025-04-01' AND {NOT_TRIAL})"
RECENT_SIGNUP = "(`Signup Date` >= '2025-06-01')"
ARR_SCOPE = "(`Billing Cycle` = 'Annual' AND `Cancel Date` IS NULL)"
SEAT_UTIL = "100.0 * SUM(`Seats Used`) / SUM(`Seats`)"
NAIVE_UTIL = "AVG(100.0 * `Seats Used` / `Seats`)"
OVERPROV = "(`Seats` >= 10 AND `Seats Used` < 0.5 * `Seats`)"
EXPANSION = "(`Plan` IN ('Starter', 'Pro') AND `Cancel Date` IS NULL AND `Seats Used` >= 0.9 * `Seats`)"
STRATEGIC = "(`Plan` = 'Enterprise' AND `MRR` >= 5000)"
SMB = "(`Seats` < 20)"
HIGH_TOUCH = "`Acquisition Channel` IN ('Outbound Sales', 'Partner', 'Event')"
SELF_SERVE = "`Acquisition Channel` IN ('Organic Search', 'Paid Search', 'Referral')"
PAID = "`Acquisition Channel` IN ('Paid Search', 'Event')"
REGULATED = "`Industry` IN ('Healthcare', 'Financial Services', 'Education')"
TENURE = "(julianday(COALESCE(`Cancel Date`, '2025-06-30')) - julianday(`Signup Date`))"
NPS_SCORE = "100.0 * (SUM(`NPS` >= 9) - SUM(`NPS` <= 5)) / COUNT(`NPS`)"
TEXTBOOK_NPS = "100.0 * (SUM(`NPS` >= 9) - SUM(`NPS` <= 6)) / COUNT(`NPS`)"
SUPPORT_HEAVY = "(`Support Tickets` >= 5 AND `Support Tickets` > 0.2 * `Seats`)"
CHURN_RATE = (f"100.0 * SUM(CASE WHEN {CHURNED} THEN 1 ELSE 0 END) "
              f"/ SUM(CASE WHEN {NOT_TRIAL} THEN 1 ELSE 0 END)")
NAIVE_CHURN_RATE = "100.0 * AVG(`Cancel Date` IS NOT NULL)"


def fy(col):  # November-October, named by the calendar year in which it ends
    return f"(CAST(strftime('%Y', {col}) AS INTEGER) + (CAST(strftime('%m', {col}) AS INTEGER) >= 11))"


Q = []


def q(qid, set_name, intent, question, required, gold_sql, naive_sql=None, ordered=False):
    item = {"id": qid, "dataset": "saas", "set": set_name, "intent": intent, "ordered": ordered,
            "question": question, "required_chunks": required, "gold_sql": " ".join(gold_sql.split())}
    if naive_sql is not None:
        item["naive_sql"] = " ".join(naive_sql.split())
    Q.append(item)


G, P = "glossary", "plain"

# ============================== Glossary questions =====================================================
# filter
q("sg_f01", G, "filter", "list the strategic accounts in EMEA", ["ss_strategic_account"],
  f"SELECT * FROM data WHERE {STRATEGIC} AND `Region` = 'EMEA'",
  "SELECT * FROM data WHERE `Plan` = 'Enterprise' AND `Region` = 'EMEA'")
q("sg_f02", G, "filter", "show the at-risk accounts on the Pro plan", ["ss_at_risk"],
  f"SELECT * FROM data WHERE {AT_RISK} AND `Plan` = 'Pro'",
  "SELECT * FROM data WHERE `Cancel Date` IS NULL AND `Support Tickets` >= 5 AND `Plan` = 'Pro'")
q("sg_f03", G, "filter", "show the recently churned accounts in North America", ["ss_recently_churned"],
  f"SELECT * FROM data WHERE {RECENT_CHURN} AND `Region` = 'North America'",
  "SELECT * FROM data WHERE `Cancel Date` >= '2025-04-01' AND `Region` = 'North America'")
q("sg_f04", G, "filter", "list over-provisioned Pro accounts in APAC", ["ss_overprovisioned"],
  f"SELECT * FROM data WHERE {OVERPROV} AND `Plan` = 'Pro' AND `Region` = 'APAC'",
  "SELECT * FROM data WHERE `Seats Used` < 0.5 * `Seats` AND `Plan` = 'Pro' AND `Region` = 'APAC'")
q("sg_f05", G, "filter", "show expansion-ready accounts in the software industry", ["ss_expansion_ready"],
  f"SELECT * FROM data WHERE {EXPANSION} AND `Industry` = 'Software'",
  "SELECT * FROM data WHERE `Seats Used` >= 0.9 * `Seats` AND `Industry` = 'Software'")
q("sg_f06", G, "filter", "which dormant accounts are billed annually?", ["ss_dormant_account"],
  f"SELECT * FROM data WHERE {DORMANT} AND `Billing Cycle` = 'Annual'",
  "SELECT * FROM data WHERE `Cancel Date` IS NULL AND `Last Login Date` < '2025-05-31' AND `Billing Cycle` = 'Annual'")
q("sg_f07", G, "filter", "list the new logos from regulated industries", ["ss_new_logo", "ss_regulated_industry"],
  f"SELECT * FROM data WHERE {NEW_LOGO} AND {REGULATED}",
  "SELECT * FROM data WHERE `Signup Date` >= '2025-04-01' AND `Industry` IN ('Healthcare', 'Financial Services')")
q("sg_f08", G, "filter", "show support-heavy SMB accounts", ["ss_support_heavy", "ss_smb_account"],
  f"SELECT * FROM data WHERE {SUPPORT_HEAVY} AND {SMB}",
  "SELECT * FROM data WHERE `Support Tickets` >= 5 AND `Plan` = 'Starter'")
# count
q("sg_c01", G, "count", "how many active accounts do we have?", ["ss_active_account"],
  f"SELECT COUNT(*) FROM data WHERE {ACTIVE}",
  "SELECT COUNT(*) FROM data WHERE `Cancel Date` IS NULL")
q("sg_c02", G, "count", "how many trial drop-offs came from paid search?", ["ss_trial_dropoff"],
  f"SELECT COUNT(*) FROM data WHERE {TRIAL} AND `Acquisition Channel` = 'Paid Search'",
  f"SELECT COUNT(*) FROM data WHERE `Cancel Date` IS NOT NULL AND {LIFE} <= 30 AND `Acquisition Channel` = 'Paid Search'")
q("sg_c03", G, "count", "how many accounts churned in fiscal year 2024?", ["ss_churned", "ss_fiscal_year"],
  f"SELECT COUNT(*) FROM data WHERE {CHURNED} AND {fy('`Cancel Date`')} = 2024",
  "SELECT COUNT(*) FROM data WHERE `Cancel Date` IS NOT NULL AND strftime('%Y', `Cancel Date`) = '2024'")
q("sg_c04", G, "count", "how many high-touch accounts are on the Starter plan?", ["ss_high_touch"],
  f"SELECT COUNT(*) FROM data WHERE {HIGH_TOUCH} AND `Plan` = 'Starter'",
  "SELECT COUNT(*) FROM data WHERE `Acquisition Channel` IN ('Outbound Sales', 'Partner') AND `Plan` = 'Starter'")
q("sg_c05", G, "count", "count the detractors in EMEA", ["ss_detractor"],
  "SELECT COUNT(*) FROM data WHERE `NPS` <= 5 AND `Region` = 'EMEA'",
  "SELECT COUNT(*) FROM data WHERE `NPS` <= 6 AND `Region` = 'EMEA'")
q("sg_c06", G, "count", "how many enterprise-scale accounts are there?", ["ss_enterprise_scale"],
  "SELECT COUNT(*) FROM data WHERE `Seats` >= 200",
  "SELECT COUNT(*) FROM data WHERE `Plan` = 'Enterprise'")
q("sg_c07", G, "count", "how many recent signups are still subscribed?", ["ss_recent_signup"],
  f"SELECT COUNT(*) FROM data WHERE {RECENT_SIGNUP} AND `Cancel Date` IS NULL",
  "SELECT COUNT(*) FROM data WHERE `Signup Date` >= '2025-04-01' AND `Cancel Date` IS NULL")
q("sg_c08", G, "count", "how many SMB accounts are on the Pro plan?", ["ss_smb_account"],
  f"SELECT COUNT(*) FROM data WHERE {SMB} AND `Plan` = 'Pro'",
  "SELECT COUNT(*) FROM data WHERE `Seats` < 50 AND `Plan` = 'Pro'")
# aggregate
q("sg_a01", G, "aggregate", "what is our ARR?", ["ss_arr"],
  f"SELECT 12 * SUM(`MRR`) FROM data WHERE {ARR_SCOPE}",
  "SELECT 12 * SUM(`MRR`) FROM data WHERE `Cancel Date` IS NULL")
q("sg_a02", G, "aggregate", "ARPA by plan", ["ss_arpa"],
  "SELECT `Plan`, AVG(`MRR`) FROM data WHERE `Cancel Date` IS NULL GROUP BY `Plan`",
  "SELECT `Plan`, AVG(`MRR`) FROM data GROUP BY `Plan`")
q("sg_a03", G, "aggregate", "seat utilization by region", ["ss_seat_utilization"],
  f"SELECT `Region`, {SEAT_UTIL} FROM data GROUP BY `Region`",
  f"SELECT `Region`, {NAIVE_UTIL} FROM data GROUP BY `Region`")
q("sg_a04", G, "aggregate", "average tenure of churned accounts for each plan", ["ss_tenure", "ss_churned"],
  f"SELECT `Plan`, AVG({TENURE}) FROM data WHERE {CHURNED} GROUP BY `Plan`",
  f"SELECT `Plan`, AVG({LIFE}) FROM data WHERE `Cancel Date` IS NOT NULL GROUP BY `Plan`")
q("sg_a05", G, "aggregate", "average ACV of strategic accounts per region", ["ss_acv", "ss_strategic_account"],
  f"SELECT `Region`, AVG(12 * `MRR`) FROM data WHERE {STRATEGIC} GROUP BY `Region`",
  "SELECT `Region`, AVG(12 * `MRR`) FROM data WHERE `Plan` = 'Enterprise' GROUP BY `Region`")
q("sg_a06", G, "aggregate", "net promoter score by industry", ["ss_net_promoter_score"],
  f"SELECT `Industry`, {NPS_SCORE} FROM data GROUP BY `Industry`",
  f"SELECT `Industry`, {TEXTBOOK_NPS} FROM data GROUP BY `Industry`")
q("sg_a07", G, "aggregate", "total MRR of at-risk accounts by plan", ["ss_at_risk"],
  f"SELECT `Plan`, SUM(`MRR`) FROM data WHERE {AT_RISK} GROUP BY `Plan`",
  "SELECT `Plan`, SUM(`MRR`) FROM data WHERE `Cancel Date` IS NULL AND `Support Tickets` >= 5 GROUP BY `Plan`")
q("sg_a08", G, "aggregate", "average number of seats of self-serve accounts by plan", ["ss_self_serve"],
  f"SELECT `Plan`, AVG(`Seats`) FROM data WHERE {SELF_SERVE} GROUP BY `Plan`",
  "SELECT `Plan`, AVG(`Seats`) FROM data WHERE `Acquisition Channel` IN ('Organic Search', 'Referral') GROUP BY `Plan`")
# compare
q("sg_p01", G, "compare", "compare the logo churn rate of monthly and annual billing", ["ss_logo_churn_rate"],
  f"SELECT `Billing Cycle`, {CHURN_RATE} FROM data GROUP BY `Billing Cycle`",
  f"SELECT `Billing Cycle`, {NAIVE_CHURN_RATE} FROM data GROUP BY `Billing Cycle`")
q("sg_p02", G, "compare", "high-touch vs self-serve accounts: average MRR", ["ss_high_touch", "ss_self_serve"],
  f"SELECT AVG(`MRR`) FROM data GROUP BY {HIGH_TOUCH}",
  "SELECT AVG(`MRR`) FROM data GROUP BY `Acquisition Channel` = 'Outbound Sales'")
q("sg_p03", G, "compare", "seat utilization of regulated industries vs everyone else",
  ["ss_regulated_industry", "ss_seat_utilization"],
  f"SELECT {SEAT_UTIL} FROM data GROUP BY {REGULATED}",
  f"SELECT {NAIVE_UTIL} FROM data GROUP BY `Industry` IN ('Healthcare', 'Financial Services')")
q("sg_p04", G, "compare", "compare ARR between EMEA and North America", ["ss_arr"],
  f"SELECT `Region`, 12 * SUM(`MRR`) FROM data WHERE {ARR_SCOPE} AND `Region` IN ('EMEA', 'North America') GROUP BY `Region`",
  "SELECT `Region`, 12 * SUM(`MRR`) FROM data WHERE `Cancel Date` IS NULL "
  "AND `Region` IN ('EMEA', 'North America') GROUP BY `Region`")
q("sg_p05", G, "compare", "do paid-acquisition accounts have a higher logo churn rate than the rest?",
  ["ss_paid_acquisition", "ss_logo_churn_rate"],
  f"SELECT {CHURN_RATE} FROM data GROUP BY {PAID}",
  f"SELECT {NAIVE_CHURN_RATE} FROM data GROUP BY `Acquisition Channel` = 'Paid Search'")
q("sg_p06", G, "compare", "Starter vs Pro: how many expansion-ready accounts does each have?", ["ss_expansion_ready"],
  f"SELECT `Plan`, COUNT(*) FROM data WHERE {EXPANSION} GROUP BY `Plan`",
  "SELECT `Plan`, COUNT(*) FROM data WHERE `Seats Used` >= 0.8 * `Seats` AND `Plan` IN ('Starter', 'Pro') GROUP BY `Plan`")
q("sg_p07", G, "compare", "compare the number of detractors and passives", ["ss_detractor", "ss_passive"],
  "SELECT COUNT(*) FROM data WHERE `NPS` <= 8 GROUP BY `NPS` <= 5",
  "SELECT COUNT(*) FROM data WHERE `NPS` <= 8 GROUP BY `NPS` <= 6")
q("sg_p08", G, "compare", "SMB accounts vs larger accounts: average support tickets", ["ss_smb_account"],
  f"SELECT AVG(`Support Tickets`) FROM data GROUP BY {SMB}",
  "SELECT AVG(`Support Tickets`) FROM data GROUP BY `Plan` = 'Starter'")
# trend
q("sg_t01", G, "trend", "number of new logos per month", ["ss_new_logo"],
  f"SELECT strftime('%Y-%m', `Signup Date`), COUNT(*) FROM data WHERE {NEW_LOGO} GROUP BY 1 ORDER BY 1",
  "SELECT strftime('%Y-%m', `Signup Date`), COUNT(*) FROM data WHERE `Signup Date` >= '2025-04-01' GROUP BY 1 ORDER BY 1")
q("sg_t02", G, "trend", "number of signups per fiscal year", ["ss_fiscal_year"],
  f"SELECT COUNT(*) FROM data GROUP BY {fy('`Signup Date`')} ORDER BY {fy('`Signup Date`')}",
  "SELECT COUNT(*) FROM data GROUP BY strftime('%Y', `Signup Date`) ORDER BY strftime('%Y', `Signup Date`)",
  ordered=True)
q("sg_t03", G, "trend", "monthly number of churned accounts in 2024", ["ss_churned"],
  f"SELECT strftime('%Y-%m', `Cancel Date`), COUNT(*) FROM data WHERE {CHURNED} "
  "AND strftime('%Y', `Cancel Date`) = '2024' GROUP BY 1 ORDER BY 1",
  "SELECT strftime('%Y-%m', `Cancel Date`), COUNT(*) FROM data WHERE `Cancel Date` IS NOT NULL "
  "AND strftime('%Y', `Cancel Date`) = '2024' GROUP BY 1 ORDER BY 1")
q("sg_t04", G, "trend", "ARR by signup year", ["ss_arr"],
  f"SELECT strftime('%Y', `Signup Date`), 12 * SUM(`MRR`) FROM data WHERE {ARR_SCOPE} GROUP BY 1 ORDER BY 1",
  "SELECT strftime('%Y', `Signup Date`), 12 * SUM(`MRR`) FROM data WHERE `Cancel Date` IS NULL GROUP BY 1 ORDER BY 1")
q("sg_t05", G, "trend", "logo churn rate by signup year", ["ss_logo_churn_rate"],
  f"SELECT strftime('%Y', `Signup Date`), {CHURN_RATE} FROM data GROUP BY 1 ORDER BY 1",
  f"SELECT strftime('%Y', `Signup Date`), {NAIVE_CHURN_RATE} FROM data GROUP BY 1 ORDER BY 1")
q("sg_t06", G, "trend", "average tenure by signup year", ["ss_tenure"],
  f"SELECT strftime('%Y', `Signup Date`), AVG({TENURE}) FROM data GROUP BY 1 ORDER BY 1",
  "SELECT strftime('%Y', `Signup Date`), AVG(julianday('2025-06-30') - julianday(`Signup Date`)) "
  "FROM data GROUP BY 1 ORDER BY 1")
q("sg_t07", G, "trend", "how has the net promoter score changed by signup year?", ["ss_net_promoter_score"],
  f"SELECT strftime('%Y', `Signup Date`), {NPS_SCORE} FROM data GROUP BY 1 ORDER BY 1",
  f"SELECT strftime('%Y', `Signup Date`), {TEXTBOOK_NPS} FROM data GROUP BY 1 ORDER BY 1")
q("sg_t08", G, "trend", "trial drop-offs per fiscal year", ["ss_trial_dropoff", "ss_fiscal_year"],
  f"SELECT COUNT(*) FROM data WHERE {TRIAL} GROUP BY {fy('`Signup Date`')} ORDER BY {fy('`Signup Date`')}",
  f"SELECT COUNT(*) FROM data WHERE `Cancel Date` IS NOT NULL AND {LIFE} <= 30 "
  "GROUP BY strftime('%Y', `Signup Date`) ORDER BY strftime('%Y', `Signup Date`)",
  ordered=True)

# ============================== Plain questions (no definition needed) =================================
q("sp_f01", P, "filter", "show Enterprise accounts in LATAM", [],
  "SELECT * FROM data WHERE `Plan` = 'Enterprise' AND `Region` = 'LATAM'")
q("sp_f02", P, "filter", "list accounts with more than 500 seats", [],
  "SELECT * FROM data WHERE `Seats` > 500")
q("sp_f03", P, "filter", "show accounts that signed up in March 2025", [],
  "SELECT * FROM data WHERE strftime('%Y-%m', `Signup Date`) = '2025-03'")
q("sp_f04", P, "filter", "list Pro accounts in the education industry that gave an NPS of 10", [],
  "SELECT * FROM data WHERE `Plan` = 'Pro' AND `Industry` = 'Education' AND `NPS` = 10")
q("sp_c01", P, "count", "how many accounts signed up in 2023?", [],
  "SELECT COUNT(*) FROM data WHERE strftime('%Y', `Signup Date`) = '2023'")
q("sp_c02", P, "count", "how many accounts have a cancel date?", [],
  "SELECT COUNT(*) FROM data WHERE `Cancel Date` IS NOT NULL")
q("sp_c03", P, "count", "how many accounts came from referrals?", [],
  "SELECT COUNT(*) FROM data WHERE `Acquisition Channel` = 'Referral'")
q("sp_c04", P, "count", "how many accounts never answered the NPS survey?", [],
  "SELECT COUNT(*) FROM data WHERE `NPS` IS NULL")
q("sp_a01", P, "aggregate", "average MRR by plan", [],
  "SELECT `Plan`, AVG(`MRR`) FROM data GROUP BY `Plan`")
q("sp_a02", P, "aggregate", "total seats by region", [],
  "SELECT `Region`, SUM(`Seats`) FROM data GROUP BY `Region`")
q("sp_a03", P, "aggregate", "average number of support tickets per industry", [],
  "SELECT `Industry`, AVG(`Support Tickets`) FROM data GROUP BY `Industry`")
q("sp_a04", P, "aggregate", "what is the total MRR of accounts on annual billing?", [],
  "SELECT SUM(`MRR`) FROM data WHERE `Billing Cycle` = 'Annual'")
q("sp_p01", P, "compare", "compare the average MRR of monthly and annual billing", [],
  "SELECT `Billing Cycle`, AVG(`MRR`) FROM data GROUP BY `Billing Cycle`")
q("sp_p02", P, "compare", "EMEA vs APAC: number of accounts", [],
  "SELECT `Region`, COUNT(*) FROM data WHERE `Region` IN ('EMEA', 'APAC') GROUP BY `Region`")
q("sp_p03", P, "compare", "do Enterprise accounts file more support tickets than Pro accounts on average?", [],
  "SELECT `Plan`, AVG(`Support Tickets`) FROM data WHERE `Plan` IN ('Enterprise', 'Pro') GROUP BY `Plan`")
q("sp_p04", P, "compare", "compare the average NPS of software and healthcare accounts", [],
  "SELECT `Industry`, AVG(`NPS`) FROM data WHERE `Industry` IN ('Software', 'Healthcare') GROUP BY `Industry`")
q("sp_t01", P, "trend", "number of signups per month in 2024", [],
  "SELECT strftime('%Y-%m', `Signup Date`), COUNT(*) FROM data "
  "WHERE strftime('%Y', `Signup Date`) = '2024' GROUP BY 1 ORDER BY 1")
q("sp_t02", P, "trend", "number of cancellations per year", [],
  "SELECT strftime('%Y', `Cancel Date`), COUNT(*) FROM data WHERE `Cancel Date` IS NOT NULL GROUP BY 1 ORDER BY 1")
q("sp_t03", P, "trend", "total MRR by signup year", [],
  "SELECT strftime('%Y', `Signup Date`), SUM(`MRR`) FROM data GROUP BY 1 ORDER BY 1")
q("sp_t04", P, "trend", "average seats per account by signup year", [],
  "SELECT strftime('%Y', `Signup Date`), AVG(`Seats`) FROM data GROUP BY 1 ORDER BY 1")


def check(ds, chunk_ids):
    problems, results = [], {}
    ids = [x["id"] for x in Q]
    if len(ids) != len(set(ids)):
        problems.append("duplicate question IDs")
    counts = Counter((x["set"], x["intent"]) for x in Q)
    for intent in INTENTS:
        for set_name, n in [(G, 8), (P, 4)]:
            if counts[(set_name, intent)] != n:
                problems.append(f"{set_name}/{intent}: {counts[(set_name, intent)]} questions (expected {n})")
    for x in Q:
        if x["set"] == G and not 1 <= len(x["required_chunks"]) <= 2:
            problems.append(f"{x['id']}: needs {len(x['required_chunks'])} chunks")
        for cid in x["required_chunks"]:
            if cid not in chunk_ids:
                problems.append(f"{x['id']}: {cid} is not in the SaaS glossary")
        try:
            gold = run_safe_query(x["gold_sql"], ds.conn)
            naive = run_safe_query(x["naive_sql"], ds.conn) if "naive_sql" in x else None
        except Exception as e:
            problems.append(f"{x['id']}: {type(e).__name__}: {e}")
            continue
        if gold.empty:
            problems.append(f"{x['id']}: empty gold result")
        if x["intent"] == "filter" and len(gold) > 200:
            problems.append(f"{x['id']}: {len(gold)} rows (> 200)")
        if x["intent"] == "count" and gold.iloc[0, 0] == 0:
            problems.append(f"{x['id']}: count is 0")
        if x["set"] == G and results_match(gold, naive, x["ordered"]):
            problems.append(f"{x['id']}: naive guess gives the same result as gold")
        results[x["id"]] = (gold, naive)
    return problems, results


def preview(df, sql, n=3):
    """Row count plus the first n rows; for SELECT * only the columns the query mentions (plus the company)."""
    if df is None:
        return ""
    if sql.startswith("SELECT *"):
        cols = ["Company Name"] + [c for c in df.columns if f"`{c}`" in sql and c != "Company Name"]
        df = df[cols]
    head = df.head(n).round(2)
    rows = ["| " + " | ".join(str(c) for c in head.columns) + " |", "|" + "---|" * len(head.columns)]
    rows += ["| " + " | ".join("" if v is None or v != v else str(v) for v in r) + " |" for r in head.itertuples(index=False)]
    return f"**{len(df)} row{'s' if len(df) != 1 else ''}**; first {min(n, len(df))}:\n\n" + "\n".join(rows)


def write_review(chunks, results):
    by_id = {x["id"]: x for x in Q}
    L = ["# SaaS held-out benchmark: review sample", "",
         "15 of the 40 glossary questions in `saas_questions.jsonl` (3 per intent), for a human check that",
         "(a) the question uses a business term without restating it, (b) the gold SQL implements the glossary",
         "definition, and (c) the naive SQL is a plausible everyday reading that gives a different answer.",
         "The 20 plain questions (no definition needed) are listed at the end.", "",
         "The data is `data/saas_subscriptions.csv` (`make_saas_dataset.py`, seed 11); the glossary is",
         "`docs/glossary/saas.md`, with the reporting as-of date 2025-06-30. Regenerate with",
         "`python eval/sql_benchmark/build_saas_questions.py`.", ""]
    for intent in INTENTS:
        L += [f"## {intent.capitalize()}", ""]
        for qid in [i for i in REVIEW_IDS if by_id[i]["intent"] == intent]:
            x = by_id[qid]
            gold, naive = results[qid]
            L += [f"### `{qid}`: \"{x['question']}\"", "", "**Required definitions**", ""]
            L += [f"- `{c}` **{chunks[c]['term']}**: {chunks[c]['definition']}" for c in x["required_chunks"]]
            L += ["", "**Gold SQL**", "", "```sql", x["gold_sql"], "```", "", preview(gold, x["gold_sql"]), "",
                  "**Naive SQL**", "", "```sql", x["naive_sql"], "```", "", preview(naive, x["naive_sql"]), ""]
    L += ["## Plain questions (no definition needed)", "", "| ID | Intent | Question | Gold SQL | Gold rows |",
          "|---|---|---|---|---|"]
    for x in Q:
        if x["set"] == P:
            L.append(f"| {x['id']} | {x['intent']} | {x['question']} | `{x['gold_sql']}` | {len(results[x['id']][0])} |")
    REVIEW_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ds = load_dataset(DATA_PATH)
    chunks = {c["id"]: c for c in load_glossary("saas")}
    problems, results = check(ds, chunks)
    if problems:
        sys.exit("SaaS benchmark problems:\n  " + "\n  ".join(problems))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for x in Q:
            f.write(json.dumps(x) + "\n")
    write_review(chunks, results)
    print(f"Wrote {len(Q)} questions to {OUT_PATH} and the review sample to {REVIEW_PATH}")
    for x in Q:
        gold = results[x["id"]][0]
        print(f"  {x['id']}: {len(gold)} gold rows" + (f", count {gold.iloc[0, 0]}" if x["intent"] == "count" else ""))


if __name__ == "__main__":
    main()
