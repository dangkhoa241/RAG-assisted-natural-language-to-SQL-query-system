# SaaS business glossary

Internal definitions used in revenue and customer-success reporting for the accounts table
(`saas_subscriptions.csv`, one row per account). Each `##` section is one retrievable chunk; the text before
the dash is the chunk ID. An empty `Cancel Date` means the account is still subscribed. All reporting is as of
2025-06-30. The `Aliases:` line lists other names for the term.

## ss_reporting_date — Reporting as-of date
All subscription reporting is as of 2025-06-30, the close of the last reporting period. "Current", "today",
"recent" and "last N days" are measured back from 2025-06-30, never from the real current date: "the last
30 days" means dates from 2025-05-31 through 2025-06-30.
Aliases: as-of date; reporting date; as of today

## ss_trial_dropoff — Trial drop-off
An account whose `Cancel Date` is no more than 14 days after its `Signup Date` (cancel date minus signup
date ≤ 14 days). These are failed trials and never count as churn.
Aliases: trial drop-off; failed trial; trial dropout

## ss_churned — Churned account
An account whose `Cancel Date` is more than 14 days after its `Signup Date`. Cancellations within the first
14 days are trial drop-offs, not churn.
Aliases: churned account; churned customer; churned; churn; lost customer

## ss_logo_churn_rate — Logo churn rate
100 × (number of churned accounts) / (number of accounts that are not trial drop-offs). A churned account
cancelled more than 14 days after signup; a trial drop-off cancelled within 14 days. Counts accounts, not MRR.
Aliases: logo churn rate; churn rate; customer churn rate

## ss_recently_churned — Recently churned
A churned account (cancelled more than 14 days after signup) whose `Cancel Date` is on or after 2025-04-01,
the start of the last quarter before the reporting date.
Aliases: recently churned; recent churn

## ss_active_account — Active account
An account with no `Cancel Date` whose `Last Login Date` is on or after 2025-05-31 (a login within 30 days
of the reporting date 2025-06-30). Subscribed accounts that have not logged in since then are not active.
Aliases: active account; active customer; active subscriber

## ss_dormant_account — Dormant account
An account with no `Cancel Date` whose `Last Login Date` is before 2025-05-01 (no login for more than 60 days
before 2025-06-30). Subscribed accounts that last logged in 31–60 days before the reporting date are neither
active nor dormant.
Aliases: dormant account; dormant customer; inactive account

## ss_at_risk — At-risk account
An account with no `Cancel Date` that has 8 or more `Support Tickets` and whose `Last Login Date` is before
2025-06-09 (no login in the 21 days before the reporting date).
Aliases: at-risk account; at risk; churn risk

## ss_new_logo — New logo
An account with a `Signup Date` on or after 2025-04-01 that is not a trial drop-off: it has no `Cancel Date`,
or it cancelled more than 14 days after signing up.
Aliases: new logo; new customer; newly won account

## ss_recent_signup — Recent signup
Any account with a `Signup Date` on or after 2025-06-01, including trial drop-offs. Used by the onboarding
team; sales reporting uses new logos instead.
Aliases: recent signup; new signup; latest signup

## ss_arr — ARR
Annual recurring revenue counts only accounts on the Annual `Billing Cycle` that have no `Cancel Date`:
ARR = SUM(`MRR`) × 12 over those accounts. Monthly-billed accounts are excluded from ARR.
Aliases: ARR; annual recurring revenue

## ss_acv — Annual contract value (ACV)
`MRR` × 12 for a single account, for either billing cycle and whether or not the account has cancelled.
Not the same as ARR.
Aliases: ACV; annual contract value; contract value

## ss_arpa — ARPA
Average revenue per account: the total `MRR` of accounts with no `Cancel Date`, divided by the number of
those accounts. Cancelled accounts are excluded from both the total and the count.
Aliases: ARPA; average revenue per account

## ss_seat_utilization — Seat utilization
SUM(`Seats Used`) / SUM(`Seats`) × 100 over the accounts in scope: a pooled percentage, not the average of
each account's own ratio.
Aliases: seat utilization; license utilization; utilization rate

## ss_overprovisioned — Over-provisioned account
An account with at least 10 `Seats` that uses fewer than half of them (`Seats Used` < 0.5 × `Seats`).
Aliases: over-provisioned account; overprovisioned; shelfware

## ss_expansion_ready — Expansion-ready account
A Starter or Pro account with no `Cancel Date` that uses at least 90% of its seats (`Seats Used` ≥ 0.9 ×
`Seats`). Enterprise accounts are managed by account executives and never count as expansion-ready.
Aliases: expansion-ready; upsell candidate; upsell-ready

## ss_strategic_account — Strategic account
An account on the Enterprise `Plan` with an `MRR` of at least 5,000.
Aliases: strategic account; key account; tier-1 account

## ss_enterprise_scale — Enterprise-scale account
Any account with 200 or more `Seats`, regardless of `Plan` or `MRR`. Used for infrastructure capacity
planning; unrelated to strategic accounts.
Aliases: enterprise-scale; large deployment

## ss_smb_account — SMB account
An account with fewer than 20 `Seats`, regardless of `Plan` (a Pro account with 12 seats is SMB).
Aliases: SMB; small business account; small and midsize business

## ss_high_touch — High-touch account
`Acquisition Channel` is Outbound Sales, Partner, or Event (event leads are worked by the sales team).
Aliases: high-touch; sales-led; sales-sourced

## ss_self_serve — Self-serve account
`Acquisition Channel` is Organic Search, Paid Search, or Referral: the account signed up without a salesperson.
Aliases: self-serve; self-service; product-led

## ss_paid_acquisition — Paid acquisition
`Acquisition Channel` is Paid Search or Event (event sponsorships are paid). Partner and Referral are not paid.
Aliases: paid acquisition; paid channel; paid marketing

## ss_regulated_industry — Regulated industry
`Industry` is Healthcare, Financial Services, or Education.
Aliases: regulated industry; regulated sector; regulated customer

## ss_fiscal_year — Fiscal year
The fiscal year runs from November 1 to October 31 and is named after the calendar year in which it ends,
based on the date being reported (usually `Signup Date`). FY2024 = 2023-11-01 through 2024-10-31.
Aliases: fiscal year; FY

## ss_tenure — Tenure
The number of days from `Signup Date` to `Cancel Date`, or to the reporting date 2025-06-30 for accounts
with no `Cancel Date`.
Aliases: tenure; customer lifetime; account age

## ss_detractor — Detractor
An account whose `NPS` is 5 or lower. Unlike the textbook rule, a score of 6 counts as passive. Accounts
with an empty `NPS` did not respond and are neither.
Aliases: detractor

## ss_passive — Passive respondent
An account whose `NPS` is 6, 7 or 8. Passives count in neither side of the net promoter score.
Aliases: passive respondent; passives

## ss_net_promoter_score — Net promoter score
Computed over accounts with a non-empty `NPS`: 100 × (share scoring 9 or 10) − 100 × (share scoring 0–5).
Aliases: net promoter score; NPS score

## ss_support_heavy — Support-heavy account
An account with at least 5 `Support Tickets` and more than one ticket per five seats (`Support Tickets` >
0.2 × `Seats`).
Aliases: support-heavy; high-maintenance account; ticket-heavy
