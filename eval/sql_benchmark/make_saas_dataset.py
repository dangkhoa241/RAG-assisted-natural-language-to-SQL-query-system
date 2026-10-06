"""Generate data/saas_subscriptions.csv: a seeded, synthetic SaaS-subscription dataset (Stage 3C held-out domain).

1,200 accounts, one row per account, signed up between 2021-01-01 and the reporting as-of date 2025-06-30.
Nothing happens after the as-of date (no cancellations or logins), so glossary terms defined relative to it
are deterministic. Re-running this script always produces the same file.

Distributions, roughly modeled on a self-serve-plus-sales B2B tool:
  - signups grow over time; Starter 50% / Pro 35% / Enterprise 15%
  - seats: Starter 1-15, Pro ~5-80, Enterprise ~25-1,000 (log-normal); MRR = seats x per-seat price,
    with an annual-billing discount and negotiated Enterprise discounts
  - ~7% of accounts cancel within 14 days of signup (trial drop-offs); the rest churn at a monthly hazard
    that is highest for Starter and lower for annual billing
  - last login: recent for most active accounts, with a dormant tail; before the cancel date for cancelled ones
  - NPS: 0-10 survey score, missing for ~35% of accounts, lower for accounts that later churned

Usage (from the repo root):  python eval/sql_benchmark/make_saas_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 11
N_ROWS = 1200
AS_OF = pd.Timestamp("2025-06-30")
START = pd.Timestamp("2021-01-01")
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "saas_subscriptions.csv"

PLANS = {  # share, per-seat monthly price, P(annual billing), monthly churn hazard
    "Starter": (0.50, 12.0, 0.20, 0.030),
    "Pro": (0.35, 29.0, 0.45, 0.016),
    "Enterprise": (0.15, 55.0, 0.80, 0.007),
}
REGIONS = {"North America": 0.45, "EMEA": 0.30, "APAC": 0.17, "LATAM": 0.08}
INDUSTRIES = {"Software": 0.24, "Financial Services": 0.14, "Healthcare": 0.12, "Retail": 0.12,
              "Education": 0.10, "Manufacturing": 0.10, "Media": 0.08, "Nonprofit": 0.10}
CHANNELS = ["Organic Search", "Paid Search", "Referral", "Partner", "Outbound Sales", "Event"]
CHANNEL_P = {"Starter": [0.40, 0.28, 0.18, 0.06, 0.02, 0.06],
             "Pro": [0.26, 0.22, 0.15, 0.13, 0.12, 0.12],
             "Enterprise": [0.06, 0.06, 0.10, 0.24, 0.42, 0.12]}
PREFIX = ["Blue", "North", "Bright", "Clear", "Swift", "Iron", "Silver", "Green", "Red", "Summit", "Harbor",
          "Pine", "Cedar", "Oak", "River", "Stone", "Golden", "Nova", "Apex", "Peak", "Delta", "Echo", "Lumen",
          "Atlas", "Orbit", "Vertex", "Cobalt", "Amber", "Maple", "Falcon", "Quartz", "Willow", "Granite",
          "Coral", "Sage", "Juniper", "Crescent", "Beacon", "Horizon", "Meridian"]
SUFFIX = ["Analytics", "Labs", "Logistics", "Health", "Systems", "Foods", "Media", "Capital", "Studios",
          "Robotics", "Partners", "Retail", "Learning", "Works", "Networks", "Dynamics", "Solutions", "Energy",
          "Bio", "Freight", "Clinics", "Academy", "Finance", "Designs", "Manufacturing", "Supply", "Group",
          "Ventures", "Software", "Collective"]


def main():
    rng = np.random.default_rng(SEED)

    # Signups: volume grows ~2.5x from 2021 to mid-2025.
    days = pd.date_range(START, AS_OF, freq="D")
    w = np.linspace(1.0, 2.5, len(days))
    signup = pd.to_datetime(np.sort(rng.choice(days, size=N_ROWS, p=w / w.sum())))

    plan = rng.choice(list(PLANS), size=N_ROWS, p=[v[0] for v in PLANS.values()])
    seats = np.array([
        int(np.clip(1 + rng.poisson(3), 1, 15)) if p == "Starter"
        else int(np.clip(round(rng.lognormal(np.log(15), 0.6)), 5, 80)) if p == "Pro"
        else int(np.clip(round(rng.lognormal(np.log(120), 0.8)), 25, 1000))
        for p in plan])
    annual = np.array([rng.random() < PLANS[p][2] for p in plan])
    price = np.array([PLANS[p][1] for p in plan]) * np.where(annual, 0.85, 1.0)
    negotiated = np.where(plan == "Enterprise", rng.uniform(0.75, 1.0, N_ROWS), 1.0)
    mrr = np.round(seats * price * negotiated, 2)

    # Cancellations: ~7% trial drop-offs (1-14 days), a few early churns just after the trial window,
    # then an exponential time-to-churn from each plan's monthly hazard.
    cancel = []
    for s, p, a in zip(signup, plan, annual):
        u = rng.random()
        trial_p = {"Starter": 0.10, "Pro": 0.05, "Enterprise": 0.02}[p]
        if u < trial_p:
            c = s + pd.Timedelta(days=int(rng.integers(1, 15)))
        elif u < trial_p + 0.03:
            c = s + pd.Timedelta(days=int(rng.integers(15, 46)))
        else:
            hazard = PLANS[p][3] * (0.6 if a else 1.0)
            c = s + pd.Timedelta(days=int(30 * rng.exponential(1 / hazard)))
        cancel.append(c if c <= AS_OF else pd.NaT)
    cancel = pd.to_datetime(cancel)
    cancelled = cancel.notna()

    # Last login: before the cancel date for cancelled accounts; mostly recent for active ones.
    last_login = []
    for s, c in zip(signup, cancel):
        if pd.notna(c):
            d = c - pd.Timedelta(days=int(rng.exponential(8)))
        else:
            r = rng.random()
            gap = rng.exponential(4) if r < 0.68 else rng.uniform(10, 60) if r < 0.88 else rng.uniform(60, 300)
            d = AS_OF - pd.Timedelta(days=int(gap))
        last_login.append(max(d, s))
    last_login = pd.to_datetime(last_login)

    util = np.where(rng.random(N_ROWS) < 0.18, rng.beta(2, 5, N_ROWS), rng.beta(6, 2, N_ROWS))
    seats_used = np.minimum(seats, np.round(seats * util)).astype(int)
    seats_used = np.where((seats_used == 0) & (rng.random(N_ROWS) < 0.7), 1, seats_used)

    base = {"Starter": 1.0, "Pro": 2.5, "Enterprise": 5.5}
    tickets = np.array([rng.poisson(base[p] * rng.uniform(0.4, 2.2)) for p in plan])

    nps = np.where(cancelled, rng.choice(np.arange(11), size=N_ROWS, p=_nps_p(churned=True)),
                   rng.choice(np.arange(11), size=N_ROWS, p=_nps_p(churned=False))).astype(float)
    nps[rng.random(N_ROWS) < 0.35] = np.nan

    names = rng.permutation([f"{a} {b}" for a in PREFIX for b in SUFFIX])[:N_ROWS]
    df = pd.DataFrame({
        "Account ID": [f"AC-{10001 + i}" for i in range(N_ROWS)],
        "Company Name": names,
        "Plan": plan,
        "Billing Cycle": np.where(annual, "Annual", "Monthly"),
        "Signup Date": signup.strftime("%Y-%m-%d"),
        "Cancel Date": cancel.strftime("%Y-%m-%d"),
        "MRR": mrr,
        "Seats": seats,
        "Seats Used": seats_used,
        "Region": rng.choice(list(REGIONS), size=N_ROWS, p=list(REGIONS.values())),
        "Industry": rng.choice(list(INDUSTRIES), size=N_ROWS, p=list(INDUSTRIES.values())),
        "Acquisition Channel": [rng.choice(CHANNELS, p=CHANNEL_P[p]) for p in plan],
        "Last Login Date": last_login.strftime("%Y-%m-%d"),
        "Support Tickets": tickets,
        "NPS": pd.array(nps, dtype="Float64").astype("Int64"),
    })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} accounts to {OUT_PATH} ({cancelled.sum()} cancelled, {(~cancelled).sum()} subscribed)")


def _nps_p(churned: bool):
    """Probabilities for NPS scores 0..10."""
    w = (np.array([3, 2, 3, 4, 5, 8, 10, 12, 12, 8, 6], float) if churned
         else np.array([1, 1, 1, 2, 2, 4, 7, 14, 22, 24, 18], float))
    return w / w.sum()


if __name__ == "__main__":
    main()
