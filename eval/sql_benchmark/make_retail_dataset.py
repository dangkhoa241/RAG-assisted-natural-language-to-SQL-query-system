"""Generate data/retail_sales.csv: a seeded, synthetic retail-sales dataset for the SQL benchmark.

1,000 orders across 2023-2024 with date, categorical, numeric and free-text columns.
Re-running this script always produces the same file.

Usage (from the repo root):  python eval/sql_benchmark/make_retail_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_ROWS = 1000
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "retail_sales.csv"

PRODUCTS = {
    "Electronics": {"Wireless Earbuds": 79.0, "Smartwatch": 199.0, "Bluetooth Speaker": 59.0, "Laptop Stand": 39.0},
    "Clothing": {"Denim Jacket": 89.0, "Running Shoes": 120.0, "Wool Sweater": 65.0, "Graphic Tee": 22.0},
    "Home & Kitchen": {"Air Fryer": 110.0, "Coffee Maker": 85.0, "Knife Set": 70.0, "Throw Blanket": 35.0},
    "Sports": {"Yoga Mat": 30.0, "Dumbbell Set": 95.0, "Tennis Racket": 140.0, "Water Bottle": 18.0},
    "Beauty": {"Face Serum": 45.0, "Perfume": 75.0, "Hair Dryer": 60.0, "Lip Balm Pack": 12.0},
}
REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Online", "In-Store"]
SEGMENTS = ["Consumer", "Corporate", "Small Business"]
FIRST = ["Ava", "Liam", "Mia", "Noah", "Zoe", "Ethan", "Ivy", "Lucas", "Nora", "Owen",
         "Ruby", "Caleb", "Leah", "Mason", "Elena", "Jack", "Aria", "Henry", "Maya", "Leo"]
LAST = ["Nguyen", "Smith", "Garcia", "Patel", "Kim", "Brown", "Lopez", "Chen", "Davis", "Wilson",
        "Martin", "Clark", "Lee", "Walker", "Young", "Hall", "Allen", "King", "Wright", "Scott"]


def main():
    rng = np.random.default_rng(SEED)

    # Seasonality: November and December get roughly twice the order volume.
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    weights = np.where(days.month.isin([11, 12]), 2.0, 1.0)
    weights = weights * np.where(days.year == 2024, 1.15, 1.0)  # mild year-over-year growth
    order_dates = np.sort(rng.choice(days, size=N_ROWS, p=weights / weights.sum()))

    categories = rng.choice(list(PRODUCTS), size=N_ROWS, p=[0.26, 0.24, 0.2, 0.16, 0.14])
    products, prices = [], []
    for cat in categories:
        name = rng.choice(list(PRODUCTS[cat]))
        products.append(name)
        prices.append(round(PRODUCTS[cat][name] * rng.uniform(0.9, 1.1), 2))

    channels = rng.choice(CHANNELS, size=N_ROWS, p=[0.6, 0.4])
    payment = np.where(
        channels == "Online",
        rng.choice(["Credit Card", "Debit Card", "PayPal"], size=N_ROWS, p=[0.5, 0.2, 0.3]),
        rng.choice(["Credit Card", "Debit Card", "Cash"], size=N_ROWS, p=[0.45, 0.3, 0.25]),
    )
    quantity = rng.integers(1, 9, size=N_ROWS)
    discount = rng.choice([0.0, 0.05, 0.1, 0.15, 0.2], size=N_ROWS, p=[0.45, 0.2, 0.18, 0.1, 0.07])
    revenue = np.round(quantity * np.array(prices) * (1 - discount), 2)
    rating = rng.choice([1, 2, 3, 4, 5], size=N_ROWS, p=[0.05, 0.08, 0.2, 0.37, 0.3])
    returned = np.where(rng.random(N_ROWS) < np.where(rating <= 2, 0.35, 0.06), "Yes", "No")

    df = pd.DataFrame({
        "Order ID": np.arange(10001, 10001 + N_ROWS),
        "Order Date": pd.to_datetime(order_dates).strftime("%Y-%m-%d"),
        "Customer Name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(N_ROWS)],
        "Customer Segment": rng.choice(SEGMENTS, size=N_ROWS, p=[0.6, 0.25, 0.15]),
        "Customer Age": rng.integers(18, 76, size=N_ROWS),
        "Region": rng.choice(REGIONS, size=N_ROWS),
        "Channel": channels,
        "Category": categories,
        "Product": products,
        "Unit Price": prices,
        "Quantity": quantity,
        "Discount": discount,
        "Revenue": revenue,
        "Payment Method": payment,
        "Rating": rating,
        "Returned": returned,
    })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
