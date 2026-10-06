"""The built-in sample datasets, with example questions (some need the business glossary)."""
from backend import ROOT_DIR

SAMPLES = {
    "healthcare": {
        "name": "Healthcare admissions",
        "description": "Hospital admissions: patients, conditions, insurers, billing and stay dates.",
        "path": ROOT_DIR / "data" / "healthcare_dataset.csv",
        "glossary": "healthcare",
        "examples": [
            ("average billing amount by insurance provider", False),
            ("how many patients have obesity", False),
            ("number of admissions per year", False),
            ("compare average billing between male and female patients", False),
            ("show female patients with diabetes who are over 80", False),
            ("how many unplanned admissions were for chronic conditions?", True),
            ("compare the average billing of premium and standard insurers", True),
        ],
    },
    "retail": {
        "name": "Retail orders",
        "description": "Store and online orders: customers, regions, products, revenue and returns.",
        "path": ROOT_DIR / "data" / "retail_sales.csv",
        "glossary": "retail",
        "examples": [
            ("total revenue by category", False),
            ("how many orders were returned", False),
            ("number of orders per month in 2023", False),
            ("compare total revenue of online and in-store orders", False),
            ("list returned orders with a rating of 1", False),
            ("how many repeat customers do we have?", True),
            ("net revenue per fiscal year", True),
        ],
    },
    "saas": {
        "name": "SaaS subscriptions",
        "description": "B2B software accounts: plans, billing cycles, MRR, seats, cancellations, support and NPS.",
        "path": ROOT_DIR / "data" / "saas_subscriptions.csv",
        "glossary": "saas",
        "examples": [
            ("total MRR by plan", False),
            ("number of signups per month in 2024", False),
            ("compare average MRR of monthly and annual billing", False),
            ("list Enterprise accounts in EMEA with more than 10 support tickets", False),
            ("what is our ARR?", True),
            ("how many active accounts do we have?", True),
            ("logo churn rate by plan", True),
        ],
    },
}
