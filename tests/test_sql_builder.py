import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from data_context import Dataset
from sql_builder import pick_date_column

ROOT_DIR = Path(__file__).resolve().parent.parent


def dataset(columns, date_cols):
    df = pd.DataFrame({c: ["2023-01-01"] for c in columns})
    return Dataset(df=df, conn=None, numeric_cols=[], categorical_cols=[], date_cols=set(date_cols))


def test_prefers_first_column_named_date_in_csv_order():
    ds = dataset(["Name", "Date of Admission", "Discharge Date"], {"Discharge Date", "Date of Admission"})
    assert pick_date_column(ds) == "Date of Admission"


def test_named_date_column_beats_earlier_unnamed_one():
    ds = dataset(["Created", "Order Date"], {"Created", "Order Date"})
    assert pick_date_column(ds) == "Order Date"


def test_falls_back_to_first_date_column_without_date_in_name():
    ds = dataset(["Name", "Shipped", "Created"], {"Created", "Shipped"})
    assert pick_date_column(ds) == "Shipped"


def test_no_date_columns():
    assert pick_date_column(dataset(["Name"], set())) is None


def test_trend_sql_is_identical_across_hash_seeds():
    """The real healthcare data has two date columns; the trend SQL must not depend on PYTHONHASHSEED."""
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "import warnings; warnings.simplefilter('ignore');"
        "from data_context import load_dataset; from sql_builder import build_sql;"
        "ds = load_dataset('data/healthcare_dataset.csv');"
        "print(build_sql('average billing by month in 2022', 'trend', ds)[1])"
    )
    outputs = set()
    for seed in ("0", "1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT_DIR, env=env,
                             capture_output=True, text=True, check=True).stdout
        outputs.add(out)
    assert len(outputs) == 1
    assert "`Date of Admission`" in outputs.pop()


def test_filter_sql_is_identical_across_hash_seeds():
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "import warnings; warnings.simplefilter('ignore');"
        "from data_context import load_dataset; from sql_builder import build_sql;"
        "ds = load_dataset('data/healthcare_dataset.csv');"
        "print(build_sql('count Aetna or Cigna patients with normal or abnormal results', 'count', ds)[1])"
    )
    outputs = {
        subprocess.run([sys.executable, "-c", code], cwd=ROOT_DIR, env=dict(os.environ, PYTHONHASHSEED=seed),
                       capture_output=True, text=True, check=True).stdout
        for seed in ("0", "1", "2", "3")
    }
    assert len(outputs) == 1
