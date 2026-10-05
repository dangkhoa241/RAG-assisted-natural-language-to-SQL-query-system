import numpy as np
import pandas as pd

from sql_metrics import normalize_value, results_match


def df(rows, cols):
    return pd.DataFrame(rows, columns=cols)


GOLD = df([("Female", 25495.0101), ("Male", 25386.4)], ["Gender", "AVG(`Billing Amount`)"])


def test_identical_results_match():
    assert results_match(GOLD, GOLD.copy())


def test_row_order_ignored_when_unordered():
    pred = df([("Male", 25386.4), ("Female", 25495.0101)], ["g", "v"])
    assert results_match(GOLD, pred)


def test_row_order_matters_when_ordered():
    pred = df([("Male", 25386.4), ("Female", 25495.0101)], ["g", "v"])
    assert not results_match(GOLD, pred, ordered=True)
    assert results_match(GOLD, GOLD.copy(), ordered=True)


def test_column_order_and_names_ignored():
    pred = df([(25495.0101, "Female"), (25386.4, "Male")], ["avg_billing", "gender"])
    assert results_match(GOLD, pred)


def test_floats_rounded_to_two_decimals():
    pred = df([("Female", 25495.01), ("Male", 25386.40)], ["g", "v"])  # e.g. ROUND(AVG(...), 2)
    assert results_match(GOLD, pred)
    wrong = df([("Female", 25495.02), ("Male", 25386.40)], ["g", "v"])
    assert not results_match(GOLD, wrong)


def test_extra_predicted_columns_allowed():
    gold = df([(154,)], ["COUNT(*)"])
    pred = df([("Obesity", 154)], ["condition", "n"])
    assert results_match(gold, pred)


def test_missing_column_fails():
    pred = df([("Female",), ("Male",)], ["g"])
    assert not results_match(GOLD, pred)


def test_different_row_count_fails():
    assert not results_match(GOLD, GOLD.head(1))
    assert not results_match(GOLD, None)


def test_values_must_stay_paired_across_columns():
    # Same column multisets, but the pairing between label and value is swapped.
    pred = df([("Female", 25386.4), ("Male", 25495.0101)], ["g", "v"])
    assert not results_match(GOLD, pred)


def test_numeric_strings_and_case_normalized():
    gold = df([("2023", 10), ("2024", 12)], ["year", "n"])
    pred = df([(2023, 10.0), (2024, 12.0)], ["year", "n"])
    assert results_match(gold, pred)
    assert normalize_value("  Male ") == "male"
    assert normalize_value(np.int64(5)) == 5.0
    assert normalize_value(float("nan")) is None


def test_empty_results_match():
    assert results_match(df([], ["a"]), df([], ["b"]))


def test_duplicate_valued_columns_are_assigned_correctly():
    # Two gold columns with identical values: the backtracking assignment must still succeed.
    gold = df([(1, 1, "a"), (2, 2, "b")], ["x", "y", "z"])
    pred = df([("a", 1, 1), ("b", 2, 2)], ["z", "x", "y"])
    assert results_match(gold, pred)
