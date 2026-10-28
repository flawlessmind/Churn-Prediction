import pytest

from src.data import DataValidationError, prepare_xy, validate_raw


def test_valid_data_passes(raw_df):
    rep = validate_raw(raw_df)
    assert rep["n_rows"] == len(raw_df) and rep["total_charges_blank"] > 0
    X, y = prepare_xy(raw_df)
    assert set(y.unique()) <= {0, 1} and X["TotalCharges"].dtype.kind == "f"


def test_missing_column_fails(raw_df):
    with pytest.raises(DataValidationError, match="Отсутствуют"):
        validate_raw(raw_df.drop(columns=["tenure"]))


def test_bad_values_fail_with_all_problems(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "Contract"] = "Weekly"
    bad.loc[1, "tenure"] = -5
    bad.loc[2, "MonthlyCharges"] = 10_000
    bad.loc[3, "TotalCharges"] = "abc"
    with pytest.raises(DataValidationError) as e:
        validate_raw(bad)
    msg = str(e.value)
    assert all(s in msg for s in ["Contract", "tenure", "MonthlyCharges", "TotalCharges"])


def test_blank_total_charges_with_positive_tenure_fails(raw_df):
    bad = raw_df.copy()
    i = bad.index[bad["tenure"] > 0][0]
    bad.loc[i, "TotalCharges"] = " "
    with pytest.raises(DataValidationError, match="tenure > 0"):
        validate_raw(bad)
