import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src import config as cfg
from src.data import prepare_xy
from src.features import ChurnFeatureEngineer


@pytest.fixture(scope="module")
def Xy(raw_df):
    return prepare_xy(raw_df)


def test_no_nans_and_all_numeric(Xy):
    X, _ = Xy
    F = ChurnFeatureEngineer().fit_transform(X)
    assert not F.isna().any().any()
    assert all(dt.kind == "f" for dt in F.dtypes)


def test_blank_total_charges_filled_with_monthly_times_tenure(raw_df):
    # сырой DataFrame без prepare_xy: TotalCharges содержит строку-пробел, как в оригинале
    row = raw_df[raw_df["tenure"] == 0].head(1)
    assert (row["TotalCharges"] == " ").all()
    fe = ChurnFeatureEngineer().fit(raw_df[cfg.RAW_FEATURES])
    out = fe.transform(row[cfg.RAW_FEATURES])
    assert out["TotalCharges"].iloc[0] == 0.0


def test_new_features(Xy):
    X, _ = Xy
    F = ChurnFeatureEngineer(clip_upper_quantile=1.0).fit_transform(X)
    np.testing.assert_allclose(F["avg_charge_per_month"], F["TotalCharges"] / (F["tenure"] + 1))
    assert ((F["tenure"] < 6).astype(float) == F["is_new_customer"]).all()
    assert F["num_services"].between(0, 6).all()


def test_train_inference_consistency(Xy):
    """Признаки одной строки при inference == признаки той же строки в батче при обучении."""
    X, _ = Xy
    fe = ChurnFeatureEngineer().fit(X)
    batch = fe.transform(X.head(50))
    single = pd.concat([fe.transform(X.iloc[[i]]) for i in range(50)])
    pd.testing.assert_frame_equal(batch, single)
    assert list(batch.columns) == fe.feature_names_


def test_unknown_category_does_not_crash(Xy):
    X, _ = Xy
    fe = ChurnFeatureEngineer().fit(X)
    row = X.head(1).copy()
    row["Contract"] = "Lifetime"
    out = fe.transform(row)
    assert out[[c for c in out.columns if c.startswith("Contract_")]].sum(axis=1).iloc[0] == 0


def test_missing_column_raises(Xy):
    X, _ = Xy
    with pytest.raises(ValueError, match="нет обязательных колонок"):
        ChurnFeatureEngineer().fit(X.drop(columns=["Contract"]))


def test_clipping_limits_extreme_values(Xy):
    X, _ = Xy
    fe = ChurnFeatureEngineer().fit(X)
    row = X.head(1).copy()
    row["TotalCharges"] = 99_999.0
    assert fe.transform(row)["TotalCharges"].iloc[0] == pytest.approx(fe.clip_bounds_["TotalCharges"])


def test_clonable(Xy):
    clone(ChurnFeatureEngineer(new_customer_threshold=3))
