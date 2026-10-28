"""Feature engineering — единственный источник правды и для обучения, и для inference.

ChurnFeatureEngineer — sklearn-трансформер, входит в Pipeline и сохраняется вместе с моделью,
поэтому на проде признаки считаются ровно так же, как при обучении.

Принятые решения (с обоснованием):
* TotalCharges: пробелы → NaN → MonthlyCharges * tenure. Пустые значения бывают только у клиентов с
  tenure = 0 (ещё не было счёта), поэтому 0 семантически верно, а медиана исказила бы «нового» клиента.
* Yes/No → 1/0. Для «No internet service» / «No phone service» берём 0: эта информация уже содержится
  в InternetService / PhoneService, а отдельные dummy-колонки дали бы идеальную мультиколлинеарность.
* One-hot только для InternetService, Contract, PaymentMethod (нет естественного порядка). Набор
  уровней фиксирован в config → одинаковый порядок/состав колонок в train и inference; неизвестный
  уровень даёт нулевую строку dummy, а не падение.
* Выбросы: у tenure/MonthlyCharges выбросов нет (ограниченные диапазоны), TotalCharges и
  avg_charge_per_month скошены вправо. Деревьям монотонные преобразования не нужны, поэтому НЕ меняем
  распределение, а только режем верхний хвост по обучающему 99.5-му перцентилу (winsorizing) — это защищает
  линейную модель и сервис от аномальных входов, почти не затрагивая реальные данные.
* Новые признаки:
    avg_charge_per_month = TotalCharges / (tenure + 1)  — «реальный» средний чек; разница с MonthlyCharges
        показывает недавнее повышение/снижение тарифа.
    num_services — число подключённых доп. услуг: вовлечённость/«прилипание» клиента.
    is_new_customer = tenure < 6 — первые месяцы — зона максимального оттока (нелинейность по tenure).
    has_autopay — автоплатёж (карта/банк): ниже трение при оплате, ниже риск ухода.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src import config as cfg


def _slug(text: str) -> str:
    """'Bank transfer (automatic)' -> 'Bank_transfer_automatic' (безопасно для LightGBM/SHAP)."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Сырые 19 признаков Telco -> числовая матрица признаков (pd.DataFrame)."""

    ONEHOT_COLS = ["InternetService", "Contract", "PaymentMethod"]
    CLIP_COLS = ["TotalCharges", "avg_charge_per_month"]

    def __init__(self, clip_upper_quantile: float = 0.995, new_customer_threshold: int = 6):
        self.clip_upper_quantile = clip_upper_quantile
        self.new_customer_threshold = new_customer_threshold

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _check_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ChurnFeatureEngineer ожидает pandas.DataFrame с сырыми колонками Telco.")
        missing = [c for c in cfg.RAW_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Во входных данных нет обязательных колонок: {missing}")

    def _build(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_input(X)
        tenure = pd.to_numeric(X["tenure"], errors="coerce")
        monthly = pd.to_numeric(X["MonthlyCharges"], errors="coerce")
        if tenure.isna().any() or monthly.isna().any():
            raise ValueError("tenure и MonthlyCharges не должны содержать пропусков/нечисловых значений.")
        total = pd.to_numeric(X["TotalCharges"], errors="coerce")      # пробелы -> NaN
        total = total.fillna(monthly * tenure)                          # NaN -> MonthlyCharges * tenure

        f: dict[str, pd.Series] = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "SeniorCitizen": pd.to_numeric(X["SeniorCitizen"], errors="coerce").fillna(0),
            "is_male": (X["gender"] == "Male"),
        }
        for c in cfg.YES_NO_COLS + ["MultipleLines"] + cfg.ADDON_COLS:
            f[c] = X[c] == "Yes"
        for c in self.ONEHOT_COLS:
            for level in cfg.CATEGORICAL_LEVELS[c]:
                f[f"{c}_{_slug(level)}"] = X[c] == level

        # --- новые признаки
        f["avg_charge_per_month"] = total / (tenure + 1)
        f["num_services"] = sum((X[c] == "Yes").astype(int) for c in cfg.ADDON_COLS)
        f["is_new_customer"] = tenure < self.new_customer_threshold
        f["has_autopay"] = X["PaymentMethod"].astype(str).str.contains("automatic", regex=False)

        return pd.DataFrame(f, index=X.index).astype(float)

    # ------------------------------------------------------------------ sklearn API
    def fit(self, X: pd.DataFrame, y=None) -> "ChurnFeatureEngineer":
        base = self._build(X)
        self.clip_bounds_ = {c: float(base[c].quantile(self.clip_upper_quantile)) for c in self.CLIP_COLS}
        self.feature_names_ = list(base.columns)
        self.n_features_in_ = len(cfg.RAW_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["clip_bounds_", "feature_names_"])
        out = self._build(X)
        for col, upper in self.clip_bounds_.items():
            out[col] = out[col].clip(lower=0, upper=upper)
        return out[self.feature_names_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)
