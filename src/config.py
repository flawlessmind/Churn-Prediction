"""Центральная конфигурация проекта: пути, константы, словари допустимых значений."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- пути
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT / "data" / "raw"
RAW_CSV = DATA_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
HOLDOUT_CSV = PROCESSED_DIR / "holdout.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_DIR = REPORTS_DIR / "eda"
MLFLOW_DIR = ROOT / "mlflow"
MLFLOW_EXPERIMENT = "telco-churn"

# ---------------------------------------------------------------- эксперимент
RANDOM_STATE = 42
TEST_SIZE = 0.20          # hold-out, который НЕ участвует ни в подборе гиперпараметров, ни в выборе порога
N_SPLITS = 5              # StratifiedKFold
BUDGET_FRACTION = 0.20    # бюджет обзвона: топ-20% клиентов по риску
MODEL_VERSION = "1.0.0"

# ---------------------------------------------------------------- схема данных
ID_COL = "customerID"
TARGET_COL = "Churn"

YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

_YES_NO = ["Yes", "No"]
_YES_NO_NOINET = ["Yes", "No", "No internet service"]

# допустимые значения всех категориальных признаков (используются и в валидации, и в FE, и в API)
CATEGORICAL_LEVELS: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": _YES_NO_NOINET,
    "OnlineBackup": _YES_NO_NOINET,
    "DeviceProtection": _YES_NO_NOINET,
    "TechSupport": _YES_NO_NOINET,
    "StreamingTV": _YES_NO_NOINET,
    "StreamingMovies": _YES_NO_NOINET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

# допустимые диапазоны числовых признаков (широкие «санитарные» границы, не статистические)
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 120),
    "MonthlyCharges": (0, 500),
    "TotalCharges": (0, 100_000),
}

RAW_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
EXPECTED_COLUMNS = [ID_COL, *RAW_FEATURES, TARGET_COL]   # 21 столбец
