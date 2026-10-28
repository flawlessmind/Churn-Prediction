"""Загрузка и строгая валидация сырых данных Telco Customer Churn."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config as cfg

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Данные не соответствуют ожидаемой схеме. Сообщение содержит ВСЕ найденные проблемы."""


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Читает CSV из data/raw/. TotalCharges намеренно остаётся «как есть» (там пробелы)."""
    path = Path(path) if path else cfg.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Файл с данными не найден: {path}\n"
            "Скачайте датасет: `python scripts/download_data.py` (нужен ~/.kaggle/kaggle.json) "
            "или положите CSV вручную в data/raw/. Для проверки пайплайна без Kaggle: "
            "`python scripts/make_synthetic_data.py` (синтетические данные, только для smoke-теста)."
        )
    return pd.read_csv(path)


def validate_raw(df: pd.DataFrame, require_target: bool = True) -> dict:
    """Проверяет колонки, типы, диапазоны, пропуски, дубликаты.

    Падает с DataValidationError и перечислением всех проблем. При успехе возвращает
    краткий отчёт (число строк, число пробелов в TotalCharges, доля ушедших).
    """
    required = cfg.EXPECTED_COLUMNS if require_target else cfg.EXPECTED_COLUMNS[:-1]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Отсутствуют обязательные колонки: {missing_cols}")
    if len(df) == 0:
        raise DataValidationError("Датасет пуст (0 строк).")

    problems: list[str] = []

    # 1) пропуски (TotalCharges проверяется отдельно: там «пустые» строки из пробелов)
    for col in required:
        if col == "TotalCharges":
            continue
        n_null = int(df[col].isna().sum())
        if n_null:
            problems.append(f"{col}: {n_null} пропусков (NaN)")

    # 2) допустимые значения категориальных признаков и таргета
    for col, levels in cfg.CATEGORICAL_LEVELS.items():
        unexpected = set(df[col].dropna().unique()) - set(levels)
        if unexpected:
            problems.append(f"{col}: недопустимые значения {sorted(map(str, unexpected))}; ожидаются {levels}")
    if require_target:
        bad_t = set(df[cfg.TARGET_COL].dropna().unique()) - {"Yes", "No"}
        if bad_t:
            problems.append(f"{cfg.TARGET_COL}: недопустимые значения {sorted(map(str, bad_t))}; ожидаются Yes/No")

    # 3) типы и диапазоны числовых признаков
    numeric: dict[str, pd.Series] = {}
    for col in ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"):
        numeric[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("SeniorCitizen", "tenure", "MonthlyCharges"):
        num = numeric[col]
        n_bad_type = int((num.isna() & df[col].notna()).sum())
        if n_bad_type:
            problems.append(f"{col}: {n_bad_type} нечисловых значений")
        lo, hi = cfg.NUMERIC_RANGES[col]
        n_out = int(((num < lo) | (num > hi)).sum())
        if n_out:
            problems.append(f"{col}: {n_out} значений вне диапазона [{lo}, {hi}]")

    # 4) TotalCharges — классическая ловушка: пробелы вместо чисел
    raw_tc = df["TotalCharges"]
    blank = raw_tc.isna() | raw_tc.astype(str).str.strip().eq("")
    garbage = numeric["TotalCharges"].isna() & ~blank
    if garbage.any():
        problems.append(f"TotalCharges: {int(garbage.sum())} значений, не являющихся числом и не пустых")
    blank_bad = blank & (numeric["tenure"] != 0)
    if blank_bad.any():
        problems.append(
            f"TotalCharges: {int(blank_bad.sum())} пустых значений у клиентов с tenure > 0 "
            "(ожидается, что пустые только у tenure = 0)"
        )
    lo, hi = cfg.NUMERIC_RANGES["TotalCharges"]
    n_out = int(((numeric["TotalCharges"] < lo) | (numeric["TotalCharges"] > hi)).sum())
    if n_out:
        problems.append(f"TotalCharges: {n_out} значений вне диапазона [{lo}, {hi}]")

    # 5) дубликаты ID
    n_dup = int(df[cfg.ID_COL].duplicated().sum())
    if n_dup:
        problems.append(f"{cfg.ID_COL}: {n_dup} дубликатов")

    if problems:
        raise DataValidationError(
            "Валидация данных не пройдена:\n - " + "\n - ".join(problems)
        )

    n_blank = int(blank.sum())
    if n_blank:
        logger.warning(
            "TotalCharges: %d пустых значений (все у tenure=0) — будут заполнены MonthlyCharges*tenure в FE",
            n_blank,
        )
    report = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "total_charges_blank": n_blank}
    if require_target:
        report["churn_rate"] = float((df[cfg.TARGET_COL] == "Yes").mean())
    logger.info("Валидация пройдена: %s", report)
    return report


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Сырой DataFrame → (X с 19 сырыми признаками, y в виде 0/1)."""
    X = df[cfg.RAW_FEATURES].copy()
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")  # пробелы → NaN
    y = (df[cfg.TARGET_COL] == "Yes").astype(int)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_dataset(path: Path | str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Загрузка + валидация + подготовка (X, y)."""
    df = load_raw(path)
    validate_raw(df)
    return prepare_xy(df)
