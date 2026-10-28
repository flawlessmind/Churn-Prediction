"""Инференс: загрузка сохранённого Pipeline (FE + модель) и предсказание по сырым признакам клиента."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src import config as cfg


class ModelNotLoadedError(RuntimeError):
    pass


class ChurnPredictor:
    def __init__(self, pipeline, metadata: dict):
        self.pipeline = pipeline
        self.metadata = metadata
        thr = metadata["thresholds"]
        self.decision_threshold: float = thr["f1_opt"]
        self.high_threshold: float = thr["top20_budget"]
        # «medium» — от порога решения до порога топ-20%; если порядок обратный, medium-зона пуста
        self.medium_threshold: float = min(self.decision_threshold, self.high_threshold)

    @classmethod
    def load(cls, model_path: Path | str, metadata_path: Path | str | None = None) -> "ChurnPredictor":
        model_path = Path(model_path)
        metadata_path = Path(metadata_path) if metadata_path else model_path.with_name("model_metadata.json")
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}. Обучите модель: python -m src.train")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Файл метаданных не найден: {metadata_path}")
        return cls(joblib.load(model_path), json.loads(metadata_path.read_text(encoding="utf-8")))

    def segment(self, p: float) -> str:
        if p >= self.high_threshold:
            return "high"
        if p >= self.medium_threshold:
            return "medium"
        return "low"

    def predict_frame(self, X: pd.DataFrame) -> list[dict]:
        proba = self.pipeline.predict_proba(X[cfg.RAW_FEATURES])[:, 1]
        return [
            {"churn_probability": round(float(p), 4),
             "churn_prediction": int(p >= self.decision_threshold),
             "risk_segment": self.segment(float(p))}
            for p in proba
        ]

    def info(self) -> dict:
        m = self.metadata
        return {
            "model_name": m["model_name"], "model_version": m["model_version"], "trained_at": m["trained_at"],
            "thresholds": m["thresholds"], "holdout_metrics": m["holdout_metrics"],
            "n_train": m["n_train"], "n_holdout": m["n_holdout"], "libs": m.get("libs", {}),
        }
