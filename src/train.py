"""Обучение и сравнение моделей: Dummy / LogisticRegression / RandomForest / LightGBM(+Optuna).

Протокол (чтобы метрики были честными):
  1. Стратифицированный hold-out 20% — трогаем ТОЛЬКО для финальной оценки.
  2. На train: StratifiedKFold(5) -> CV PR-AUC (по ней выбираем гиперпараметры и модель-чемпион)
     и OOF-предсказания (по ним выбираем порог — без подглядывания в hold-out).
  3. Чемпион среди «основных» моделей (RF / LightGBM) дообучается на всём train и сохраняется.

Запуск:  python -m src.train --n-trials 30
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as cfg
from src.data import load_dataset, load_raw, validate_raw
from src.features import ChurnFeatureEngineer
from src.metrics import budget_threshold, evaluate, pick_f1_threshold
from src.plots import plot_confusion, plot_gain_curves, plot_pr_curves

logger = logging.getLogger("train")
KK = int(cfg.BUDGET_FRACTION * 100)  # 20


# ----------------------------------------------------------------------------- модели
def make_lgbm(params: dict | None = None) -> lgb.LGBMClassifier:
    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=cfg.RANDOM_STATE,
                n_jobs=1, verbosity=-1, deterministic=True, force_row_wise=True)
    return lgb.LGBMClassifier(**{**base, **(params or {})})


def make_pipeline(model) -> Pipeline:
    return Pipeline([("fe", ChurnFeatureEngineer()), ("model", model)])


def build_candidates(lgbm_params: dict) -> dict[str, Pipeline]:
    return {
        "dummy_prior": make_pipeline(DummyClassifier(strategy="prior")),
        "logistic_regression": Pipeline([
            ("fe", ChurnFeatureEngineer()), ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=1.0, random_state=cfg.RANDOM_STATE)),
        ]),
        "random_forest": make_pipeline(RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, max_features="sqrt",
            n_jobs=-1, random_state=cfg.RANDOM_STATE)),
        "lightgbm_optuna": make_pipeline(make_lgbm(lgbm_params)),
    }


MAIN_MODELS = ("random_forest", "lightgbm_optuna")   # из них выбирается чемпион (SHAP TreeExplainer)


# ----------------------------------------------------------------------------- CV
def cv_oof(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, skf: StratifiedKFold):
    """Возвращает OOF-вероятности и PR-AUC по каждому фолду."""
    oof = np.zeros(len(y))
    fold_ap = []
    for tr, va in skf.split(X, y):
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        fold_ap.append(average_precision_score(y.iloc[va], p))
    return oof, np.array(fold_ap)


# ----------------------------------------------------------------------------- Optuna
def tune_lgbm(X, y, skf, n_trials: int):
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 64),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        _, aps = cv_oof(make_pipeline(make_lgbm(params)), X, y, skf)
        trial.set_user_attr("cv_std", float(aps.std()))
        return float(aps.mean())

    study = optuna.create_study(direction="maximize", study_name="lgbm_pr_auc",
                                sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_STATE))

    def log_cb(_study, trial):
        mlflow.log_metric("trial_cv_pr_auc", trial.value, step=trial.number)
        mlflow.log_metric("best_cv_pr_auc_so_far", _study.best_value, step=trial.number)

    study.optimize(objective, n_trials=n_trials, callbacks=[log_cb])
    return study


# ----------------------------------------------------------------------------- MLflow
def setup_mlflow() -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{cfg.MLFLOW_DIR / 'mlflow.db'}")
    if uri.startswith("sqlite"):
        cfg.MLFLOW_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(cfg.MLFLOW_EXPERIMENT) is None:
        kwargs = {}
        if uri.startswith("sqlite"):
            kwargs["artifact_location"] = (cfg.MLFLOW_DIR / "artifacts").resolve().as_uri()
        mlflow.create_experiment(cfg.MLFLOW_EXPERIMENT, **kwargs)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT)
    logger.info("MLflow tracking URI: %s", uri)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) if isinstance(v, str) else f"{v:.3f}" for v in r) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main(n_trials: int, data_path: str | None) -> None:
    t0 = time.time()
    for d in (cfg.MODELS_DIR, cfg.REPORTS_DIR, cfg.FIGURES_DIR, cfg.PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 1) данные
    df_raw = load_raw(data_path)
    validate_raw(df_raw)
    X, y = load_dataset(data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE)
    X_tr, X_te = X_tr.reset_index(drop=True), X_te.reset_index(drop=True)
    y_tr, y_te = y_tr.reset_index(drop=True), y_te.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.RANDOM_STATE)
    logger.info("train=%d test=%d churn_rate train=%.3f test=%.3f",
                len(y_tr), len(y_te), y_tr.mean(), y_te.mean())

    setup_mlflow()
    with mlflow.start_run(run_name="churn-training"):
        mlflow.log_params({"random_state": cfg.RANDOM_STATE, "n_splits": cfg.N_SPLITS,
                           "test_size": cfg.TEST_SIZE, "budget_fraction": cfg.BUDGET_FRACTION,
                           "n_optuna_trials": n_trials, "n_train": len(y_tr), "n_test": len(y_te)})

        # 2) Optuna для LightGBM
        with mlflow.start_run(run_name="lightgbm_optuna_search", nested=True):
            study = tune_lgbm(X_tr, y_tr, skf, n_trials)
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_cv_pr_auc", study.best_value)
            trials_csv = cfg.REPORTS_DIR / "optuna_trials.csv"
            study.trials_dataframe().to_csv(trials_csv, index=False)
            mlflow.log_artifact(str(trials_csv))
        best_params = {**study.best_params, "subsample_freq": 1}
        logger.info("Optuna best CV PR-AUC=%.4f params=%s", study.best_value, best_params)

        # 3) CV + hold-out для всех моделей
        results: dict[str, dict] = {}
        holdout_scores: dict[str, np.ndarray] = {}
        fitted: dict[str, Pipeline] = {}
        for name, pipe in build_candidates(best_params).items():
            with mlflow.start_run(run_name=name, nested=True):
                oof, aps = cv_oof(pipe, X_tr, y_tr, skf)
                thr_f1 = pick_f1_threshold(y_tr, oof)
                thr_budget = budget_threshold(oof, cfg.BUDGET_FRACTION)

                model = clone(pipe).fit(X_tr, y_tr)
                p_te = model.predict_proba(X_te)[:, 1]
                m = evaluate(y_te, p_te, thr_f1, cfg.BUDGET_FRACTION)
                m.update(cv_pr_auc_mean=float(aps.mean()), cv_pr_auc_std=float(aps.std()),
                         threshold_f1=thr_f1, threshold_top20=thr_budget)

                mlflow.log_param("model", name)
                mlflow.log_params({f"model__{k}": v for k, v in pipe.named_steps["model"].get_params().items()
                                   if isinstance(v, (int, float, str, bool, type(None)))})
                mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})
                results[name], holdout_scores[name], fitted[name] = m, p_te, model
                logger.info("%-20s CV PR-AUC=%.4f±%.4f | hold-out PR-AUC=%.4f ROC-AUC=%.4f F1=%.4f Recall@%d=%.4f",
                            name, aps.mean(), aps.std(), m["pr_auc"], m["roc_auc"], m["f1"], KK, m[f"recall_at_{KK}"])

        # 4) чемпион среди основных моделей — по CV PR-AUC (не по hold-out!)
        champion = max(MAIN_MODELS, key=lambda n: results[n]["cv_pr_auc_mean"])
        champ_m = results[champion]
        logger.info("Чемпион: %s (CV PR-AUC=%.4f)", champion, champ_m["cv_pr_auc_mean"])

        # 5) отчёт и графики
        table = pd.DataFrame([{
            "model": n, "cv_pr_auc": r["cv_pr_auc_mean"], "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"],
            "f1": r["f1"], f"recall@{KK}%": r[f"recall_at_{KK}"], f"lift@{KK}%": r[f"lift_at_{KK}"],
        } for n, r in results.items()])
        table.to_csv(cfg.REPORTS_DIR / "model_comparison.csv", index=False)

        thr_f1, thr_top = champ_m["threshold_f1"], champ_m["threshold_top20"]
        (cfg.REPORTS_DIR / "results.md").write_text(
            f"# Результаты (hold-out, n={len(y_te)}, доля ушедших={y_te.mean():.3f})\n\n"
            f"{_md_table(table)}\n\n"
            f"`cv_pr_auc` — среднее по {cfg.N_SPLITS} фолдам на train; остальные метрики — на hold-out. "
            f"F1/precision/recall считаются на пороге, выбранном по OOF-предсказаниям train.\n\n"
            f"**Чемпион:** `{champion}`. **Порог (max F1 на OOF):** {thr_f1:.3f}. "
            f"**Порог «топ-{KK}% бюджет» (OOF):** {thr_top:.3f}.\n\n"
            f"На hold-out при обзвоне топ-{KK}% клиентов ловим **{champ_m[f'recall_at_{KK}']:.1%}** всех уходов "
            f"(precision {champ_m[f'precision_at_{KK}']:.1%}, lift ×{champ_m[f'lift_at_{KK}']:.2f}).\n",
            encoding="utf-8")

        plot_pr_curves(y_te, holdout_scores, cfg.FIGURES_DIR / "pr_curves.png")
        plot_gain_curves(y_te, holdout_scores, cfg.BUDGET_FRACTION, cfg.FIGURES_DIR / "gain_curves.png")
        plot_confusion(champ_m["tn"], champ_m["fp"], champ_m["fn"], champ_m["tp"],
                       cfg.FIGURES_DIR / "confusion_matrix.png",
                       f"{champion}, порог={thr_f1:.2f}")

        # 6) сохранение модели, метаданных и hold-out
        joblib.dump(fitted[champion], cfg.MODEL_PATH)
        holdout_df = X_te.copy()
        holdout_df[cfg.TARGET_COL] = y_te.values
        holdout_df.to_csv(cfg.HOLDOUT_CSV, index=False)

        metadata = {
            "model_name": champion,
            "model_version": cfg.MODEL_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "random_state": cfg.RANDOM_STATE,
            "primary_metric": "pr_auc",
            "thresholds": {"f1_opt": thr_f1, "top20_budget": thr_top},
            "holdout_metrics": {k: v for k, v in champ_m.items() if isinstance(v, (int, float))},
            "all_models_holdout": {n: {k: v for k, v in r.items() if isinstance(v, (int, float))}
                                   for n, r in results.items()},
            "best_params": best_params,
            "n_train": int(len(y_tr)), "n_holdout": int(len(y_te)),
            "feature_names": list(fitted[champion].named_steps["fe"].feature_names_),
            "libs": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        }
        cfg.METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        for p in (cfg.MODEL_PATH, cfg.METADATA_PATH, cfg.REPORTS_DIR / "results.md",
                  cfg.REPORTS_DIR / "model_comparison.csv"):
            mlflow.log_artifact(str(p))
        for p in cfg.FIGURES_DIR.glob("*.png"):
            mlflow.log_artifact(str(p), artifact_path="figures")
        mlflow.set_tags({"champion": champion})

    logger.info("Готово за %.0f c. Модель: %s", time.time() - t0, cfg.MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials", type=int, default=30, help="число trials Optuna для LightGBM (≥20)")
    ap.add_argument("--data", default=None, help="путь к CSV (по умолчанию data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv)")
    a = ap.parse_args()
    main(a.n_trials, a.data)
