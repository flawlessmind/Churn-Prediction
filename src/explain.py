"""SHAP-интерпретация чемпиона: global bar/beeswarm, 3 waterfall, топ-5 факторов «вверх»/«вниз».

Запуск:  python -m src.explain     (после python -m src.train)
"""
from __future__ import annotations

import json
import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src import config as cfg  # noqa: E402

logger = logging.getLogger("explain")


def _positive_class_shap(explainer: shap.TreeExplainer, X: pd.DataFrame):
    """Единый формат (values[n, f], base_value) для класса «ушёл» независимо от версии shap/модели."""
    sv = explainer.shap_values(X)
    ev = explainer.expected_value
    if isinstance(sv, list):                 # старый API: [class0, class1]
        sv, ev = sv[1], np.atleast_1d(ev)[-1]
    elif np.ndim(sv) == 3:                   # RandomForest в новых версиях: (n, f, n_classes)
        sv, ev = sv[:, :, 1], np.atleast_1d(ev)[-1]
    else:
        ev = np.atleast_1d(ev)[-1]
    return np.asarray(sv), float(ev)


def _waterfall(values, base, X_row: pd.Series, path, title: str) -> None:
    exp = shap.Explanation(values=values, base_values=base, data=X_row.values, feature_names=list(X_row.index))
    plt.figure()
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    if not cfg.MODEL_PATH.exists() or not cfg.HOLDOUT_CSV.exists():
        raise FileNotFoundError("Нет модели или hold-out. Сначала выполните: python -m src.train")
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(cfg.MODEL_PATH)
    meta = json.loads(cfg.METADATA_PATH.read_text(encoding="utf-8"))
    hold = pd.read_csv(cfg.HOLDOUT_CSV)
    X_raw, y = hold[cfg.RAW_FEATURES], hold[cfg.TARGET_COL]
    X = pipe.named_steps["fe"].transform(X_raw)          # те же признаки, что видит модель
    model = pipe.named_steps["model"]

    explainer = shap.TreeExplainer(model)
    sv, base = _positive_class_shap(explainer, X)
    proba = pipe.predict_proba(X_raw)[:, 1]

    # --- global
    plt.figure()
    shap.summary_plot(sv, X, plot_type="bar", max_display=15, show=False)
    plt.tight_layout(); plt.savefig(cfg.FIGURES_DIR / "shap_bar.png", dpi=130, bbox_inches="tight"); plt.close("all")
    plt.figure()
    shap.summary_plot(sv, X, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(cfg.FIGURES_DIR / "shap_beeswarm.png", dpi=130, bbox_inches="tight"); plt.close("all")

    # --- local: самый рискованный, пограничный (около порога), самый «надёжный»
    thr = meta["thresholds"]["f1_opt"]
    picks = {
        "highest_risk": int(np.argmax(proba)),
        "near_threshold": int(np.argmin(np.abs(proba - thr))),
        "lowest_risk": int(np.argmin(proba)),
    }
    local = {}
    for tag, i in picks.items():
        _waterfall(sv[i], base, X.iloc[i], cfg.FIGURES_DIR / f"shap_waterfall_{tag}.png",
                   f"{tag}: P(churn)={proba[i]:.2f}, факт={'ушёл' if y.iloc[i] else 'остался'}")
        contrib = pd.Series(sv[i], index=X.columns)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:5]
        local[tag] = {"row": i, "churn_probability": round(float(proba[i]), 4),
                      "top_contributions": {k: round(float(v), 4) for k, v in top.items()}}

    # --- топ-5 факторов «вверх» / «вниз»: знак = корреляция значения признака и его SHAP-вклада
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    direction = {}
    for j, c in enumerate(X.columns):
        direction[c] = np.nan if X[c].nunique() < 2 else float(np.corrcoef(X[c].values, sv[:, j])[0, 1])
    d = pd.Series(direction)
    up = imp[d[d > 0].index].sort_values(ascending=False).head(5)
    down = imp[d[d < 0].index].sort_values(ascending=False).head(5)

    out = {
        "note": "Вклады — в log-odds (LightGBM) или в вероятности (RandomForest). Направление: "
                "знак корреляции между значением признака и SHAP-вкладом.",
        "global_importance_mean_abs_shap": {k: round(float(v), 4) for k, v in imp.sort_values(ascending=False).head(15).items()},
        "top5_increase_risk": {k: round(float(v), 4) for k, v in up.items()},
        "top5_decrease_risk": {k: round(float(v), 4) for k, v in down.items()},
        "local_explanations": local,
    }
    (cfg.REPORTS_DIR / "shap_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    def fmt(s: pd.Series) -> str:
        return "\n".join(f"{i}. `{k}` — mean|SHAP| = {v:.3f}" for i, (k, v) in enumerate(s.items(), 1))

    (cfg.REPORTS_DIR / "shap_summary.md").write_text(
        "# SHAP: факторы ухода\n\n## Топ-5 факторов, ПОВЫШАЮЩИХ риск ухода\n" + fmt(up) +
        "\n\n## Топ-5 факторов, СНИЖАЮЩИХ риск ухода\n" + fmt(down) +
        "\n\nГрафики: `reports/figures/shap_bar.png`, `shap_beeswarm.png`, `shap_waterfall_*.png`.\n",
        encoding="utf-8")
    logger.info("↑ риск: %s", list(up.index))
    logger.info("↓ риск: %s", list(down.index))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
