"""EDA: графики -> reports/eda/*.png, выводы -> reports/eda_summary.md.   Запуск: python -m src.eda"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config as cfg  # noqa: E402
from src.data import load_raw, prepare_xy, validate_raw  # noqa: E402
from src.features import ChurnFeatureEngineer  # noqa: E402

logger = logging.getLogger("eda")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
sns.set_theme(style="whitegrid")
NUM = ["tenure", "MonthlyCharges", "TotalCharges"]


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(cfg.EDA_DIR / name, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg.EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    validate_raw(df)
    X, y = prepare_xy(df)
    d = X.assign(Churn=y)
    lines: list[str] = ["# EDA summary (генерируется автоматически: `python -m src.eda`)\n"]

    # 1. target
    rate = y.mean()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.countplot(x=df[cfg.TARGET_COL], ax=ax, order=["No", "Yes"])
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    ax.set_title(f"Распределение Churn (доля ушедших = {rate:.1%})")
    _save(fig, "01_target.png")
    lines.append(f"## 1. Target\nУшли {int(y.sum())} из {len(y)} ({rate:.1%}); дисбаланс ≈ 1:{(1 - rate) / rate:.1f}. "
                 "Accuracy вводит в заблуждение (константа «остался» даёт "
                 f"{1 - rate:.1%}) → основная метрика PR-AUC, StratifiedKFold.\n")

    # 2. пропуски (включая «пустые строки из пробелов»)
    nulls = df.isna().sum()
    blanks = df.apply(lambda s: s.astype(str).str.strip().eq("").sum() if s.dtype == object or str(s.dtype).startswith("str") else 0)
    miss = pd.DataFrame({"NaN": nulls, "пробелы": blanks})
    miss = miss[(miss > 0).any(axis=1)]
    fig, ax = plt.subplots(figsize=(6, 3))
    if len(miss):
        miss.plot.barh(ax=ax)
    else:
        ax.text(0.5, 0.5, "пропусков нет", ha="center")
    ax.set_title("Пропуски по колонкам (NaN и «пустые» строки)")
    _save(fig, "02_missing.png")
    n_blank = int(blanks.get("TotalCharges", 0))
    only_t0 = bool((df.loc[pd.to_numeric(df["TotalCharges"], errors="coerce").isna(), "tenure"] == 0).all())
    lines.append(f"## 2. Пропуски\nNaN нет. `TotalCharges`: {n_blank} значений — пробелы (из-за них колонка читается как object). "
                 f"Все они у клиентов с tenure = 0: {only_t0} → заполняем MonthlyCharges × tenure = 0.\n")

    # 3. числовые: гистограммы + boxplot
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for j, c in enumerate(NUM):
        sns.histplot(data=d, x=c, hue="Churn", kde=True, stat="density", common_norm=False, ax=axes[0, j])
        sns.boxplot(data=d, x="Churn", y=c, ax=axes[1, j])
    _save(fig, "03_numeric.png")
    skew = X[NUM].skew()
    iqr_out = {c: int(((X[c] < X[c].quantile(.25) - 1.5 * (X[c].quantile(.75) - X[c].quantile(.25))) |
                       (X[c] > X[c].quantile(.75) + 1.5 * (X[c].quantile(.75) - X[c].quantile(.25)))).sum()) for c in NUM}
    lines.append("## 3. Числовые признаки\n| признак | skew | выбросов по IQR |\n|---|---|---|\n" +
                 "\n".join(f"| {c} | {skew[c]:.2f} | {iqr_out[c]} |" for c in NUM) +
                 "\n\nМедиана tenure: ушедшие "
                 f"{d.loc[d.Churn == 1, 'tenure'].median():.0f} мес. vs оставшиеся {d.loc[d.Churn == 0, 'tenure'].median():.0f} мес. "
                 "Выбросы в смысле ошибок данных отсутствуют; TotalCharges скошен вправо → верхний хвост режется (winsorizing), "
                 "для деревьев распределение не трансформируется.\n")

    # 4. churn rate по категориям
    cats = [c for c in cfg.CATEGORICAL_LEVELS] + ["SeniorCitizen"]
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    rows = []
    for ax, c in zip(axes.ravel(), cats):
        g = d.groupby(c)["Churn"].agg(["mean", "size"]).reset_index()
        sns.barplot(data=g, x=c, y="mean", ax=ax, color="steelblue")
        ax.axhline(rate, color="red", ls="--", lw=1)
        ax.set_ylabel("churn rate"); ax.set_title(c); ax.tick_params(axis="x", rotation=20)
        rows += [(c, str(r[c]), r["mean"], int(r["size"])) for _, r in g.iterrows()]
    for ax in axes.ravel()[len(cats):]:
        ax.axis("off")
    _save(fig, "04_churn_by_category.png")
    cr = pd.DataFrame(rows, columns=["feature", "value", "churn_rate", "n"]).query("n >= 100").sort_values("churn_rate")
    top = cr.tail(5)[::-1]; low = cr.head(5)
    lines.append("## 4. Churn rate по категориям (группы n ≥ 100)\n**Самые рискованные:**\n" +
                 "\n".join(f"- {r.feature} = {r.value}: {r.churn_rate:.1%} (n={r.n})" for r in top.itertuples()) +
                 "\n\n**Самые надёжные:**\n" +
                 "\n".join(f"- {r.feature} = {r.value}: {r.churn_rate:.1%} (n={r.n})" for r in low.itertuples()) + "\n")

    # 5. корреляции (на признаках после FE)
    F = ChurnFeatureEngineer().fit(X).transform(X)
    corr = F.assign(Churn=y).corr()
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": .6}, xticklabels=True, yticklabels=True)
    ax.set_title("Корреляции признаков после FE (Pearson)")
    _save(fig, "05_correlation.png")
    tgt = corr["Churn"].drop("Churn")
    tgt_top = tgt.reindex(tgt.abs().sort_values(ascending=False).index)[:8]
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(F.columns) for b in F.columns[i + 1:] if abs(corr.loc[a, b]) > 0.8]
    lines.append("## 5. Корреляции\n**Сильнее всего связаны с Churn:**\n" +
                 "\n".join(f"- {k}: {v:+.2f}" for k, v in tgt_top.items()) +
                 "\n\n**Мультиколлинеарные пары (|r| > 0.8):**\n" +
                 ("\n".join(f"- {a} ~ {b}: {r:+.2f}" for a, b, r in pairs) or "- нет") +
                 "\n\nЛинейной модели мультиколлинеарность вредит (поэтому L2-регуляризация + StandardScaler); "
                 "для бустинга/леса — не проблема.\n")

    lines.append("## Вывод\n- Информативны: тип контракта, tenure, тип интернета, способ оплаты, наличие техподдержки/онлайн-безопасности.\n"
                 "- Слабо информативны: gender и ряд флагов (см. churn rate ≈ среднему).\n"
                 "- Требуют преобразования: TotalCharges (пробелы, скошенность), категории с «No internet service» (сводятся к 0), "
                 "tenure (нелинейность → is_new_customer), цена/длительность (→ avg_charge_per_month).\n")
    (cfg.REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("EDA готов: %s", cfg.EDA_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
