"""Графики для отчёта (matplotlib, без GUI-бэкенда)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402


def plot_pr_curves(y_true, scores: dict[str, np.ndarray], path: Path) -> None:
    from sklearn.metrics import average_precision_score
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(y_true, s):.3f})")
    ax.axhline(np.mean(y_true), color="gray", ls="--", lw=1, label=f"доля ушедших = {np.mean(y_true):.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall кривые (hold-out)", ylim=(0, 1.02))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(tn: int, fp: int, fn: int, tp: int, path: Path, title: str) -> None:
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["остался", "ушёл"], yticklabels=["остался", "ушёл"],
           xlabel="Предсказание", ylabel="Факт", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_gain_curves(y_true, scores: dict[str, np.ndarray], k: float, path: Path) -> None:
    """Кумулятивный охват уходов в зависимости от доли обзвонённых клиентов."""
    y = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(1, len(y) + 1) / len(y)
    for name, s in scores.items():
        order = np.argsort(-np.asarray(s), kind="mergesort")
        ax.plot(x, np.cumsum(y[order]) / y.sum(), label=name)
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="случайный обзвон")
    ax.axvline(k, color="red", ls=":", lw=1, label=f"бюджет {int(k * 100)}%")
    ax.set(xlabel="Доля обзвонённых клиентов (по убыванию риска)", ylabel="Доля пойманных уходов (Recall)",
           title="Cumulative gain: сколько уходов ловим при заданном бюджете")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
