"""Метрики качества и бизнес-метрики (Recall@K, Lift@K), выбор порога."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


def _top_k_idx(score: np.ndarray, k: float) -> np.ndarray:
    n_top = max(1, int(np.ceil(k * len(score))))
    return np.argsort(-score, kind="mergesort")[:n_top]


def recall_at_k(y_true, y_score, k: float = 0.2) -> float:
    """Доля ВСЕХ реальных уходов, попавших в топ-k% клиентов по скору (бизнес-метрика обзвона)."""
    y, s = np.asarray(y_true), np.asarray(y_score)
    total = y.sum()
    return float(y[_top_k_idx(s, k)].sum() / total) if total > 0 else 0.0


def precision_at_k(y_true, y_score, k: float = 0.2) -> float:
    """Доля реальных уходов среди обзвонённых."""
    y, s = np.asarray(y_true), np.asarray(y_score)
    return float(y[_top_k_idx(s, k)].mean())


def lift_at_k(y_true, y_score, k: float = 0.2) -> float:
    """Во сколько раз обзвон топ-k% эффективнее случайного обзвона."""
    base = float(np.mean(y_true))
    return precision_at_k(y_true, y_score, k) / base if base > 0 else 0.0


def pick_f1_threshold(y_true, y_score) -> float:
    """Порог, максимизирующий F1 (подбирать нужно на OOF-предсказаниях train, не на hold-out!)."""
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def budget_threshold(y_score, k: float = 0.2) -> float:
    """Минимальный скор среди топ-k% клиентов: порог, соответствующий бюджету обзвона."""
    s = np.asarray(y_score)
    return float(np.sort(s)[::-1][len(_top_k_idx(s, k)) - 1])


def evaluate(y_true, y_score, threshold: float, k: float = 0.2) -> dict:
    """Полный набор метрик: PR-AUC (основная), ROC-AUC, P/R/F1 на пороге, Recall@k, Lift@k, confusion matrix."""
    y, s = np.asarray(y_true), np.asarray(y_score)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    kk = int(round(k * 100))
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        f"recall_at_{kk}": recall_at_k(y, s, k),
        f"precision_at_{kk}": precision_at_k(y, s, k),
        f"lift_at_{kk}": lift_at_k(y, s, k),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
