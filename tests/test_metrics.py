import numpy as np

from src.metrics import budget_threshold, evaluate, lift_at_k, pick_f1_threshold, recall_at_k


def test_recall_at_k_perfect_and_random():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert recall_at_k(y, np.linspace(1, 0, 10), 0.2) == 1.0     # идеальный скор: оба ухода в топ-20%
    assert recall_at_k(y, np.linspace(0, 1, 10), 0.2) == 0.0     # худший скор
    assert lift_at_k(y, np.linspace(1, 0, 10), 0.2) == 5.0       # 100% / 20% базовая доля


def test_thresholds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    thr = budget_threshold(s, 0.2)
    assert (s >= thr).sum() >= 100
    assert 0 < pick_f1_threshold(y, s) < 1
    m = evaluate(y, s, 0.5)
    assert m["tp"] + m["fn"] == y.sum() and 0 <= m["pr_auc"] <= 1 and "recall_at_20" in m
