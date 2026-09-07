"""Eval metrics for the headline experiment (plan sections 7-8).

Conventions (preregistered, plan section 7):
  * Net gain = hallucinations removed MINUS correct claims removed/corrupted,
    with abstention counting as removal on BOTH sides (else abstain-everything
    scores 100% fixed).
  * Headline deltas are claimed over detect-then-best-uniform and over the
    repair-outcome policy — not over no-repair.
  * Paired design: same claims through every arm; adjudicate disagreements;
    McNemar / paired bootstrap for the small deltas.
"""
from __future__ import annotations

import numpy as np


def net_gain(fixed_hall: int, broken_correct: int) -> int:
    return int(fixed_hall) - int(broken_correct)


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(y_true), np.asarray(scores)))


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(np.asarray(y_true), np.asarray(scores)))


def f1_at_threshold(y_true: np.ndarray, scores: np.ndarray,
                    threshold: float = 0.5) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score

    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    return {"f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "accuracy": float((pred == y).mean()),
            "threshold": threshold}


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err, tot = 0.0, 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        err += m.sum() * abs(y[m].mean() - p[m].mean())
        tot += m.sum()
    return float(err / tot) if tot else float("nan")


def risk_coverage_curve(y_true: np.ndarray, risks: np.ndarray,
                        grid: int = 20) -> list[tuple[float, float]]:
    """(coverage, error-rate-on-covered) sweeping a risk threshold downward."""
    y = np.asarray(y_true, dtype=int)
    r = np.asarray(risks, dtype=float)
    out = []
    for t in np.linspace(1.0, 0.0, grid):
        m = r <= t
        cov = float(m.mean())
        err = float(y[m].mean()) if m.sum() else float("nan")
        out.append((cov, err))
    return out


def paired_bootstrap_delta(a_correct: np.ndarray, b_correct: np.ndarray,
                           n_boot: int = 1000, seed: int = 0
                           ) -> tuple[float, float, float]:
    """Mean(b - a) with 95% paired-bootstrap CI. Same claims, both arms."""
    a = np.asarray(a_correct, dtype=float)
    b = np.asarray(b_correct, dtype=float)
    assert a.shape == b.shape and a.size > 0
    rng = np.random.default_rng(seed)
    delta = float((b - a).mean())
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, a.size, a.size)
        boots.append(float((b[idx] - a[idx]).mean()))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return delta, lo, hi


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """McNemar on claims where the two arms disagree. Returns b01, b10, p."""
    from scipy.stats import binomtest

    a = np.asarray(a_correct, dtype=int)
    b = np.asarray(b_correct, dtype=int)
    b01 = int(((a == 0) & (b == 1)).sum())  # b right where a wrong
    b10 = int(((a == 1) & (b == 0)).sum())
    if b01 + b10 == 0:
        return {"b01": 0, "b10": 0, "p": 1.0}
    res = binomtest(min(b01, b10), b01 + b10, 0.5)
    return {"b01": b01, "b10": b10, "p": float(res.pvalue)}
