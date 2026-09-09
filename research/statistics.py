"""Paired image-cluster statistics. Scores are not calibrated probabilities by fiat."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)


def validate(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("aligned nonempty vectors required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("binary labels and finite unit scores required")
    return y.astype(int), p


def reliability(y, p, bins=10):
    y, p = validate(y, p)
    if bins < 1:
        raise ValueError("positive bin count required")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return [{"bin": b, "n": int((idx == b).sum()),
             "score": float(p[idx == b].mean()), "frequency": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def metrics(y, p, threshold=.5):
    y, p = validate(y, p)
    pred = p >= threshold
    rel = reliability(y, p)
    return {"n": len(y), "positive": int(y.sum()), "negative": int((1-y).sum()),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "auprc": float(average_precision_score(y, p)) if y.sum() else None,
            "brier": float(brier_score_loss(y, p)),
            "ece": sum(r["n"] * abs(r["score"]-r["frequency"]) for r in rel)/len(y),
            "threshold": threshold}


def selective(y, p, verdicts):
    y, p = validate(y, p)
    if len(verdicts) != len(y) or any(v not in ("supported", "contradicted", "unresolved") for v in verdicts):
        raise ValueError("aligned three-state verdicts required")
    keep = np.array([v != "unresolved" for v in verdicts])
    correct = np.array([v == "contradicted" for v in verdicts]) == y
    return {"coverage": float(keep.mean()), "decided": int(keep.sum()),
            "decided_correct": int((keep & correct).sum()),
            "selective_accuracy": float(correct[keep].mean()) if keep.any() else None,
            "unresolved": int((~keep).sum()),
            "unconditional_correct_fraction": float((keep & correct).mean())}


def risk_coverage(y, p):
    y, p = validate(y, p)
    # Include all equal-confidence cases together: no favorable tie ordering.
    confidence = np.abs(p-.5)
    return [{"coverage": float((confidence >= t).mean()),
             "selective_risk": float(((p >= .5) != y)[confidence >= t].mean()),
             "confidence_threshold": float(.5+t)}
            for t in sorted(set(confidence), reverse=True)]


def cluster_bootstrap(groups, statistic, repeats=1000, seed=42):
    if repeats < 1 or not groups:
        raise ValueError("positive repeats and nonempty groups required")
    groups = np.asarray(groups)
    unique = np.unique(groups)
    blocks = [np.flatnonzero(groups == g) for g in unique]
    rng, values = np.random.default_rng(seed), []
    for _ in range(repeats):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        v = statistic(idx)
        if v is not None and np.isfinite(v):
            values.append(float(v))
    return {"ci95": np.quantile(values, [.025, .975]).tolist() if values else None,
            "valid_resamples": len(values), "requested_resamples": repeats,
            "clusters": len(unique), "seed": seed, "method": "image-cluster percentile bootstrap"}


def paired_comparison(y, a, b, groups, repeats=1000, seed=42):
    y, a = validate(y, a)
    _, b = validate(y, b)
    diff = ((b >= .5) == y).astype(float) - ((a >= .5) == y)
    ci = cluster_bootstrap(groups, lambda idx: diff[idx].mean(), repeats, seed)
    # Randomization swaps complete image blocks, preserving within-image dependence.
    groups_arr = np.asarray(groups)
    sums = np.array([diff[groups_arr == g].sum() for g in np.unique(groups_arr)])
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    extreme = sum(abs((sums*rng.choice([-1, 1], len(sums))).sum()/len(diff)) >= obs-1e-12
                  for _ in range(repeats))
    return {"delta_accuracy_b_minus_a": float(diff.mean()), **ci,
            "paired_cluster_randomization_p": (extreme+1)/(repeats+1),
            "interpretation": "exploratory; no correction for multiple comparisons"}


def repair_metrics(rows):
    """Human/exact gold outcomes, not model-judge verdicts. Abstention is not repair."""
    if not rows:
        raise ValueError("no adjudicated outcomes")
    for r in rows:
        if any(type(r.get(k)) is not bool for k in
               ("original_hallucinated", "original_correct", "abstained")):
            raise ValueError("explicit independent original labels required")
        if not r["abstained"] and any(type(r.get(k)) is not bool for k in
                                      ("final_hallucinated", "final_correct")):
            raise ValueError("independent final labels required")
    n = len(rows)
    h = sum(r["original_hallucinated"] for r in rows)
    good = sum(r["original_correct"] for r in rows)
    fixed = sum(r["original_hallucinated"] and not r["abstained"] and
                not r["final_hallucinated"] and r["final_correct"] for r in rows)
    preserved = sum(r["original_correct"] and not r["abstained"] and r["final_correct"] for r in rows)
    broken = good-preserved
    final_h = sum(not r["abstained"] and r["final_hallucinated"] for r in rows)
    return {"n": n, "original_hallucinations": h, "corrected_answers": fixed,
            "correction_success_rate": fixed/h if h else None,
            "answer_preservation": preserved/good if good else None,
            "overcorrection_or_removal": broken/good if good else None,
            "final_answer_accuracy": sum(not r["abstained"] and r["final_correct"] for r in rows)/n,
            "abstention_rate": sum(r["abstained"] for r in rows)/n,
            "hallucination_reduction_including_abstention": (h-final_h)/h if h else None,
            "net_corrected_minus_correct_lost": fixed-broken}
