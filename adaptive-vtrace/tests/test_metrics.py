#!/usr/bin/env python3
"""Tests for eval/metrics.py — synthetic labels, CPU only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (  # noqa: E402
    auprc,
    auroc,
    ece,
    f1_at_threshold,
    mcnemar,
    net_gain,
    paired_bootstrap_delta,
    risk_coverage_curve,
)


def test_net_gain_abstention_counts_both_sides():
    # 10 fixed hallucinations, but 4 correct claims abstained-away => net 6.
    assert net_gain(10, 4) == 6
    assert net_gain(0, 0) == 0


def test_auroc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) > 0.99
    rng = np.random.default_rng(0)
    assert abs(auroc(rng.integers(0, 2, 500), rng.random(500)) - 0.5) < 0.1


def test_ece_calibrated_is_low():
    rng = np.random.default_rng(1)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)
    assert ece(y, p) < 0.05, ece(y, p)
    assert ece(y, np.zeros_like(p)) > 0.2  # wildly miscalibrated


def test_risk_coverage_monotone_coverage():
    y = np.array([0, 1, 0, 1, 1])
    r = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    curve = risk_coverage_curve(y, r, grid=5)
    covs = [c for c, _ in curve]
    assert covs == sorted(covs, reverse=True), covs  # threshold 1->0: coverage falls
    assert curve[0][0] == 1.0  # threshold 1 covers everything


def test_paired_bootstrap_covers_true_delta():
    rng = np.random.default_rng(2)
    a = (rng.random(400) < 0.70).astype(int)
    b = (rng.random(400) < 0.75).astype(int)
    d, lo, hi = paired_bootstrap_delta(a, b, n_boot=300, seed=0)
    assert lo <= d <= hi
    assert -0.05 < d < 0.15, d


def test_auprc_and_f1():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auprc(y, s) > 0.9
    m = f1_at_threshold(y, s, 0.5)
    assert m["f1"] == 1.0 and m["accuracy"] == 1.0
    m2 = f1_at_threshold(y, np.array([0.9, 0.9, 0.1, 0.1]), 0.5)
    assert m2["f1"] == 0.0


def test_mcnemar_ignores_agreements():
    a = np.array([1, 1, 0, 0, 1, 0])
    b = np.array([1, 0, 1, 0, 1, 0])
    out = mcnemar(a, b)
    assert out["b01"] == 1 and out["b10"] == 1 and out["p"] == 1.0
    out2 = mcnemar(np.zeros(10, int), np.ones(10, int))
    assert out2["b01"] == 10 and out2["p"] < 0.01


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
