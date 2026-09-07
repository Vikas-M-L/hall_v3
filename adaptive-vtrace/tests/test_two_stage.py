#!/usr/bin/env python3
"""Tests for fusion/two_stage.py — synthetic taxonomy-shaped data, CPU only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusion.two_stage import FEATURES, TwoStageClassifier  # noqa: E402

rng = np.random.default_rng(0)


def synth(n: int = 600):
    """Rows follow the taxonomy conjunctions: M1 = conf high + prior high +
    vig low; M2 = conf low-mid + grounding weak; M3 = conf high + prior low +
    uncertainty high; else M0 (not hallucinated)."""
    X = rng.random((n, len(FEATURES)))
    yh = np.zeros(n, dtype=int)
    ym = np.full(n, "M0", dtype=object)
    for i in range(n):
        c, p, v, _, g, u = (X[i, 0], X[i, 1], X[i, 2], X[i, 3], X[i, 4], X[i, 5])
        if c > 0.7 and p > 0.7 and v < 0.3:
            yh[i], ym[i] = 1, "M1"
        elif c < 0.5 and g < 0.3:
            yh[i], ym[i] = 1, "M2"
        elif c > 0.7 and p < 0.3 and u > 0.7:
            yh[i], ym[i] = 1, "M3"
    return X, yh, ym


def test_gbm_learns_conjunctions():
    X, yh, ym = synth(900)
    clf = TwoStageClassifier("xgboost", "xgboost").fit(X, yh, ym)
    ph, pm = clf.predict(X)
    assert (ph == yh).mean() > 0.90, (ph == yh).mean()
    m = yh == 1
    assert (pm[m] == ym[m]).mean() > 0.80, (pm[m] == ym[m]).mean()


def test_m0_noop_for_clean_claims():
    X, yh, ym = synth(300)
    clf = TwoStageClassifier("xgboost", "logreg").fit(X, yh, ym)
    ph, pm = clf.predict(X[yh == 0][:50])
    # Clean claims must overwhelmingly route to decline-to-repair.
    assert (pm == "M0").mean() >= 0.95, pm


def test_oof_column_runs():
    X, yh, ym = synth(400)
    clf = TwoStageClassifier("logreg", "logreg").fit(X, yh, ym, oof_column=6)
    ph, pm = clf.predict(X[:50])
    assert len(ph) == 50 and len(pm) == 50


def test_empty_hallucinated_raises():
    X = rng.random((20, len(FEATURES)))
    try:
        TwoStageClassifier("logreg", "logreg").fit(X, np.zeros(20, dtype=int),
                                                   np.full(20, "M0", dtype=object))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


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
