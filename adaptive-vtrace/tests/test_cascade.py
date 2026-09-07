#!/usr/bin/env python3
"""Tests for repair/cascade.py — the cost-accuracy proof, synthetic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from repair.cascade import cascade_decide, decisiveness, pareto_table  # noqa: E402

rng = np.random.default_rng(0)


def test_decisive_clip_claim_never_escalates():
    d = cascade_decide(0.05, float("nan"))
    assert d["stage"] == "stage0" and d["cost"] == 0.0 and not d["escalated"]


def test_tossup_escalates_to_full():
    d = cascade_decide(0.51, 0.9)
    assert d["escalated"] and d["stage"] in ("stage1", "stage2")
    assert d["cost"] > 0


def test_nan_clip_always_escalates():
    d = cascade_decide(float("nan"), 0.8)
    assert d["escalated"]


def test_cascade_matches_full_accuracy_at_fraction_of_cost():
    # World where CLIP is right and decisive on 80% of claims, toss-up on 20%
    # where the full stack rescues them.
    n = 500
    y = rng.integers(0, 2, n)
    clip = np.where(rng.random(n) < 0.8,
                    np.where(y == 1, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n)),
                    0.5 + rng.normal(0, 0.05, n))
    full = np.where(y == 1, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n))
    rows = pareto_table(y, clip, full, taus=(0.6,))
    r = rows[0]
    full_acc = float((((full >= 0.5).astype(int)) == y).mean())
    assert abs(r["accuracy"] - full_acc) < 0.03, (r["accuracy"], full_acc)
    assert r["mean_cost"] < 0.35 * r["full_cost"], r  # <35% of full-stack cost


def test_strict_tau_costs_more():
    n = 300
    y = rng.integers(0, 2, n)
    clip = rng.random(n)
    full = np.where(y == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    rows = pareto_table(y, clip, full, taus=(0.3, 0.9))
    assert rows[1]["mean_cost"] >= rows[0]["mean_cost"], rows


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
