#!/usr/bin/env python3
"""Tests for signals/self_consistency.py — pure python, no models."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.self_consistency import (  # noqa: E402
    claim_consistency,
    f1_overlap,
    sample_agreement_matrix,
    score_claims_consistency,
)


def test_identical_is_one_disjoint_is_zero():
    assert f1_overlap("a red cat", "a red cat") == 1.0
    assert f1_overlap("a red cat", "xyzzy qqq") == 0.0
    assert f1_overlap("", "a cat") == 0.0


def test_stable_claim_scores_high():
    out = claim_consistency("a red cat on the mat",
                            ["a red cat is on the mat", "there is a red cat on a mat",
                             "a red cat on the mat!"])
    assert out["agreement"] > 0.7, out
    assert out["n_samples"] == 3


def test_hallucinated_claim_scores_low():
    out = claim_consistency("a blue elephant",
                            ["a red cat is on the mat", "there is a red cat on a mat"])
    assert out["agreement"] < 0.3, out
    assert out["uncertainty"] > 0.7


def test_empty_samples_is_nan():
    out = claim_consistency("a cat", [])
    assert out["agreement"] != out["agreement"] and out["n_samples"] == 0


def test_matrix_is_symmetric_with_unit_diagonal():
    s = ["a cat", "a cat sat", "a dog"]
    m = sample_agreement_matrix(s)
    assert m[0][0] == 1.0 and abs(m[0][1] - m[1][0]) < 1e-9


def test_sampler_failure_degrades_to_nan():
    def bad(n, t):
        raise RuntimeError("no gpu")

    outs = score_claims_consistency(["a cat"], bad)
    assert outs[0]["agreement"] != outs[0]["agreement"]


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
