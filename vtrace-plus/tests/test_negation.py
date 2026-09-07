#!/usr/bin/env python3
"""Tests for fusion/negation.py — pure python, instant."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusion.negation import apply_negation, has_negation, invert_for_negation  # noqa: E402


def test_detects_common_negations():
    for s in ["There is no dog.", "No cat on the mat", "The room is without windows.",
              "Dogs are absent", "It isn't red", "Nothing on the table"]:
        assert has_negation(s), s
    for s in ["A red square.", "A dog playing.", "Notes on the table."]:
        assert not has_negation(s), s


def test_negated_strong_match_is_high_risk():
    # Dog photo, claim "there is no dog": CLIP matches "dog" strongly
    # (match 0.8) => claim is FALSE => risk must be HIGH.
    r, applied = apply_negation("there is no dog", 1 - 0.8, 1 - 0.75)
    assert applied and abs(r - 0.775) < 1e-9, (r, applied)


def test_negated_weak_match_is_low_risk():
    # No dog anywhere, "there is no dog" is TRUE => low risk.
    r, applied = apply_negation("there is no dog", 1 - 0.1, 1 - 0.2)
    assert applied and r < 0.3, (r, applied)


def test_plain_claims_pass_through():
    r, applied = apply_negation("a red square", 0.4, 0.4)
    assert applied is False and r != r


def test_nan_propagates():
    assert invert_for_negation(float("nan")) != invert_for_negation(float("nan"))
    r, applied = apply_negation("no dog", float("nan"), float("nan"))
    assert applied and r != r


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
