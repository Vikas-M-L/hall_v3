#!/usr/bin/env python3
"""Tests for fusion/diagnose.py — pure rules, instant."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusion.diagnose import MECHANISMS, diagnose  # noqa: E402


def test_low_risk_is_m0_accept():
    d = diagnose(0.15, 0.2, 0.2)
    assert d["mechanism"] == "M0" and d["repair"] == "accept" and d["reasons"]


def test_count_mismatch_is_m2_with_recount():
    d = diagnose(0.55, 0.5, 0.5, count_match=False)
    assert d["mechanism"] == "M2" and "recount" in d["repair"]


def test_vlm_refute_weak_grounding_is_m2():
    d = diagnose(0.7, 0.8, 0.75, vlm_supported=False)
    assert d["mechanism"] == "M2"


def test_vlm_refute_strong_grounding_is_m4():
    d = diagnose(0.55, 0.2, 0.3, claim_type="spatial", vlm_supported=False)
    assert d["mechanism"] == "M4", d


def test_high_uncertainty_is_m3_abstain():
    d = diagnose(0.55, 0.5, 0.5, uncertainty=0.8)
    assert d["mechanism"] == "M3" and d["repair"] == "abstain or hedge"


def test_clip_split_on_relation_is_m4():
    d = diagnose(0.5, 0.2, 0.6, claim_type="relation")
    assert d["mechanism"] == "M4", d


def test_plain_unsupported_defaults_m1():
    d = diagnose(0.75, 0.7, 0.7)
    assert d["mechanism"] == "M1" and d["cost"] == "cheap"


def test_nan_is_unknown_not_m0():
    d = diagnose(float("nan"), float("nan"), float("nan"))
    assert d["mechanism"] == "M?"


def test_taxonomy_has_m4():
    assert MECHANISMS == ("M0", "M1", "M2", "M3", "M4")


def test_refutation_is_not_ignored_when_clip_risk_is_low():
    result = diagnose(0.1, 0.1, 0.1, vlm_supported=False)
    assert result["mechanism"] != "M0"
    assert "VLM directly refuted the claim" in result["reasons"]


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
