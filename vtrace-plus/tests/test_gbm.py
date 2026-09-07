#!/usr/bin/env python3
"""Tests for fusion/gbm_diagnoser.py — small/fast training, no live wiring."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusion.gbm_diagnoser import (  # noqa: E402
    FEATURES,
    ROUTING,
    diagnose_claim,
    generate_synthetic,
    load_bundle,
    save_bundle,
    train,
)


def test_synthetic_covers_all_mechanisms():
    df = generate_synthetic(20, seed=0)
    assert set(df["diagnosis"]) == {"M0", "M1", "M2", "M3", "M4"}
    assert list(df.columns)[:-1] == FEATURES


def test_train_reports_and_confuses():
    b = train(generate_synthetic(40, seed=1), n_estimators=30)
    assert b["n_train"] + b["n_test"] == 200
    assert b["report"]["accuracy"] > 0.85, b["report"]["accuracy"]
    assert b["confusion"].shape == (5, 5)


def test_routing_covers_every_mechanism():
    assert set(ROUTING) == {"M0", "M1", "M2", "M3", "M4"}
    assert ROUTING["M2"]["action"] == "RECOUNT"


def test_save_load_roundtrip_and_diagnose():
    b = train(generate_synthetic(40, seed=2), n_estimators=30)
    p = Path(tempfile.mkdtemp()) / "m.joblib"
    b2 = load_bundle(save_bundle(b, p))
    out = diagnose_claim(b2, {"clip_similarity": 0.32, "region_evidence": 0.25,
                              "owl_count": 2, "claimed_count": 3,
                              "gemini_consistency": 0.40, "gemini_confidence": 0.35})
    assert out["mechanism"] in ("M0", "M1", "M2", "M3", "M4")
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6
    assert out["action"] == ROUTING[out["mechanism"]]["action"]


def test_feature_mismatch_refuses():
    import joblib

    p = Path(tempfile.mkdtemp()) / "bad.joblib"
    joblib.dump({"model": None, "features": ["wrong"]}, p)
    try:
        load_bundle(p)
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
