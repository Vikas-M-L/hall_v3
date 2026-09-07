#!/usr/bin/env python3
"""Tests for finetune/build_lora_data.py — pure python, no models, no GPU."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.build_lora_data import build_jsonl, row_to_record  # noqa: E402


def _row(mech: str, i: int = 0, **kw):
    d = {"image": f"img{i}.jpg", "claim": "a cat", "mechanism": mech,
         "provenance": "synthetic", "evidence": 0.8, "similarity": 0.7}
    d.update(kw)
    return d


def test_record_format():
    r = row_to_record(**_row("M2"))
    assert r["label"] == "M2" and "M0 M1 M2 M3 M4" in r["prompt"]
    assert "evidence=0.80" in r["prompt"] and r["provenance"] == "synthetic"


def test_bad_mechanism_raises():
    try:
        row_to_record(**_row("M9"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_nan_and_missing_format_cleanly():
    r = row_to_record(**_row("M0", evidence=float("nan")))
    assert "evidence=n/a" in r["prompt"] and "uncertainty=n/a" in r["prompt"]


def test_balances_to_minority_class():
    rows = [_row("M0", i) for i in range(6)] + [_row("M1", i) for i in range(2)]
    n = build_jsonl(rows, Path(tempfile.mkdtemp()) / "train.jsonl")
    assert n == 4, n


def test_jsonl_is_parseable():
    p = Path(tempfile.mkdtemp()) / "t.jsonl"
    build_jsonl([_row("M0"), _row("M1")], p)
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").strip().split("\n")]
    assert len(recs) == 2 and all(set(r) >= {"image", "prompt", "label"} for r in recs)


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
