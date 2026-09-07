#!/usr/bin/env python3
"""Tests for data/pope_loader.py — synthetic fixture, no downloads."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pope_loader import load_pope, three_way_summary  # noqa: E402


def _fixture() -> Path:
    entries = []
    for i, cat in enumerate(["random", "popular", "adversarial"]):
        for j in range(6):
            entries.append({"question_id": f"{cat}_{j}",
                            "image": f"COCO_val2014_{i}{j}.jpg",
                            "text": f"Is there a thing{i} in the image?",
                            "label": "yes" if j % 2 == 0 else "no",
                            "category": cat})
    entries.append({"question_id": "bad", "text": "no image here"})
    p = Path(tempfile.mkdtemp()) / "pope_mini.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return p


def test_loads_and_skips_bad_entries():
    es = load_pope(_fixture())
    assert len(es) == 18, len(es)
    assert all(e["question"] and e["image"] for e in es)


def test_three_way_counts_and_yes_rate():
    es = load_pope(_fixture())
    rep = three_way_summary(es)
    assert rep["category_counts"] == {"random": 6, "popular": 6, "adversarial": 6}
    assert abs(rep["overall_yes_rate"] - 0.5) < 1e-9
    assert all(v["yes_rate"] == 0.5 for v in
               [rep["random"], rep["popular"], rep["adversarial"]])


def test_key_variants_tolerated():
    p = Path(tempfile.mkdtemp()) / "v.json"
    p.write_text(json.dumps([{"id": 1, "image_id": "x.jpg",
                              "question": "Is there a cat?", "answer": "Yes",
                              "split": "Adversarial Split"}]), encoding="utf-8")
    es = load_pope(p)
    assert es[0]["label"] == 1 and es[0]["category"] == "adversarial"


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
