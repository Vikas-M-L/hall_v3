#!/usr/bin/env python3
"""Tests for models/owl_wrapper.py — parser + verdict logic only (no weights)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.owl_wrapper import count_verdict, parse_count  # noqa: E402


def test_parses_counts_and_labels():
    assert parse_count("2 dogs in the room.") == (2, "dogs")
    assert parse_count("There are three red cars.") == (3, "red cars")
    assert parse_count("A dog.") == (None, None)
    assert parse_count("The square is red.") == (None, None)


def test_verdict_logic():
    assert count_verdict(2, "dogs", {"count": 2})["match"] is True
    assert count_verdict(2, "dogs", {"count": 0})["match"] is False
    assert "no 'dogs'" in count_verdict(2, "dogs", {"count": 0})["reason"]
    assert count_verdict(2, "dogs", {"count": 1})["match"] is False


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
