#!/usr/bin/env python3
"""Tests for models/gemini_wrapper.py — _post is monkeypatched, no network,
no API spend. Verifies parsing, fallbacks, and model-fallback logic."""
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models.gemini_wrapper as gw  # noqa: E402


def _fake_post_factory(pages: dict, fail_first_with: int | None = None):
    calls = {"n": 0}

    def fake(model, parts, **kw):
        calls["n"] += 1
        if fail_first_with and calls["n"] == 1:
            raise urllib.error.HTTPError("url", fail_first_with, "busy", {}, None)
        text = parts[0]["text"] if isinstance(parts[0], dict) and "text" in parts[0] else ""
        return pages.get("default", "")
    fake.calls = calls
    return fake


def test_decompose_parses_clean_json():
    gw_real = gw._post
    gw._post = lambda model, parts, **kw: (
        '[{"claim": "The square is red.", "source": "red"}]')
    try:
        w = gw.GeminiWrapper.__new__(gw.GeminiWrapper)
        w.model = "test"
        out = gw.GeminiWrapper.decompose(w, "A red square.")
        assert len(out) == 1 and out[0]["claim"] == "The square is red.", out
    finally:
        gw._post = gw_real


def test_decompose_unparseable_returns_empty():
    gw_real = gw._post
    gw._post = lambda model, parts, **kw: "Sure! Here is nothing parseable, sorry."
    try:
        w = gw.GeminiWrapper.__new__(gw.GeminiWrapper)
        w.model = "test"
        assert gw.GeminiWrapper.decompose(w, "hi") == []
    finally:
        gw._post = gw_real


def test_overload_falls_back_to_secondary_model():
    seen = []
    orig_post = gw._post

    def flaky(model, parts, **kw):
        seen.append(model)
        if len(seen) == 1:
            raise urllib.error.HTTPError("url", 503, "busy", {}, None)
        return "fallback works"

    gw._post = flaky
    try:
        w = gw.GeminiWrapper.__new__(gw.GeminiWrapper)
        w.model = "primary"
        assert gw.GeminiWrapper._call(w, [{"text": "hi"}]) == "fallback works"
        assert seen == ["primary", gw.FALLBACK_MODEL], seen
    finally:
        gw._post = orig_post


def test_non_retryable_error_raises():
    def bad(model, parts, **kw):
        raise urllib.error.HTTPError("url", 400, "bad request", {}, None)

    orig_post = gw._post
    gw._post = bad
    try:
        w = gw.GeminiWrapper.__new__(gw.GeminiWrapper)
        w.model = "primary"
        try:
            gw.GeminiWrapper._call(w, [{"text": "hi"}])
        except urllib.error.HTTPError as e:
            assert e.code == 400
            return
        raise AssertionError("expected HTTPError")
    finally:
        gw._post = orig_post


def test_missing_key_fails_fast():
    import os

    env_file = Path(__file__).resolve().parent.parent / ".env"
    backup = env_file.with_suffix(".env.bak")
    had = os.environ.pop("GEMINI_API_KEY", None)
    moved = False
    try:
        if env_file.exists():
            env_file.rename(backup)
            moved = True
        try:
            gw._key()
        except RuntimeError:
            return
        raise AssertionError("expected RuntimeError without key")
    finally:
        if moved:
            backup.rename(env_file)
        if had is not None:
            os.environ["GEMINI_API_KEY"] = had


def test_verify_parses_yes_no_and_garbage():
    from PIL import Image as _I

    img = _I.new("RGB", (4, 4), (255, 0, 0))
    w = gw.GeminiWrapper.__new__(gw.GeminiWrapper)
    w.model = "test"
    orig = gw.GeminiWrapper._call
    try:
        gw.GeminiWrapper._call = lambda self, parts, **kw: "YES. The square is red."
        assert gw.GeminiWrapper.verify(w, img, "red square")["supported"] is True
        gw.GeminiWrapper._call = lambda self, parts, **kw: "no, there is no dog here."
        assert gw.GeminiWrapper.verify(w, img, "a dog")["supported"] is False
        gw.GeminiWrapper._call = lambda self, parts, **kw: "Hmm, hard to say really."
        assert gw.GeminiWrapper.verify(w, img, "x")["supported"] is None
    finally:
        gw.GeminiWrapper._call = orig


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
