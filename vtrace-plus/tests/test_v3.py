#!/usr/bin/env python3
"""Tests for v3 pairwise contradiction + abstention. Logic tests are instant;
model tests reuse cached CLIP-B/32 on CPU (no downloads after first run)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusion.negation import make_counterclaim  # noqa: E402
from fusion.vtrace_fusion import (  # noqa: E402
    FusionResult,
    SignalBundle,
    aggregate,
    evidence_coverage,
    fuse,
)

_CLIP = None


def get_clip():
    global _CLIP
    if _CLIP is None:
        from models.clip_wrapper import CLIPWrapper
        from PIL import Image

        _CLIP = (CLIPWrapper({"device": "cpu", "dtype": "float32", "seed": 0,
                              "clip": {"backend": "transformers",
                                       "checkpoint": "google/siglip-base-patch16-224",
                                       "region_grid": 1, "region_overlap": 0.2,
                                       "cos_min": -0.10, "cos_max": 0.12}}),
                 Image.new("RGB", (224, 224), (200, 30, 30)))
    return _CLIP


def _v3(claim_text: str, image=None, ev: float = 0.5, sim: float = 0.5):
    clip, default_img = get_clip()
    img = default_img if image is None else image
    regions = clip.encode_regions(img)
    cc = make_counterclaim(claim_text)
    pw = clip.pairwise_scores(claim_text, cc["counterclaim_text"], regions)
    b = SignalBundle(claim_text=claim_text, evidence=ev, clip_similarity=sim,
                     stubbed=("uniprobe", "counterfactual"))
    active = {"confidence": False, "evidence": True, "clip_similarity": True,
              "uniprobe": False, "counterfactual": False}
    return fuse(b, mode="v3", active_signals=active, drop_stubbed=True,
                pairwise=pw), pw, cc


def test_pairwise_positive_claim_wins():
    r, pw, _ = _v3("a red square")
    assert pw["margin"] > 0.12, pw
    assert r.verdict == "supported", (r.verdict, r.reasons)


def test_pairwise_negative_claim_wins():
    r, pw, _ = _v3("a blue elephant")
    assert pw["margin"] < -0.12, pw
    assert r.verdict == "contradicted", (r.verdict, r.reasons)
    assert any("counterclaim wins" in x for x in r.reasons)


def test_pairwise_tie_becomes_unresolved():
    b = SignalBundle(claim_text="x", evidence=0.5, clip_similarity=0.5)
    active = {"confidence": False, "evidence": True, "clip_similarity": True,
              "uniprobe": False, "counterfactual": False}
    pw = {"claim_match": 0.54, "counter_match": 0.51, "margin": 0.03,
          "ambiguity": 0.91, "reason": "ok"}
    r = fuse(b, mode="v3", active_signals=active, pairwise=pw)
    assert r.verdict == "unresolved"
    assert any("tied" in x for x in r.reasons)


def test_pairwise_negated_claim_is_oriented_correctly():
    cc = make_counterclaim("There is no red square.")
    assert cc["is_negated"] and cc["counterclaim_text"].lower().startswith("red")
    r, pw, _ = _v3("There is no red square.")
    # False denial: positive form matches strongly -> contradicted.
    assert pw["counter_match"] > 0.60, pw
    assert r.verdict == "contradicted", (r.verdict, r.reasons)


def test_false_denial_is_contradicted():
    # "No blue" where blue IS present: positive form matches -> contradicted.
    from PIL import Image

    half = Image.new("RGB", (224, 224), (200, 30, 30))
    half.paste(Image.new("RGB", (112, 224), (30, 30, 200)), (112, 0))
    r, pw, _ = _v3("There is no blue here.", image=half)
    assert r.verdict == "contradicted", (r.verdict, r.reasons, pw)
    assert any("denial is false" in x for x in r.reasons)


def test_true_denial_is_supported():
    # True denial: positive form matches nothing -> supported.
    r, pw, _ = _v3("There is no elephant in the image.")
    assert r.verdict == "supported", (r.verdict, r.reasons, pw)


def test_hedged_claim_does_not_generate_hard_negation():
    for s in ["It appears to be a cat.", "maybe a dog", "possibly red"]:
        cc = make_counterclaim(s)
        assert cc["counterclaim_text"] is None, (s, cc)
    b = SignalBundle(claim_text="maybe a dog", evidence=0.5, clip_similarity=0.5)
    active = {"confidence": False, "evidence": True, "clip_similarity": True,
              "uniprobe": False, "counterfactual": False}
    clip, img = get_clip()
    pw = clip.pairwise_scores("maybe a dog", None, clip.encode_regions(img))
    r = fuse(b, mode="v3", active_signals=active, pairwise=pw)
    assert r.verdict == "unresolved" and any("no pairwise" in x for x in r.reasons)


def test_v3_preserves_v1_and_v2_behavior():
    b = SignalBundle(confidence=0.8, evidence=0.6, clip_similarity=0.5,
                     uniprobe=0.3, counterfactual=0.4)
    all_on = {s: True for s in ("confidence", "evidence", "clip_similarity",
                                "uniprobe", "counterfactual")}
    r1 = fuse(b, mode="v1", active_signals=all_on)
    assert abs(r1.risk - 0.4) < 1e-6 and r1.verdict is None and r1.reasons == []
    r2 = fuse(b, mode="v2", active_signals=all_on)
    assert r2.verdict is None and r2.pairwise == {}
    try:
        fuse(b, mode="v4", active_signals=all_on)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for v4")


def test_evidence_coverage_excludes_hedged_claims():
    def res(verdict=None, risk=0.1, rules=()):
        return FusionResult(risk=risk, weights={}, risks={}, contributions={},
                            ced=float("nan"), disagreement=float("nan"),
                            rules_fired=list(rules), mode="v1",
                            verdict=verdict, reasons=[])
    rows = [res("supported", 0.1), res("contradicted", 0.9),
            res("unresolved", 0.5),
            res("unresolved", 0.5, rules=["no pairwise margin available (hedged)"])]
    cov = evidence_coverage(rows)
    assert cov == {"decisive": 2, "total": 3, "coverage": 2 / 3}, cov
    cov_all = evidence_coverage(rows, exclude_hedged=False)
    assert cov_all["total"] == 4


def test_pairwise_scores_use_the_same_calibration_window():
    clip, img = get_clip()
    assert (clip.cos_min, clip.cos_max) == (-0.10, 0.12)
    regions = clip.encode_regions(img)
    pw = clip.pairwise_scores("a red square", "No red square", regions)
    assert 0.0 <= pw["claim_match"] <= 1.0 and 0.0 <= pw["counter_match"] <= 1.0
    assert abs(pw["margin"] - (pw["claim_match"] - pw["counter_match"])) < 1e-9
    assert 0.0 <= pw["ambiguity"] <= 1.0


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
