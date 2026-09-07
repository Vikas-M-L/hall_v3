#!/usr/bin/env python3
"""Ordering accuracy guard — needs CLIP weights (CPU) but no VLM/GPU.

Asserts the core accuracy property: a true claim scores lower risk than a
false claim on the same image. If this fails, the verifier is broken, not
just uncalibrated. Run:  python tests/test_ordering.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from fusion.vtrace_fusion import SignalBundle, fuse
from models.clip_wrapper import CLIPWrapper


def make_clip(grid: int = 1):
    return CLIPWrapper({
        "device": "cpu", "dtype": "float32", "seed": 42,
        "clip": {"backend": "transformers",
                 "checkpoint": "openai/clip-vit-base-patch32",
                 "region_grid": grid, "region_overlap": 0.2,
                 "cos_min": 0.10, "cos_max": 0.40},
    })


ACTIVE = {"confidence": False, "evidence": True, "clip_similarity": True,
          "uniprobe": False, "counterfactual": False}


def risk_for(clip, image, claim: str) -> float:
    regions = clip.encode_regions(image)
    score, _, _ = clip.evidence(claim, regions)
    sim = clip.similarity(claim, regions)
    b = SignalBundle(claim_text=claim, evidence=score, clip_similarity=sim,
                     stubbed=("uniprobe", "counterfactual"))
    return fuse(b, mode="v1", active_signals=ACTIVE, drop_stubbed=True).risk


def test_true_claim_beats_false_claim():
    clip = make_clip()
    red = Image.new("RGB", (224, 224), (200, 30, 30))
    r_true = risk_for(clip, red, "a red square")
    r_false = risk_for(clip, red, "a blue elephant")
    assert r_true < r_false, f"true={r_true:.3f} false={r_false:.3f}"
    assert r_false - r_true > 0.05, f"margin too thin: {r_false - r_true:.3f}"


def test_attribute_beats_wrong_attribute():
    clip = make_clip()
    red = Image.new("RGB", (224, 224), (200, 30, 30))
    r_red = risk_for(clip, red, "a red square")
    r_blue = risk_for(clip, red, "a blue square")
    assert r_red < r_blue, f"red={r_red:.3f} blue={r_blue:.3f}"


def test_whole_won_flag_distinguishes_global_from_local():
    clip = make_clip()
    solid = Image.new("RGB", (224, 224), (200, 30, 30))
    _, _, whole_won = clip.evidence("a red square", clip.encode_regions(solid))
    assert whole_won is True  # global match: UI must NOT draw a crop box
    half = Image.new("RGB", (224, 224), (200, 30, 30))
    half.paste(Image.new("RGB", (112, 224), (30, 30, 200)), (112, 0))
    clip3 = make_clip(grid=3)
    _, box, whole_won = clip3.evidence("a red square", clip3.encode_regions(half))
    assert whole_won is False and box is not None and box[2] <= 130, box


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
