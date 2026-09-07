#!/usr/bin/env python3
"""Precompute gallery results (local signals only, no API spend).

Writes assets/gallery_results.json: per scenario, per claim {risk, verdict,
rules_fired, expected, correct?}. Count-scenario verdicts additionally need
the live OWL check — the gallery marks them; the UI proves them on click.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from PIL import Image

from fusion.negation import apply_negation
from fusion.vtrace_fusion import SignalBundle, fuse
from models.clip_wrapper import CLIPWrapper

ASSETS = REPO / "assets"
ACTIVE = {"confidence": False, "evidence": True, "clip_similarity": True,
          "uniprobe": False, "counterfactual": False}


def verdict(risk: float) -> str:
    if risk != risk:
        return "N/A"
    if risk < 0.4:
        return "grounded"
    if risk < 0.6:
        return "uncertain"
    return "hallucinated"


def main() -> int:
    scenarios = json.loads((ASSETS / "scenarios.json").read_text(encoding="utf-8"))
    clip = CLIPWrapper({"device": "cpu", "dtype": "float32", "seed": 0,
                        "clip": {"backend": "transformers",
                                 "checkpoint": "google/siglip-base-patch16-224",
                                 "region_grid": 3, "region_overlap": 0.2,
                                 "cos_min": -0.10, "cos_max": 0.12}})
    out = []
    for sc in scenarios:
        image = Image.open(ASSETS / sc["image"]).convert("RGB")
        regions = clip.encode_regions(image)
        rows = []
        for cl in sc["claims"]:
            t = cl["text"]
            s, box, whole = clip.evidence(t, regions)
            sim = clip.similarity(t, regions)
            r = fuse(SignalBundle(claim_text=t, evidence=s, clip_similarity=sim,
                                  stubbed=("uniprobe", "counterfactual")),
                     mode="v1", active_signals=ACTIVE, drop_stubbed=True)
            corr, neg = apply_negation(t, r.risks.get("evidence", float("nan")),
                                       r.risks.get("clip_similarity", float("nan")))
            if neg and corr == corr:
                r.risk = corr
                r.rules_fired = [*r.rules_fired, "negation-inverted"]
            v = verdict(r.risk)
            rows.append({"text": t, "risk": round(float(r.risk), 3), "verdict": v,
                         "expected": cl["expected"],
                         "correct": v == cl["expected"],
                         "correction": cl.get("correction", ""),
                         "rules": r.rules_fired,
                         "whole_won": whole})
            print(f"{sc['id']:10s} {r.risk:.3f} ({v:12s} exp {cl['expected']:12s}) {t[:50]}",
                  flush=True)
        out.append({**sc, "results": rows,
                    "pairwise_ok": bool(rows[0]["risk"] < rows[1]["risk"])})
    (ASSETS / "gallery_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    n_ok = sum(1 for sc in out for r in sc["results"] if r["correct"])
    n = sum(len(sc["results"]) for sc in out)
    n_pair = sum(1 for sc in out if sc["pairwise_ok"])
    print(f"\ngallery absolute accuracy: {n_ok}/{n}")
    print(f"gallery pairwise ranking: {n_pair}/{len(out)} (the claimed capability)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
