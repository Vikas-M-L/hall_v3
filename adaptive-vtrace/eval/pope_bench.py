#!/usr/bin/env python3
"""Real POPE benchmark: real COCO images + real POPE labels vs our verifier.

Pipeline (the user's diagram, minus the VLM step which needs an API key/GPU):
    Real images (COCO, embedded in HF parquet)
      -> claims parsed from POPE questions ("Is there a {obj}...?" -> "a {obj}")
      -> V-TRACE+ CLIP/SigLIP scoring (no VLM)
      -> ground-truth labels (answer no => hallucinated)
      -> AUROC / AUPRC / F1 vs baselines

Baselines: siglip-grid (ours), siglip-whole (no regions), clipB32-grid,
chance. Run:  python eval/pope_bench.py --n-per-split 20 --out results/pope.json
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "vtrace-plus"))

QUESTION_RE = re.compile(r"is there a[n]?\s+(.+?)\s+in the (?:image|picture|photo)\??\s*$",
                         re.IGNORECASE)

SPLITS = {"random": "Full/random-00000-of-00001.parquet",
          "popular": "Full/popular-00000-of-00001.parquet",
          "adversarial": "Full/adversarial-00000-of-00001.parquet"}
HF = "hf://datasets/lmms-lab-encoder/POPE/"


def question_to_claim(q: str) -> str:
    m = QUESTION_RE.match(q.strip())
    return f"a {m.group(1).strip()}" if m else q.strip()


def load_split(split: str, n: int, seed: int = 0) -> list[dict]:
    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(HF + SPLITS[split])
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    out = []
    for _, r in df.iterrows():
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        out.append({"image": img,
                    "image_id": str(r["image_source"]),
                    "claim": question_to_claim(str(r["question"])),
                    "y_hall": 1 if str(r["answer"]).strip().lower() != "yes" else 0,
                    "split": split})
    return out


METHODS = {
    "so400m-grid": {"checkpoint": "google/siglip-so400m-patch14-384",
                    "grid": 3, "window": (-0.06, 0.17)},
    "siglip-grid": {"checkpoint": "google/siglip-base-patch16-224",
                    "grid": 3, "window": (-0.10, 0.12)},
    "siglip-whole": {"checkpoint": "google/siglip-base-patch16-224",
                     "grid": 1, "window": (-0.10, 0.12)},
    "clipB32-grid": {"checkpoint": "openai/clip-vit-base-patch32",
                     "grid": 3, "window": (0.15, 0.35)},
    # v3: pairwise claim-vs-counterclaim + abstention (same SigLIP, whole image).
    "v3-siglip": {"checkpoint": "google/siglip-base-patch16-224",
                  "grid": 1, "window": (-0.10, 0.12), "v3": True},
    "chance": None,
}


def make_clip(checkpoint: str, window):
    from models.clip_wrapper import CLIPWrapper

    return CLIPWrapper({"device": "cpu", "dtype": "float32", "seed": 0,
                        "clip": {"backend": "transformers", "checkpoint": checkpoint,
                                 "region_grid": 3, "region_overlap": 0.2,
                                 "cos_min": window[0], "cos_max": window[1]}})


def score_all(rows: list[dict], methods: list[str]) -> tuple[dict, dict]:
    """Returns (risks per method, verdicts per method — v3 only)."""
    import numpy as np

    risks: dict[str, list[float]] = {m: [] for m in methods}
    verdicts: dict[str, list[str | None]] = {m: [] for m in methods}
    clips = {m: make_clip(**{k: v for k, v in METHODS[m].items()
                             if k not in ("grid", "v3")})
             for m in methods if METHODS[m] is not None}
    from fusion.negation import make_counterclaim
    from fusion.vtrace_fusion import SignalBundle, fuse

    active = {"confidence": False, "evidence": True, "clip_similarity": True,
              "uniprobe": False, "counterfactual": False}
    for i, row in enumerate(rows):
        print(f"[{i + 1}/{len(rows)}] {row['image_id']} {row['claim'][:50]}", flush=True)
        for m in methods:
            spec = METHODS[m]
            if spec is None:
                risks[m].append(0.5)
                verdicts[m].append(None)
                continue
            clip = clips[m]
            regions = clip.encode_regions(row["image"], grid=spec["grid"])
            score, _, _ = clip.evidence(row["claim"], regions)
            if spec.get("v3"):
                sim = clip.similarity(row["claim"], regions)
                cc = make_counterclaim(row["claim"])
                pw = clip.pairwise_scores(row["claim"], cc["counterclaim_text"], regions)
                r = fuse(SignalBundle(claim_text=row["claim"], evidence=score,
                                      clip_similarity=sim,
                                      stubbed=("uniprobe", "counterfactual")),
                         mode="v3", active_signals=active, drop_stubbed=True,
                         pairwise=pw)
                risks[m].append(float(r.risk) if r.risk == r.risk else float("nan"))
                verdicts[m].append(r.verdict)
            else:
                risks[m].append(float(1.0 - score) if score == score else float("nan"))
                verdicts[m].append(None)
    return risks, verdicts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-split", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["siglip-grid", "siglip-whole",
                                                     "clipB32-grid", "chance"])
    ap.add_argument("--out", default="results/pope_bench.json")
    args = ap.parse_args()

    t0 = time.time()
    rows = []
    for s in ("random", "popular", "adversarial"):
        rows += load_split(s, args.n_per_split, args.seed)
    print(f"loaded {len(rows)} rows in {time.time() - t0:.1f}s", flush=True)

    from eval.metrics import auprc, auroc, f1_at_threshold

    risks, verdicts = score_all(rows, args.methods)
    claims_dump = [{"split": rows[i]["split"], "image": rows[i]["image_id"],
                    "claim": rows[i]["claim"], "y_hall": rows[i]["y_hall"],
                    **{m: risks[m][i] for m in args.methods},
                    **{f"{m}_verdict": verdicts[m][i] for m in args.methods
                       if any(v is not None for v in verdicts[m])}}
                   for i in range(len(rows))]

    def _bca_auroc(y, sc, n_boot=500, seed=0):
        """Bootstrap 95% CI for AUROC (resample items, stratified-ish)."""
        import numpy as np

        rng = np.random.default_rng(seed)
        y = np.asarray(y)
        sc = np.asarray(sc)
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if len(set(y[idx].tolist())) < 2:
                continue
            vals.append(auroc(y[idx], sc[idx]))
        if not vals:
            return (float("nan"), float("nan"))
        return (round(float(np.percentile(vals, 2.5)), 3),
                round(float(np.percentile(vals, 97.5)), 3))
    table = {}
    for m in args.methods:
        table[m] = {}
        r = dict(zip([f"{x['split']}:{i}" for i, x in enumerate(rows)], risks[m]))
        for s in ("random", "popular", "adversarial", "all"):
            idx = [i for i, x in enumerate(rows) if s == "all" or x["split"] == s]
            y = [rows[i]["y_hall"] for i in idx]
            sc = [risks[m][i] for i in idx]
            ok = [v == v for v in sc]
            y = [v for v, k in zip(y, ok) if k]
            sc = [v for v in sc if v == v]
            table[m][s] = {"n": len(y), "auroc": auroc(y, sc), "auprc": auprc(y, sc),
                           "auroc_ci95": list(_bca_auroc(y, sc)),
                           **f1_at_threshold(y, sc)}
            # v3 three-state metrics: coverage + accuracy on decisive claims only.
            vs = [verdicts[m][i] for i in idx]
            if any(v is not None for v in vs):
                dec = [(yy, v) for yy, v in zip([rows[i]["y_hall"] for i in idx], vs)
                       if v in ("supported", "contradicted")]
                table[m][s]["coverage"] = len(dec) / max(1, len(idx))
                table[m][s]["decisive_acc"] = (
                    sum(1 for yy, v in dec if (v == "contradicted") == bool(yy))
                    / len(dec)) if dec else float("nan")
                table[m][s]["n_unresolved"] = sum(1 for v in vs if v == "unresolved")
    table["_meta"] = {"n_per_split": args.n_per_split, "seed": args.seed,
                      "seconds": round(time.time() - t0, 1),
                      "note": "claims parsed from POPE questions; no VLM responses "
                              "(needs API key/GPU). y_hall=1 when POPE answer is no."}
    table["_claims"] = claims_dump
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(json.dumps({m: {s: round(v["auroc"], 3) for s, v in t.items() if s != "all" or True}
                      for m, t in table.items() if not m.startswith("_")}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
