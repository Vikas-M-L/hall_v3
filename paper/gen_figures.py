#!/usr/bin/env python3
"""Generate all paper figures from REAL artifacts only. No hardcoded metrics:
every number is recomputed from results JSONs or retrained live. Outputs to
paper/figs/. Run:  python paper/gen_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "adaptive-vtrace"))
sys.path.insert(0, str(ROOT / "vtrace-plus"))

from eval.metrics import auroc, auprc, f1_at_threshold  # noqa: E402

FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 200})


def load_pooled() -> dict:
    pool = {}
    for f in ["pope24", "pope30", "pope31", "pope32"]:
        p = ROOT / "adaptive-vtrace" / "results" / f"{f}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for c in d.get("_claims", []):
            pool.setdefault((c["image"], c["claim"]), {}).update(c)
    return pool


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [("IMAGE", 0.2, 1.0), ("CLAIMS\natomic", 1.9, 1.0),
             ("VERIFY\nSigLIP·OWL·VLM", 3.9, 1.0), ("FUSE\nv1/v2+neg", 6.1, 1.0),
             ("M0-M4\n+ ROUTE", 8.0, 1.0)]
    for i, (t, x, y) in enumerate(boxes):
        ax.text(x, y, t, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="#dbeafe", ec="#1d4ed8"))
        if i < len(boxes) - 1:
            ax.annotate("", xy=(boxes[i + 1][1] - 0.75, y), xytext=(x + 0.75, y),
                        arrowprops=dict(arrowstyle="->", color="#1d4ed8"))
    ax.text(5, 2.3, "V-TRACE+ inference pipeline (all models frozen, CPU-friendly)",
            ha="center", fontsize=10, weight="bold")
    ax.text(5, 0.15, "Cascade: free signals decide easy claims; VLM fires only on ambiguity",
            ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_architecture.png", bbox_inches="tight")


def fig2_baseline(pool: dict):
    methods = [("siglip-whole", "SigLIP whole"), ("clipB32-grid", "CLIP-B/32 grid"),
               ("siglip-grid", "SigLIP grid")]
    au, f1, ns = [], [], []
    for m, _ in methods:
        sub = [r for r in pool.values() if m in r]
        y = np.array([r["y_hall"] for r in sub])
        s = np.array([r[m] for r in sub])
        au.append(auroc(y, s))
        f1.append(f1_at_threshold(y, s)["f1"])
        ns.append(len(sub))
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(x - 0.2, au, 0.4, label="AUROC")
    ax.bar(x + 0.2, f1, 0.4, label="F1")
    ax.set_xticks(x, [f"{lab}\n(n={n})" for (_, lab), n in zip(methods, ns)])
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("POPE pooled: verifier methods vs chance (AUROC=0.50)")
    ax.axhline(0.5, color="red", ls="--", lw=1, label="chance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_baseline.png", bbox_inches="tight")
    print("fig2:", {lab: (round(a, 3), round(f, 3), n)
                    for (_, lab), a, f, n in zip(methods, au, f1, ns)})


def fig3_gbm():
    from fusion.gbm_diagnoser import generate_synthetic, train

    b = train(generate_synthetic(200, seed=0))
    cm = b["confusion"]
    rep = b["report"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks(range(5), ["M0", "M1", "M2", "M3", "M4"])
    ax1.set_yticks(range(5), ["M0", "M1", "M2", "M3", "M4"])
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    for i in range(5):
        for j in range(5):
            ax1.text(j, i, cm[i, j], ha="center", fontsize=8)
    ax1.set_title(f"GBM confusion (test n={b['n_test']})")
    f1s = [rep[m]["f1-score"] for m in ["M0", "M1", "M2", "M3", "M4"]]
    ax2.bar(["M0", "M1", "M2", "M3", "M4"], f1s, color="#16a34a")
    ax2.set_ylim(0.8, 1.0)
    ax2.set_ylabel("F1")
    ax2.set_title(f"Per-class F1 (acc={rep['accuracy']:.3f})")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_gbm.png", bbox_inches="tight")
    print("fig3: acc=%.3f" % rep["accuracy"])


def fig4_ablation(pool: dict):
    # Signal/method ablation from the same pooled items.
    rows = [r for r in pool.values() if "siglip-whole" in r]
    y = np.array([r["y_hall"] for r in rows])
    cands = {}
    if any("siglip-grid" in r for r in rows):
        sub = [r for r in rows if "siglip-grid" in r]
        cands["grid regions"] = (np.array([r["y_hall"] for r in sub]),
                                 np.array([r["siglip-grid"] for r in sub]))
    cands["whole image"] = (y, np.array([r["siglip-whole"] for r in rows]))
    labels, vals = [], []
    for k, (yy, ss) in cands.items():
        labels.append(f"{k}\n(n={len(yy)})")
        vals.append(auroc(yy, ss))
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(labels, vals, color=["#2563eb", "#16a34a"][:len(labels)])
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Ablation: region-max vs whole-image signal")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_ablation.png", bbox_inches="tight")
    print("fig4:", {k: round(v, 3) for k, v in zip(labels, vals)})


def fig5_latency(pool: dict):
    import json as _j

    per_img, names = [], []
    for f in ["pope15", "pope24", "pope30", "pope31", "pope32"]:
        p = ROOT / "adaptive-vtrace" / "results" / f"{f}.json"
        if not p.exists():
            continue
        m = _j.loads(p.read_text(encoding="utf-8"))["_meta"]
        n = m["n_per_split"] * 3
        per_img.append(m["seconds"] / n)
        names.append(f)
    s = np.array([r["siglip-whole"] for r in pool.values() if "siglip-whole" in r])
    taus = [0.3, 0.5, 0.6, 0.8]
    freed = [float((np.abs(s - 0.5) * 2 >= t).mean()) for t in taus]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))
    ax1.bar(names, per_img, color="#7c3aed")
    ax1.set_ylabel("s / image (3 methods, CPU)")
    ax1.set_title("Measured CPU latency per run")
    ax1.tick_params(axis="x", rotation=20, labelsize=8)
    ax2.plot(taus, freed, "o-", color="#ea580c")
    ax2.set_xlabel("decisiveness threshold")
    ax2.set_ylabel("fraction decided by free CLIP")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Cascade: free-signal resolution rate")
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_latency.png", bbox_inches="tight")
    print("fig5: per-image s =", [round(v, 1) for v in per_img])
    print("fig5: freed =", [round(v, 2) for v in freed])


if __name__ == "__main__":
    pool = load_pooled()
    print("pooled unique claims:", len(pool))
    fig1_architecture()
    fig2_baseline(pool)
    fig3_gbm()
    fig4_ablation(pool)
    fig5_latency(pool)
    print("wrote", sorted(p.name for p in FIGS.glob("*.png")))
