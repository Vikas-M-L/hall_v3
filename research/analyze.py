"""Reanalyze archived scores without rerunning models or tuning on inspected data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from research.core import read_jsonl, write_json, write_jsonl
from research.statistics import cluster_bootstrap, metrics, paired_comparison, reliability, risk_coverage, selective

ROOT = Path(__file__).resolve().parents[1]


def load_cohort(files, methods):
    pool, conflicts, inventory = {}, [], []
    for path in files:
        raw = path.read_bytes()
        data = json.loads(raw)
        inventory.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(),
                          "rows": len(data.get("_claims", [])), "meta": data.get("_meta", {})})
        for row in data.get("_claims", []):
            key = (row["image"], row["claim"])
            dst = pool.setdefault(key, {"image": key[0], "claim": key[1], "y_hall": row["y_hall"],
                                        "sources": [], "variants": []})
            if dst["y_hall"] != row["y_hall"]:
                raise ValueError(f"conflicting labels: {key}")
            dst["sources"].append(path.name)
            dst["variants"].append(row.get("split"))
            for m in methods:
                if m in row:
                    if m in dst and not np.isclose(dst[m], row[m], atol=1e-6):
                        conflicts.append({"image": key[0], "claim": key[1], "method": m,
                                          "previous": dst[m], "new": row[m], "source": path.name})
                    dst.setdefault(m, row[m])
                if f"{m}_verdict" in row:
                    if f"{m}_verdict" in dst and dst[f"{m}_verdict"] != row[f"{m}_verdict"]:
                        raise ValueError("duplicate verdict conflict")
                    dst[f"{m}_verdict"] = row[f"{m}_verdict"]
    if conflicts:
        raise ValueError(f"conflicting duplicate scores require separate runs: {conflicts[:3]}")
    return list(pool.values()), inventory


def summarize(rows, method, repeats, seed):
    y, p = np.array([r["y_hall"] for r in rows]), np.array([r[method] for r in rows])
    groups = [r["image"] for r in rows]
    result = metrics(y, p)
    result["images"] = len(set(groups))
    result["auroc_uncertainty"] = cluster_bootstrap(groups,
        lambda idx: metrics(y[idx], p[idx])["auroc"], repeats, seed)
    result["reliability"] = reliability(y, p)
    result["risk_coverage"] = risk_coverage(y, p)
    result["confidence_slices"] = {}
    for threshold in (.90, .95):
        keep = np.maximum(p, 1-p) >= threshold
        result["confidence_slices"][str(threshold)] = {
            "n": int(keep.sum()), "accuracy": float(((p[keep] >= .5) == y[keep]).mean()) if keep.any() else None,
            "note": "raw score extremity; not validated confidence"}
    return result


def figures(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 300})
    out.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
    shared = report["matched_cohort"]["methods"]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    names = list(shared)
    x = np.arange(len(names))
    ax.bar(x-.18, [shared[m]["auroc"] for m in names], .36, label="AUROC", color=".25")
    ax.bar(x+.18, [shared[m]["f1"] for m in names], .36, label="F1 @ 0.5", color=".75", edgecolor="black")
    ax.set_xticks(x, names); ax.set_ylim(0, 1.05); ax.legend(); ax.set_ylabel("Score")
    ax.set_title(f"Matched exploratory cohort: {report['matched_cohort']['n']} claims")
    save(fig, "detection_matched")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], "k--", label="ideal calibration")
    for name, m in shared.items():
        ax.plot([b["score"] for b in m["reliability"]], [b["frequency"] for b in m["reliability"]], "o-", label=name)
    ax.set(xlabel="Uncalibrated risk score (bin mean)", ylabel="Observed error frequency", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=7); save(fig, "calibration_raw")
    fig, ax = plt.subplots(figsize=(4, 3))
    for name, m in shared.items():
        ax.plot([r["coverage"] for r in m["risk_coverage"]], [r["selective_risk"] for r in m["risk_coverage"]], label=name)
    ax.set(xlabel="Coverage", ylabel="Error among answered claims", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=7); save(fig, "risk_coverage")
    fig, ax = plt.subplots(figsize=(5, 3))
    sel = report["selective_runs"]
    labels = list(sel)
    ax.bar(np.arange(len(labels))-.18, [sel[n]["v3"]["coverage"] for n in labels], .36, label="Coverage", color=".3")
    ax.bar(np.arange(len(labels))+.18, [sel[n]["v3"]["selective_accuracy"] or 0 for n in labels], .36, label="Selective accuracy", color=".7")
    ax.set_xticks(range(len(labels)), [n.replace(".json", "") for n in labels]); ax.set_ylim(0, 1.05)
    ax.set_title("Archived v3: deduplicated, development-exposed"); ax.legend(fontsize=7)
    save(fig, "selective_development")
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.axis("off")
    names = ["Image + question\n+ original response", "Detect + localize\nunsupported spans", "Predict error type\nroute action", "Candidate repair", "Verify grounding\nAND answer relevance"]
    for i, label in enumerate(names):
        x = .08 + .205*i
        ax.text(x, .6, label, ha="center", va="center", fontsize=7,
                bbox={"boxstyle": "round", "fc": "white", "ec": "black"}, transform=ax.transAxes)
        if i:
            ax.annotate("", xy=(x-.085, .6), xytext=(x-.12, .6), xycoords="axes fraction",
                        arrowprops={"arrowstyle": "->"})
    ax.text(.5, .15, "Accept verified candidate / reject and abstain. Gold used only by the evaluator.\nClosed-loop protocol; empirical superiority remains unmeasured.", ha="center", fontsize=8, transform=ax.transAxes)
    save(fig, "architecture_audited")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reanalysis.json")
    ap.add_argument("--out", default="results/audited")
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    methods, repeats, seed = config["methods"], config["bootstrap_repeats"], config["seed"]
    base = ROOT / "adaptive-vtrace/results"
    rows, inventory = load_cohort([base / f for f in config["cohort"]], methods)
    common = [r for r in rows if all(m in r for m in methods)]
    matched = {m: summarize(common, m, repeats, seed) for m in methods}
    report = {"status": config["status"], "config": config, "source_files": inventory,
              "pooled_claims": len(rows), "pooled_images": len({r['image'] for r in rows}),
              "per_method_available": {m: summarize([r for r in rows if m in r], m, repeats, seed) for m in methods},
              "matched_cohort": {"n": len(common), "methods": matched}, "selective_runs": {}, "comparisons": {}}
    for m in methods[1:]:
        report["comparisons"][f"{m}_vs_{methods[0]}"] = paired_comparison(
            [r['y_hall'] for r in common], [r[methods[0]] for r in common], [r[m] for r in common],
            [r['image'] for r in common], repeats, seed)
    for name in config["selective_runs"]:
        sr, inv = load_cohort([base / name], ["siglip-whole", "v3-siglip"])
        y = [r['y_hall'] for r in sr]
        p = [r['siglip-whole'] for r in sr]
        v = [r['v3-siglip_verdict'] for r in sr]
        k = sum(x != "unresolved" for x in v)
        # Compare against the same number of most decisive BASE scores. Ties are
        # broken by stable case identity; this is descriptive, not threshold tuning.
        order = sorted(range(len(sr)), key=lambda i: (-abs(p[i]-.5), sr[i]['image'], sr[i]['claim']))
        base_v = ["unresolved"]*len(sr)
        for i in order[:k]:
            base_v[i] = "contradicted" if p[i] >= .5 else "supported"
        report["selective_runs"][name] = {
            "n_unique": len(sr), "images": len({r['image'] for r in sr}), "source": inv,
            "base_forced": metrics(y, p), "v3": selective(y, [r['v3-siglip'] for r in sr], v),
            "base_at_matched_coverage": selective(y, p, base_v),
            "note": "same-coverage descriptive baseline, no untouched test; no calibration guarantee"}
    write_json(out / "report.json", report)
    write_jsonl(out / "pooled_predictions.jsonl", rows)
    # Queue is deliberately NOT labeled 'manually inspected'. Gold only covers
    # object presence; downstream diagnosis/repair outcomes remain unannotated.
    queue = []
    for i, row in enumerate(rows):
        queue.append({"case_id": f"legacy-{i:04d}", "image_id": row['image'], "claim": row['claim'],
                      "ground_truth_object_absence": row['y_hall'], "scores": {m: row[m] for m in methods if m in row},
                      "review_status": "pending_human_review", "annotator": None,
                      "failure_category": None, "question": None, "original_response": None,
                      "repair": None, "final_verification": None})
    write_jsonl(out / "failure_review_queue.jsonl", queue)
    write_json(out / "development_exclusion.json", sorted({r['image'] for r in rows}))
    table = ["# Matched-cohort exploratory results", "", "All rows below use identical claims and image-cluster intervals.", "",
             "| Method | n | Images | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | ECE | Brier |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in matched.items():
        table.append(f"| {name} | {m['n']} | {m['images']} | " + " | ".join(f"{m[k]:.4f}" for k in
                     ("accuracy", "precision", "recall", "f1", "auroc", "auprc", "ece", "brier")) + " |")
    table += ["", "Raw-score ECE/Brier diagnose miscalibration; scores are not validated probabilities.",
              "Original VLM, prompt detector, authentic existing mitigation, fine-tuned detector, and full repair-system results: N/A.",
              "Per-method inference latency, tokens, memory, API cost and correction outcomes: N/A in legacy artifacts."]
    (out / "main_table.md").write_text("\n".join(table)+"\n", encoding="utf-8")
    figures(report, out / "figures")
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "pandas", "torch", "transformers", "streamlit", "joblib", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write_json(out / "environment.json", {"python": platform.python_version(), "platform": platform.platform(), "packages": packages})
    print("Pooled:", report["pooled_claims"], "claims /", report["pooled_images"], "images")
    print("Matched:", len(common), "claims")
    for name, r in report["selective_runs"].items():
        print(name, "v3", r["v3"], "base matched", r["base_at_matched_coverage"])
    print("Artifacts:", out)


if __name__ == "__main__":
    main()
