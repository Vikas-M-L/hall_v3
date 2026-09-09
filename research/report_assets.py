"""Generate paper-ready assets from audited outputs only; no retraining here."""
import json
from pathlib import Path

import numpy as np

from research.core import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/audited"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "savefig.dpi": 300})
    def save(fig, name):
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / "figures" / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)
    data = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    gbm = json.loads((OUT / "synthetic_gbm.json").read_text(encoding="utf-8"))
    cm = np.array(gbm["confusion"])
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    axes[0].imshow(cm, cmap="Greys")
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j]>25 else "black", fontsize=7)
    axes[0].set_xticks(range(5), [f"M{i}" for i in range(5)])
    axes[0].set_yticks(range(5), [f"M{i}" for i in range(5)])
    axes[0].set(xlabel="Predicted", ylabel="Synthetic label", title="Synthetic holdout (n=200)")
    normal = [e["same_generator"]["accuracy"] for e in gbm["evaluations"]]
    shifted = [e["count_pattern_shift"]["accuracy"] for e in gbm["evaluations"]]
    axes[1].bar([0, 1], [np.mean(normal), np.mean(shifted)],
                yerr=[np.std(normal, ddof=1), np.std(shifted, ddof=1)], color=[".3", ".7"], capsize=4)
    axes[1].set_xticks([0,1], ["Same generator", "Variable counts"])
    axes[1].set(ylim=(0,1.05), ylabel="Accuracy", title="Three synthetic evaluation seeds")
    save(fig, "diagnosis_shift")
    telemetry = []
    for name in ("live_smoke", "controlled_edit_smoke"):
        p = OUT / f"{name}.json"
        if not p.exists():
            continue
        run = json.loads(p.read_text(encoding="utf-8"))
        for trace in run["traces"]:
            events = trace["events"]
            logical_seconds = sum(e.get("recorded_seconds", e.get("seconds", 0)) or 0 for e in events)
            tokens = [(e.get("recorded_usage") or e.get("usage") or {}).get("totalTokenCount") for e in events]
            telemetry.append({"run": name, "strategy": trace["strategy"], "status": trace["status"],
                              "logical_calls": len(events), "network_calls": sum(e.get("network_calls",0) for e in events),
                              "recorded_service_seconds": logical_seconds,
                              "tokens": sum(tokens) if all(t is not None for t in tokens) else None,
                              "accuracy": None, "note": "development smoke; replay service durations are historical, not new requests"})
    write_json(OUT / "live_telemetry.json", telemetry)
    fig, axes = plt.subplots(1,2,figsize=(7,2.7))
    subset = [r for r in telemetry if r["strategy"] != "original"]
    names = [r["run"].replace("_smoke", "")+"\n"+r["strategy"].replace("uniform_self_correction","uniform") for r in subset]
    axes[0].bar(names, [r["logical_calls"] for r in subset], color=".35")
    axes[0].set(ylabel="Logical service calls", title="Smoke traces, not savings experiment")
    axes[1].bar(names, [r["recorded_service_seconds"] for r in subset], color=".6")
    axes[1].set(ylabel="Recorded service seconds", title="Excludes original answer generation")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6)
    save(fig, "smoke_telemetry")
    fig, ax = plt.subplots(figsize=(6,2.8))
    ax.axis("off")
    taxonomy = [["object / attribute", "visual or attribute recheck"], ["counting / spatial / relation", "recount or focused recheck"],
                ["OCR / reasoning", "text or reasoning recheck"], ["knowledge / insufficient evidence", "external evidence needed / abstain"]]
    ax.table(cellText=taxonomy, colLabels=["Predicted error content (not causal proof)", "Candidate action"], loc="center", cellLoc="left")
    save(fig, "taxonomy")
    table = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Method & Acc. & Prec. & Rec. & F1 & AUROC & ECE \\", r"\midrule"]
    for name, m in data["matched_cohort"]["methods"].items():
        table.append(name.replace("_", r"\_") + " & " + " & ".join(f"{m[k]:.3f}" for k in ("accuracy","precision","recall","f1","auroc","ece")) + r" \\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "main_table.tex").write_text("\n".join(table)+"\n", encoding="utf-8")
    missing = {"hallucination_reduction": "independent before/after labels absent",
               "full_component_ablation": "only protocol arms/legacy encoder variants available",
               "repair_routing_distribution_on_test": "no untouched annotated full-system test",
               "fifty_manually_inspected_failures": "review queue created; no completed human annotations"}
    write_json(OUT / "missing_figures.json", missing)
    print("Paper assets and smoke telemetry generated from recorded outputs.")


if __name__ == "__main__":
    main()
