"""Generate ablation configs; actual outcomes come from research.run, not tables."""
import argparse
import json
from pathlib import Path
from research.core import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/closed_loop.json")
    ap.add_argument("--out", default="experiments/ablation_configs")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    # Every implemented strategy is a controlled protocol arm. Do not label
    # API prompt variants as an implementation of Woodpecker/VCD/Self-Refine.
    arms = {"no_repair": "detector_only", "no_router": "detect_then_fixed",
            "no_diagnosis": "no_diagnosis", "no_verifier": "routed_no_verifier",
            "random_router": "random_router", "surface_type_router": "type_routing",
            "uniform_correction": "uniform_self_correction", "full": "full",
            "preservation_full": "preservation_full", "preservation_fixed": "preservation_fixed",
            "preservation_gate_disabled": "preservation_gate_disabled"}
    for name, strategy in arms.items():
        write_json(Path(args.out) / f"{name}.json", {**cfg, "strategies": [strategy]})
    write_json(Path(args.out) / "missing_arms.json", {
        "status": "N/A until implemented and executed",
        "arms": ["authentic Woodpecker/VCD", "natural-label learned router", "no OWL", "no SigLIP",
                 "LoRA ranks", "SFT detector", "DPO detector", "cross-backbone", "cross-dataset"]})
    print("Ablation configurations written. No result values were generated.")


if __name__ == "__main__":
    main()
