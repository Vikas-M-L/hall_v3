"""Reproduce the legacy GBM, expose its generator assumptions; never live-wire it."""
import argparse
import hashlib
import importlib.util
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from research.core import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/audited/synthetic_gbm.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "vtrace-plus/fusion/gbm_diagnoser.py"
    spec = importlib.util.spec_from_file_location("legacy_gbm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    start = time.perf_counter()
    b = module.train(module.generate_synthetic(200, seed=0))
    evaluations = []
    for seed in (11, 12, 13):
        df = module.generate_synthetic(200, seed=seed)
        pred = b["model"].predict(df[module.FEATURES])
        normal = {"accuracy": float(accuracy_score(df.diagnosis, pred)),
                  "macro_f1": float(f1_score(df.diagnosis, pred, average="macro"))}
        # Same labels and continuous features, replace fixed category counts
        # by variable counts that preserve equality/mismatch semantics.
        rng = np.random.default_rng(seed)
        n = rng.integers(1, 8, len(df))
        df["owl_count"], df["claimed_count"] = n, n
        df.loc[df.diagnosis == "M1", "owl_count"] = 0
        df.loc[df.diagnosis == "M2", "claimed_count"] += 1
        shifted = b["model"].predict(df[module.FEATURES])
        evaluations.append({"seed": seed, "n": len(df), "same_generator": normal,
                            "count_pattern_shift": {"accuracy": float(accuracy_score(df.diagnosis, shifted)),
                            "macro_f1": float(f1_score(df.diagnosis, shifted, average="macro"))}})
    artifact = root / "vtrace-plus/models/vtrace_m0_m4_diagnosis.joblib"
    write_json(args.out, {"status": "synthetic diagnostic experiment, not natural M0-M4 accuracy",
                         "train_generator_seed": 0, "split_seed": 42,
                         "n_train": b["n_train"], "n_holdout": b["n_test"],
                         "holdout_report": b["report"], "confusion": b["confusion"].tolist(),
                         "evaluations": evaluations, "elapsed_seconds": time.perf_counter()-start,
                         "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "existing_joblib_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else None})
    print("Synthetic GBM study saved; no app connection or adapter training.")


if __name__ == "__main__":
    main()
