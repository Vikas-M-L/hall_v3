"""CPU diagnosis training on explicit image-group partitions; no live wiring."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

from research.core import write_json

FEATURES = ["clip_similarity", "region_evidence", "owl_count", "claimed_count",
            "gemini_consistency", "gemini_confidence"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/diagnosis_gbm.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    df = pd.read_csv(args.data)
    required = FEATURES + ["id", "group_id", "split", "diagnosis", "label_source"]
    if not set(required) <= set(df.columns) or df[required].isna().any().any():
        raise ValueError("missing fields/values; do not invent absent Gemini confidence")
    if df.id.duplicated().any() or (df.groupby("group_id")["split"].nunique() > 1).any():
        raise ValueError("duplicate IDs or cross-partition groups")
    if not set(df.split) <= {"train", "calibration", "test"}:
        raise ValueError("explicit train/calibration/test labels required")
    if "test" in set(df.split):
        raise ValueError("training command must not load sealed test rows")
    if not set(df.label_source) <= {"human_adjudicated", "synthetic_fixture"}:
        raise ValueError("unsupported label provenance")
    tr, va = df[df.split == "train"], df[df.split == "calibration"]
    if tr.empty or va.empty or set(tr.diagnosis) != {f"M{i}" for i in range(5)}:
        raise ValueError("train needs all M0-M4 classes and validation must be nonempty")
    X = df[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("nonfinite features")
    started = time.perf_counter()
    model = GradientBoostingClassifier(**cfg["parameters"]).fit(tr[FEATURES], tr.diagnosis)
    seconds = time.perf_counter()-started
    pred = model.predict(va[FEATURES])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "config": cfg}, out / "model.joblib")
    write_json(out / "report.json", {
        "scientific_status": "synthetic_only" if "synthetic_fixture" in set(df.label_source) else "natural_validation_only",
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(), "config": cfg,
        "train_rows": len(tr), "validation_rows": len(va), "training_seconds": seconds,
        "train_groups": sorted(tr.group_id.unique().tolist()), "validation_groups": sorted(va.group_id.unique().tolist()),
        "class_counts": tr.diagnosis.value_counts().to_dict(),
        "classification_report": classification_report(va.diagnosis, pred, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(va.diagnosis, pred, labels=[f"M{i}" for i in range(5)]).tolist(),
        "test_result": None, "backbone_finetuning": False})


if __name__ == "__main__":
    main()
