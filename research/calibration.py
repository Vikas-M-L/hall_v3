"""Offline calibration: fit on a named calibration partition, never test."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from research.core import canonical_hash, read_jsonl, write_json
from research.statistics import metrics, validate


def fit_calibrator(rows):
    if not rows or any(r.get("split") != "calibration" for r in rows):
        raise ValueError("only a nonempty calibration partition may fit probabilities")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate calibration rows")
    y, p = validate([r["label"] for r in rows], [r["score"] for r in rows])
    if len(set(y)) != 2:
        raise ValueError("both classes required for calibration")
    model = LogisticRegression(C=1, random_state=42).fit(p.reshape(-1, 1), y)
    return {"method": "logistic_score_calibration", "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]), "fit_ids": sorted(r["id"] for r in rows),
            "fit_groups": sorted({r["group_id"] for r in rows}), "seed": 42,
            "input_sha256": canonical_hash(rows), "n": len(rows)}


def transform(bundle, rows):
    if any(r["id"] in bundle["fit_ids"] or r["group_id"] in bundle["fit_groups"] for r in rows):
        raise ValueError("calibration/evaluation image overlap")
    _, p = validate([r["label"] for r in rows], [r["score"] for r in rows])
    z = np.clip(bundle["coefficient"]*p + bundle["intercept"], -700, 700)
    return 1/(1+np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bundle = fit_calibrator(read_jsonl(args.calibration))
    rows = read_jsonl(args.evaluation)
    calibrated = transform(bundle, rows)
    y = [r["label"] for r in rows]
    write_json(args.out, {"calibrator": bundle,
                         "raw": metrics(y, [r["score"] for r in rows]),
                         "calibrated": metrics(y, calibrated),
                         "note": "calibration gains are empirical, not a guarantee"})


if __name__ == "__main__":
    main()
