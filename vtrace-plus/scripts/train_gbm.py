#!/usr/bin/env python3
"""Train the offline GBM M0-M4 diagnoser (STANDALONE — writes an artifact,
wires nothing live).

    python scripts/train_gbm.py --n 200 --out models/vtrace_m0_m4_diagnosis.joblib

Prints the classification report + confusion matrix. The artifact is gitignored
(*.joblib): it is a build product, not source. To present: "Gradient Boosting
M0-M4 Diagnosis Classifier" — never "fine-tuned Qwen".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fusion.gbm_diagnoser import generate_synthetic, save_bundle, train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="rows per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/vtrace_m0_m4_diagnosis.joblib")
    ap.add_argument("--data-out", default=None, help="also save the CSV")
    args = ap.parse_args()

    df = generate_synthetic(args.n, args.seed)
    if args.data_out:
        p = REPO / args.data_out
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        print("data:", p, len(df), "rows")
    bundle = train(df)
    print(f"train={bundle['n_train']} test={bundle['n_test']}")
    print(bundle["report_text"])
    print("Confusion Matrix (rows=true M0..M4):")
    print(bundle["confusion"])
    print("saved:", save_bundle(bundle, REPO / args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
