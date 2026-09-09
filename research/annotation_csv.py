#!/usr/bin/env python3
"""Convert filled annotation CSVs to the JSONL format research.prepare_dataset expects.

Usage:
    python -m research.annotation_csv annotator_A.csv annotator_B.csv --out annotations.jsonl

Combines A/B votes: rows both annotators agree on become adjudicated directly;
disagreements are flagged for adjudicator C (adjudicator column empty = pending).
Only fully adjudicated rows are emitted. Never invents labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BOOLS = {"true": True, "false": False, "": None}


def parse_bool(value: str, field: str):
    v = (value or "").strip().lower()
    if v not in ("true", "false"):
        raise ValueError(f"{field} must be TRUE/FALSE, got {value!r}")
    return BOOLS[v]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adjudicator", default="C")
    args = ap.parse_args()

    sheets = []
    for path in args.sheets:
        with open(path, encoding="utf-8-sig") as f:
            sheets.append({r["id"]: r for r in csv.DictReader(f) if r.get("id")})

    if set(sheets[0]) != set(sheets[1]):
        missing = set(sheets[0]) ^ set(sheets[1])
        raise ValueError(f"annotator sheets cover different case IDs: {sorted(missing)[:5]}")

    out, pending = [], []
    for cid in sorted(sheets[0]):
        a, b = sheets[0][cid], sheets[1][cid]
        row = dict(a)
        row["annotator_ids"] = ["A", "B"]
        row["adjudicator"] = args.adjudicator
        agree = all((a.get(k) or "") == (b.get(k) or "") for k in
                    ("original_hallucinated", "original_correct", "hallucination_type",
                     "hallucinated_spans", "corrected_answer", "severity"))
        if agree:
            row["annotation_status"] = "adjudicated"
            out.append(row)
        else:
            row["annotation_status"] = "pending_adjudication"
            pending.append(cid)

    for r in out:
        for k in ("original_hallucinated", "original_correct"):
            r[k] = parse_bool(r[k], k)
        r["hallucinated_spans"] = json.loads(r["hallucinated_spans"] or "[]")
        r["visual_evidence"] = json.loads(r["visual_evidence"] or "[]")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"adjudicated: {len(out)}, pending: {len(pending)}")
    if pending:
        print("pending IDs:", ", ".join(pending[:10]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
