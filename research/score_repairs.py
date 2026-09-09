"""Evaluate independently adjudicated outputs; model verdicts cannot be gold."""
import argparse
from collections import defaultdict
from research.core import read_jsonl, write_json
from research.statistics import repair_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adjudicated", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    groups, seen = defaultdict(list), set()
    for row in read_jsonl(args.adjudicated):
        key = (row["id"], row["strategy"])
        if key in seen:
            raise ValueError("duplicate adjudicated case/strategy")
        if row.get("label_source") not in ("human_adjudicated", "exact_official_answer"):
            raise ValueError("independent gold required; model-verifier judgments are not gold")
        seen.add(key)
        groups[row["strategy"]].append(row)
    write_json(args.out, {s: repair_metrics(rows) for s, rows in groups.items()})


if __name__ == "__main__":
    main()
