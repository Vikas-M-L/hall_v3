"""Score independent original-fact preservation labels on matched strategy cases."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from research.core import read_jsonl, write_json
from research.statistics import cluster_bootstrap, repair_metrics


def evaluate(rows, baseline, proposed, repeats=1000, seed=42):
    indexed = defaultdict(dict)
    for r in rows:
        if r.get("label_source") not in ("human_adjudicated", "exact_official_answer"):
            raise ValueError("Model-audit decisions are not independent gold")
        if not r.get("group_id") or r["id"] in indexed[r["strategy"]]:
            raise ValueError("Group ID required; duplicate case/strategy forbidden")
        counts = [r.get(k) for k in ("supported_facts_before", "supported_facts_retained", "new_unsupported_facts")]
        if any(type(v) is not int or v < 0 for v in counts) or counts[1] > counts[0]:
            raise ValueError("Valid independent fact counts required")
        if r["abstained"] and counts[1] != 0:
            raise ValueError("Abstention cannot retain answered facts")
        repair_metrics([r])
        indexed[r["strategy"]][r["id"]] = r
    if baseline not in indexed or proposed not in indexed:
        raise ValueError("Both requested strategies must have adjudicated outcomes")
    ids = sorted(indexed[baseline])
    if not ids or set(ids) != set(indexed[proposed]):
        raise ValueError("Same complete case set required; do not silently drop unmatched cases")
    a, b = [[indexed[s][i] for i in ids] for s in (baseline, proposed)]
    for x, y in zip(a, b):
        if any(x[k] != y[k] for k in ("group_id", "original_correct", "original_hallucinated", "supported_facts_before")):
            raise ValueError("Original gold/group differs between strategies")
    def summary(rs):
        total = sum(r["supported_facts_before"] for r in rs)
        return {**repair_metrics(rs), "supported_facts_before": total,
                "supported_facts_retained": sum(r["supported_facts_retained"] for r in rs),
                "fact_preservation_rate": sum(r["supported_facts_retained"] for r in rs)/total if total else None,
                "new_unsupported_facts": sum(r["new_unsupported_facts"] for r in rs)}
    def utility(r):
        corrected = r["original_hallucinated"] and not r["abstained"] and r["final_correct"] and not r["final_hallucinated"]
        lost = r["original_correct"] and (r["abstained"] or not r["final_correct"])
        return int(corrected)-int(lost)
    diff = np.array([utility(y)-utility(x) for x, y in zip(a, b)], dtype=float)
    return {"status": "independently_adjudicated_paired_comparison", "seed": seed,
            "baseline": {"strategy": baseline, **summary(a)}, "proposed": {"strategy": proposed, **summary(b)},
            "delta_preservation_aware_utility": float(diff.mean()),
            "uncertainty": cluster_bootstrap([r["group_id"] for r in a], lambda idx: diff[idx].mean(), repeats, seed),
            "note": "Annotation independence and provenance must be audited externally; schema validation cannot establish them."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adjudicated", required=True)
    ap.add_argument("--baseline", default="preservation_fixed")
    ap.add_argument("--proposed", default="preservation_full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    write_json(args.out, evaluate(read_jsonl(args.adjudicated), args.baseline, args.proposed, seed=args.seed))


if __name__ == "__main__":
    main()
