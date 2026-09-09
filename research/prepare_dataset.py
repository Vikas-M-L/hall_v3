"""Validate independently annotated cases and separate prediction inputs from gold.

No labels or answers are generated here. Group split assignment occurs before
export; official partitions take precedence. A separate custodian must hold test
gold. Development-exposed image groups can never become a new test.
"""
import argparse
from collections import Counter
from pathlib import Path

from research.core import Case, TYPES, assign_split, canonical_hash, read_jsonl, validate_cases, write_json, write_jsonl


def prepare(rows, salt, development_groups=()):
    import dataclasses
    inputs, gold = [], []
    required = {"id", "group_id", "image", "image_sha256", "question", "response",
                "correct_answer", "original_hallucinated", "original_correct",
                "hallucination_type", "hallucinated_spans", "visual_evidence", "severity",
                "corrected_answer", "source_dataset", "source_id", "source_license",
                "generator_model", "generator_revision", "prompt_id", "annotation_status",
                "annotator_ids", "adjudicator"}
    for row in rows:
        if not required <= set(row):
            raise ValueError(f"missing annotation fields: {sorted(required-set(row))}")
        if row["annotation_status"] != "adjudicated" or len(set(row["annotator_ids"])) < 2 or not row["adjudicator"]:
            raise ValueError("two independent annotators and adjudication required")
        if row["hallucination_type"] not in (*TYPES, "none"):
            raise ValueError("invalid annotation taxonomy")
        if any(type(row[k]) is not bool for k in ("original_hallucinated", "original_correct")):
            raise ValueError("explicit gold correctness required")
        if not row["source_license"]:
            raise ValueError("record source license/rights before export")
        split = row.get("official_partition") or assign_split(row["group_id"], salt, development_groups)
        if row["group_id"] in development_groups and split == "test":
            raise ValueError("development-exposed group cannot enter sealed test")
        c = Case.parse({**{k: row[k] for k in Case.__dataclass_fields__ if k != "split"}, "split": split})
        for span in row["hallucinated_spans"]:
            if (not isinstance(span, list) or len(span) != 2 or
                    any(type(x) is not int for x in span) or not 0 <= span[0] < span[1] <= len(c.response)):
                raise ValueError("invalid gold span")
        inputs.append(c)
        gold.append({k: v for k, v in row.items() if k not in ("image", "image_sha256", "question", "response")})
        gold[-1]["split"] = split
    validate_cases(inputs)
    return [dataclasses.asdict(c) for c in inputs], gold


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--salt", required=True)
    ap.add_argument("--development-exclusion", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = read_jsonl(args.annotations)
    exclusion = json.loads(Path(args.development_exclusion).read_text(encoding="utf-8"))
    inputs, gold = prepare(rows, args.salt, exclusion)
    out = Path(args.out)
    for split in sorted({c["split"] for c in inputs}):
        write_jsonl(out / f"{split}_inputs.jsonl", [r for r in inputs if r["split"] == split])
        write_jsonl(out / f"{split}_gold.jsonl", [r for r in gold if r["split"] == split])
    write_json(out / "manifest.json", {"input_sha256": canonical_hash(inputs), "gold_sha256": canonical_hash(gold),
                                      "split_salt": args.salt, "n": len(inputs),
                                      "type_counts": dict(Counter(r["hallucination_type"] for r in gold)),
                                      "split_counts": dict(Counter(r["split"] for r in inputs)),
                                      "note": "Move test gold to an independent evaluation custodian."})


if __name__ == "__main__":
    main()
