"""Make an explicitly synthetic replay fixture; no scientific claims or images."""
import argparse
from dataclasses import asdict
from pathlib import Path

from research.backends import ReplayBackend
from research.core import Case, STRATEGIES, run_case, write_jsonl


class Fixture:
    def __init__(self):
        self.records = {}

    def call(self, stage, inputs, action):
        if stage == "detect":
            out = {"claims": [{"text": inputs["response"], "source": "three cats",
                               "risk": .9, "type": "counting", "verdict": "contradicted",
                               "reason": "synthetic fixture evidence, not a model prediction"}]}
        elif stage == "repair":
            out = {"response": "There are two cats."}
        elif stage == "preservation":
            out = {"alignments": [{"fact_id": f["fact_id"], "relation": "contradicted",
                                    "quote": inputs["candidate"], "reason": "synthetic fixture"}
                                   for f in inputs["original_facts"]], "candidate_extraction_complete": True}
        else:
            out = {"supported": True, "answers_question": True, "reason": "fixture"}
        key = ReplayBackend.key(stage, inputs, action)
        self.records[key] = {"key": key, "output": out, "synthetic": True,
                             "model_requested": "fixture", "usage": None, "seconds": None}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/fixture")
    args = ap.parse_args()
    c = Case("fixture-count", "fixture-scene", "not-a-real-image.jpg", "a"*64,
             "How many cats?", "There are three cats.", "fixture")
    b = Fixture()
    for strategy in STRATEGIES:
        run_case(c, b, strategy)
    out = Path(args.out)
    write_jsonl(out / "inputs.jsonl", [asdict(c)])
    write_jsonl(out / "recording.jsonl", list(b.records.values()))
    print("Synthetic fixture written; use only to verify replay plumbing.")


if __name__ == "__main__":
    main()
