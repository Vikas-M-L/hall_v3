import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio_service import analyze, repair_and_verify

CFG = {"question": "How many cats and what are they doing?", "grid": 3, "clip_model": "fixture",
       "aggregation": "max", "fusion_mode": "v1", "crosscheck": True, "crosscheck_limit": 3}


class Clip:
    def encode_regions(self, image, **kw): return "regions"
    def evidence(self, text, regions): return .9, (0,0,10,10), False
    def similarity(self, text, regions): return .9


class Gem:
    def __init__(self, supported=False, relevant=True):
        self.supported, self.relevant = supported, relevant
    def verify(self, image, text): return {"supported": self.supported, "raw": "Fixture observation"}
    def repair(self, image, text, question): return "Two cats are sleeping."
    def decompose(self, text): return [{"claim": text, "source": text}]
    def verify_answer(self, image, response, question): return {"supported": True, "answers_question": self.relevant}
    def audit_preservation(self, payload):
        return {"alignments": [{"fact_id": f["fact_id"], "relation": "preserved",
                                "quote": payload["candidate"], "reason": "fixture"}
                               for f in payload["original_facts"]], "candidate_extraction_complete": True}


def test_conflicting_facts_low_embedding_risk_do_not_accept():
    out = analyze(None, "5 cats are eating", {**CFG, "verify_all": True}, Clip(), Gem())
    assert len(out["claims"]) == 3
    assert out["final_verdict"] == "contradicted"
    assert out["decision_counts"]["contradicted"] == 3


def test_missing_budget_fact_remains_unresolved():
    out = analyze(None, "5 cats are eating", {**CFG, "crosscheck_limit": 1}, Clip(), Gem(True))
    assert out["final_verdict"] == "unresolved"
    assert out["warnings"]


def test_repair_rejected_if_not_relevant_even_when_supported():
    out = repair_and_verify(None, "5 cats are eating", CFG["question"], Gem(True, False), Clip(), CFG)
    assert out["status"] == "rejected_candidate" and out["final_response"] is None
    assert out["original"] == "5 cats are eating"


def test_candidate_requires_all_fact_checks_and_relevance():
    out = repair_and_verify(None, "5 cats are eating", CFG["question"], Gem(True, True), Clip(), CFG)
    assert out["accepted"] and out["status"] == "model_verified_candidate"
    assert not repair_and_verify(None, "5 cats", CFG["question"], Gem(False), Clip(), CFG)["accepted"]


def test_no_silent_truncation():
    with pytest.raises(ValueError, match="silently dropped"):
        analyze(None, "5 cats are eating", {**CFG, "max_claims": 2}, Clip())
