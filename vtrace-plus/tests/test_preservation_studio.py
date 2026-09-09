import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio_service import repair_and_verify
from test_studio_service import Clip, CFG


class Gem:
    def __init__(self, drop=False): self.drop = drop
    def decompose(self, text):
        return [{"claim": s.strip()+".", "source": s.strip()+"."} for s in text.split(".") if s.strip()]
    def verify(self, image, text):
        return {"supported": "ball" not in text, "raw": "fixture observation"}
    def repair(self, image, text, question):
        return "Cats are present." if self.drop else "Cats are present. The sofa is pink."
    def verify_answer(self, image, candidate, question):
        return {"supported": True, "answers_question": True}
    def audit_preservation(self, payload):
        return {"alignments": [{"fact_id": f["fact_id"],
                                "relation": "preserved" if f["text"] in payload["candidate"] else "omitted",
                                "quote": f["text"] if f["text"] in payload["candidate"] else None}
                               for f in payload["original_facts"]], "candidate_extraction_complete": True}


def test_studio_rejects_deletion_of_supported_sofa_despite_judge_approval():
    out = repair_and_verify(None, "Cats are present. The sofa is pink. A ball is present.",
                            "Describe the scene", Gem(drop=True), Clip(), CFG)
    assert not out["accepted"] and out["preservation"]["changed_or_omitted_protected"]


def test_studio_accepts_model_checked_preservation_not_claimed_gold():
    out = repair_and_verify(None, "Cats are present. The sofa is pink. A ball is present.",
                            "Describe the scene", Gem(), Clip(), CFG)
    assert out["accepted"]
    assert out["preservation"]["preserved_count"] == 2
    assert out["preservation"]["refuted_original_removed"]


def test_missing_auditor_does_not_silently_fall_back_to_accept():
    gem = Gem()
    gem.audit_preservation = None
    out = repair_and_verify(None, "Cats are present.", "Describe", gem, Clip(), CFG)
    assert not out["accepted"] and out["preservation"]["audit_error"] == "TypeError"
