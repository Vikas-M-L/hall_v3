"""Preservation contract tests. Canned evidence is synthetic, never a benchmark."""
import copy

import pytest

from research.preservation import ledger, alignment_inputs, check_alignment, assess
from research.core import Case, run_case

ORIGINAL = "Two cats rest on a pink sofa. A red ball is nearby."
CANDIDATE = "Two cats rest on a pink sofa."


def fact(text, verdict):
    return {"text": text, "source": text, "verdict": verdict}


def setup(candidate=CANDIDATE):
    before = ledger(ORIGINAL, [fact(CANDIDATE, "supported"), fact("A red ball is nearby.", "contradicted")])
    after = ledger(candidate, [fact(candidate, "supported")], "c")
    alignment = {"alignments": [{"fact_id": "o0", "relation": "preserved", "quote": candidate},
                                 {"fact_id": "o1", "relation": "omitted", "quote": None}],
                 "candidate_extraction_complete": True}
    return before, after, alignment


def test_valid_supported_fact_preservation_and_removed_error():
    b, a, m = setup()
    r = assess(b, a, CANDIDATE, m, {"supported": True, "answers_question": True})
    assert r["accepted"] and r["preserved_count"] == 1
    assert r["refuted_original_removed"] == ["o1"]
    assert r["model_observed_preservation_rate"] == 1
    assert "correction_success" not in r


@pytest.mark.parametrize("relation,quote", [("omitted", None), ("uncertain", None), ("contradicted", CANDIDATE)])
def test_harmful_or_unresolved_edit_rejected_despite_whole_answer_approval(relation, quote):
    b, a, m = setup()
    m["alignments"][0].update(relation=relation, quote=quote)
    r = assess(b, a, CANDIDATE, m, {"supported": True, "answers_question": True})
    assert not r["accepted"]
    assert r["changed_or_omitted_protected"] or r["uncertain_protected"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "fabricated_quote", "null", "nonboolean"])
def test_alignment_schema_fails_closed(mutation):
    b, a, m = setup()
    if mutation == "missing": m["alignments"].pop()
    elif mutation == "duplicate": m["alignments"].append(m["alignments"][0])
    elif mutation == "unknown": m["alignments"][0]["fact_id"] = "o99"
    elif mutation == "fabricated_quote": m["alignments"][0]["quote"] = "five dogs"
    elif mutation == "nonboolean": m["candidate_extraction_complete"] = "true"
    else: m = None
    r = assess(b, a, CANDIDATE, m, {"supported": True, "answers_question": True})
    assert not r["accepted"] and any("invalid" in x for x in r["reasons"])


def test_new_unsupported_or_unresolved_fact_rejected():
    b, a, m = setup()
    for verdict in ("contradicted", "unresolved"):
        a["facts"][0]["evidence_verdict"] = verdict
        assert not assess(b, a, CANDIDATE, m, {"supported": True, "answers_question": True})["accepted"]


def test_incomplete_candidate_decomposition_rejected():
    b, a, m = setup()
    m["candidate_extraction_complete"] = False
    assert not assess(b, a, CANDIDATE, m, {"supported": True, "answers_question": True})["accepted"]


def test_original_refuted_fact_retained_requires_review():
    b, a, m = setup(candidate=ORIGINAL)
    m["alignments"][1].update(relation="preserved", quote="A red ball is nearby.")
    r = assess(b, a, ORIGINAL, m, {"supported": True, "answers_question": True})
    assert not r["accepted"] and r["refuted_original_retained"] == ["o1"]


def test_zero_protected_facts_is_not_100_percent_preservation():
    b, a, m = setup()
    b["facts"][0].update(protected=False, evidence_verdict="unresolved")
    r = assess(b, a, CANDIDATE, m, {"supported": True, "answers_question": True})
    assert r["protected_count"] == 0 and r["model_observed_preservation_rate"] is None


def test_alignment_auditor_does_not_receive_visual_verdict_or_protection_flag():
    b, a, _ = setup()
    p = alignment_inputs(b, a, CANDIDATE)
    assert all(set(f) == {"fact_id", "text"} for f in p["original_facts"] + p["candidate_facts"])


def test_no_preservation_ablation_accepts_edit_gate_rejects():
    b, a, m = setup()
    m["alignments"][0].update(relation="omitted", quote=None)
    v = {"supported": True, "answers_question": True}
    assert not assess(b, a, CANDIDATE, m, v)["accepted"]
    assert assess(b, a, CANDIDATE, m, v, enabled=False)["accepted"]


def test_real_orchestration_collects_equal_evidence_for_gate_ablation():
    class Backend:
        def __init__(self): self.stages = []
        def call(self, stage, inputs, action):
            self.stages.append(stage)
            if stage == "detect":
                text = inputs["response"]
                original = text == ORIGINAL
                facts = [{"text": CANDIDATE, "source": CANDIDATE, "type": "object", "risk": .1, "verdict": "supported"}]
                if original:
                    facts.append({"text": "A red ball is nearby.", "source": "A red ball is nearby.", "type": "object", "risk": .9, "verdict": "contradicted"})
                return {"claims": facts}
            if stage == "repair": return {"response": CANDIDATE}
            if stage == "verify": return {"supported": True, "answers_question": True}
            return {"alignments": [{"fact_id": f["fact_id"], "relation": "omitted", "quote": None} for f in inputs["original_facts"]],
                    "candidate_extraction_complete": True}
    c = Case("x", "g", "fixture.jpg", "a"*64, "Describe the scene", ORIGINAL, "fixture")
    enabled, disabled = Backend(), Backend()
    r = run_case(c, enabled, "preservation_full")
    ablated = run_case(c, disabled, "preservation_gate_disabled")
    assert r["status"] == "rejected_repair_abstained"
    assert ablated["status"] == "accepted_repair"
    assert enabled.stages == disabled.stages == ["detect", "repair", "verify", "detect", "preservation"]
