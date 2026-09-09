import pytest
from research.evaluate_preservation import evaluate


def row(strategy, **kw):
    return {"id": "x", "group_id": "image1", "strategy": strategy, "label_source": "human_adjudicated",
            "original_hallucinated": True, "original_correct": False, "abstained": False,
            "final_hallucinated": False, "final_correct": True, "supported_facts_before": 2,
            "supported_facts_retained": 2, "new_unsupported_facts": 0, **kw}


def test_abstention_does_not_match_successful_repair():
    r = evaluate([row("a", abstained=True, final_correct=None, final_hallucinated=None, supported_facts_retained=0), row("b")], "a", "b", repeats=20)
    assert r["baseline"]["correction_success_rate"] == 0
    assert r["proposed"]["fact_preservation_rate"] == 1
    assert r["delta_preservation_aware_utility"] == 1


def test_requires_matched_cases_and_consistent_gold():
    with pytest.raises(ValueError, match="Same complete"):
        evaluate([row("a"), row("b", id="y")], "a", "b")
    with pytest.raises(ValueError, match="Original gold"):
        evaluate([row("a"), row("b", group_id="other")], "a", "b")


def test_api_judge_results_cannot_be_gold():
    with pytest.raises(ValueError, match="independent gold"):
        evaluate([row("a", label_source="gemini"), row("b")], "a", "b")
