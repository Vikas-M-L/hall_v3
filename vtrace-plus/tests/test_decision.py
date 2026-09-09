"""Regression for cat-count/action claim refuted by Gemini but labeled grounded."""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.decision import finalize, json_safe, summarize


def supplied_case():
    return {"claim_text": "5 cats are eating", "text": "5 cats are eating",
            "claim_type": "count", "risk": 0.35193152674897143,
            "verdict": None, "reasons": [], "pairwise": {},
            "risks": {"evidence": 0.2640286921100182, "clip_similarity": 0.43983436138792464},
            "contributions": {"evidence": 0.1320143460550091, "clip_similarity": 0.21991718069396232},
            "vlm_verdict": {"supported": False,
                            "raw": "NO. There are only two cats in the image, and they are sleeping, not eating."},
            "count_check": None, "consistency_uncertainty": float("nan")}


def test_supplied_case_refuted_with_unchanged_risk_and_count_diagnosis():
    before = supplied_case()
    result = finalize(before)
    assert result["verdict"] == "contradicted"
    assert result["final_decision"]["evidence_disagreement"]
    assert result["diagnosis"]["mechanism"] == "M2"
    assert "activity" in result["diagnosis"]["action"]
    assert result["risk"] == before["risk"]
    assert result["contributions"] == before["contributions"]
    assert before["verdict"] is None
    summary = summarize([result])
    assert summary["verdict"] == "contradicted"
    assert summary["counts"] == {"supported": 0, "contradicted": 1, "unresolved": 0}


def test_without_direct_verification_count_is_not_accepted():
    r = supplied_case()
    r["vlm_verdict"] = None
    r["risk"] = .01
    out = finalize(r)
    assert out["verdict"] == "unresolved" and out["diagnosis"]["repair"] != "accept"


def test_matching_count_cannot_confirm_eating():
    r = supplied_case()
    r["count_check"] = {"match": True}
    assert finalize(r)["verdict"] == "contradicted"
    r["vlm_verdict"] = None
    assert finalize(r)["verdict"] == "unresolved"


def test_conflicting_direct_checks_require_review():
    r = supplied_case()
    r["vlm_verdict"]["supported"] = True
    r["count_check"] = {"match": False}
    assert finalize(r)["verdict"] == "unresolved"


def test_detector_miss_alone_does_not_prove_falsehood():
    r = supplied_case()
    r["vlm_verdict"] = None
    r["count_check"] = {"match": False}
    assert finalize(r)["verdict"] == "unresolved"


def test_v3_unresolved_does_not_become_green_from_low_risk():
    r = supplied_case()
    r.update(claim_type="object", verdict="unresolved", vlm_verdict=None, risk=.01)
    assert finalize(r)["verdict"] == "unresolved"


def test_strict_json_roundtrip_preserves_decision_without_nan():
    out = finalize(supplied_case())
    data = json.loads(json.dumps(json_safe(out), allow_nan=False))
    assert data["consistency_uncertainty"] is None
    assert finalize(data)["verdict"] == "contradicted"
    assert finalize(data)["fusion_verdict"] is None


def test_missing_risk_roundtrip_does_not_crash():
    assert finalize({"risk": None, "risks": {"evidence": None}, "verdict": None})["verdict"] == "unresolved"


def test_clothing_refutation_routes_to_attribute_check_not_unrelated_relation():
    r = supplied_case()
    r.update(claim_text="the boy with the black colour dress", claim_type="attribute",
             risk=0.47375076526606624,
             risks={"evidence": 0.43183471533385187, "clip_similarity": 0.5156668151982806},
             vlm_verdict={"supported": False, "raw": "NO. The child is wearing a pink jacket, not a black dress."})
    out = finalize(r)
    assert out["verdict"] == "contradicted"
    assert out["diagnosis"]["mechanism"] == "M2"
    assert out["diagnosis"]["repair"] == "attribute verification"
    assert "blue cup" not in out["diagnosis"]["action"]
    assert out["risk"] == r["risk"]
    assert not any("Low embedding risk" in x for x in out["final_decision"]["reasons"])


def test_summary_unresolved_and_refuted_outrank_support():
    supported = finalize({"risk": .1, "claim_type": "object"})
    unresolved = finalize({"risk": .1, "claim_type": "count"})
    assert summarize([supported, unresolved])["verdict"] == "unresolved"
    assert summarize([supported, unresolved, finalize(supplied_case())])["verdict"] == "contradicted"
