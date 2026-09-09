"""Semantic/evaluation invariants, no downloaded weights or API credentials."""
import dataclasses
import json

import numpy as np
import pytest

from research.backends import ReplayBackend
from research.calibration import fit_calibrator, transform
from research.core import Case, canonical_hash, run_case, validate_cases, validate_detection, write_jsonl
from research.statistics import cluster_bootstrap, metrics, paired_comparison, repair_metrics, selective
from research.prepare_dataset import prepare


def case(**changes):
    return dataclasses.replace(Case("x", "scene1", "x.jpg", "a"*64, "How many cats?",
                                    "There are three cats.", "fixture"), **changes)


class Fake:
    def __init__(self, verified=True, relevant=True, kind="counting", risk=.9):
        self.calls = []
        self.verified, self.relevant, self.kind, self.risk = verified, relevant, kind, risk

    def call(self, stage, inputs, action):
        assert not {"label", "correct_answer", "hallucinated_answer", "gold"} & set(inputs)
        self.calls.append((stage, inputs, action))
        if stage == "detect":
            return {"claims": [{"text": "There are three cats.", "source": "three cats",
                                "type": self.kind, "risk": self.risk,
                                "verdict": "contradicted" if self.risk > .5 else "supported"}]}
        if stage == "repair":
            return {"response": "There are two cats."}
        return {"supported": self.verified, "answers_question": self.relevant}


def test_gold_rejected_at_model_boundary():
    with pytest.raises(ValueError, match="gold belongs separately"):
        Case.parse({**dataclasses.asdict(case()), "correct_answer": "two"})


def test_groups_and_image_hashes_cannot_cross_partitions():
    with pytest.raises(ValueError, match="crosses"):
        validate_cases([case(split="train"), case(id="y", split="test")])
    with pytest.raises(ValueError, match="crosses"):
        validate_cases([case(split="train"), case(id="y", group_id="other", split="test")])


def test_original_performs_no_model_calls():
    backend = Fake()
    out = run_case(case(), backend, "original")
    assert out["final_response"] == case().response and backend.calls == []


def test_full_executes_repair_and_independent_verification():
    backend = Fake()
    out = run_case(case(), backend)
    assert [s for s, _, _ in backend.calls] == ["detect", "repair", "verify"]
    assert out["action"] == "recount" and out["status"] == "accepted_repair"
    assert out["detection"]["claims"][0]["span"] == [10, 20]
    assert "evidence" not in backend.calls[-1][1]
    assert backend.calls[-1][1]["question"] == case().question


@pytest.mark.parametrize("supported,relevant", [(False, True), (None, True), (True, False), (True, None)])
def test_grounding_and_relevance_both_required(supported, relevant):
    out = run_case(case(), Fake(supported, relevant))
    assert out["status"] == "rejected_repair_abstained" and out["final_response"] is None
    assert out["original_response"] == case().response


def test_knowledge_without_external_evidence_abstains():
    backend = Fake(kind="knowledge")
    out = run_case(case(), backend)
    assert out["status"] == "abstained" and len(backend.calls) == 1


def test_no_verifier_ablation_is_explicit():
    backend = Fake()
    out = run_case(case(), backend, "routed_no_verifier")
    assert out["status"] == "accepted_unverified" and len(backend.calls) == 2


def test_no_diagnosis_does_not_supply_types_or_reasons_to_repair():
    backend = Fake()
    run_case(case(), backend, "no_diagnosis")
    claims = backend.calls[1][1]["evidence"]["claims"]
    assert all("type" not in c and "reason" not in c for c in claims)


def test_missing_replay_is_failure_not_synthetic_success(tmp_path):
    p = tmp_path / "records.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="missing recorded"):
        run_case(case(), ReplayBackend(p))


def test_replay_roundtrip_and_path_independence(tmp_path):
    inputs = {"image": "first.jpg", "image_sha256": "a"*64, "question": "Q", "response": "A"}
    key = ReplayBackend.key("verify", inputs, "none")
    p = tmp_path / "records.jsonl"
    write_jsonl(p, [{"key": key, "output": {"supported": False, "answers_question": True}}])
    backend = ReplayBackend(p)
    assert backend.call("verify", {**inputs, "image": "other.jpg"}, "none")["supported"] is False
    assert backend.events[0]["network_calls"] == 0


def test_nan_detection_never_accepted():
    d = Fake(risk=float("nan")).call("detect", {}, "none")
    with pytest.raises(ValueError):
        validate_detection(d)


def test_metrics_no_zero_imputation_and_degenerate_auc():
    assert metrics([0, 0], [.1, .2])["auroc"] is None
    with pytest.raises(ValueError):
        metrics([0, 1], [.1, float("nan")])


def test_abstain_everything_cannot_count_as_corrected_answers():
    rows = [{"original_hallucinated": bool(i%2), "original_correct": not bool(i%2),
             "abstained": True, "final_hallucinated": None, "final_correct": None} for i in range(8)]
    m = repair_metrics(rows)
    assert m["hallucination_reduction_including_abstention"] == 1
    assert m["correction_success_rate"] == 0 and m["answer_preservation"] == 0
    assert m["final_answer_accuracy"] == 0 and m["net_corrected_minus_correct_lost"] == -4


def test_selective_accuracy_reports_denominator():
    m = selective([0, 1, 1], [.1, .9, .2], ["supported", "contradicted", "unresolved"])
    assert m["decided"] == 2 and m["coverage"] == 2/3 and m["selective_accuracy"] == 1


def test_bootstrap_resamples_whole_image_blocks():
    def stat(idx):
        assert (0 in idx) == (1 in idx)
        assert list(idx).count(0) == list(idx).count(1)
        return len(idx)
    result = cluster_bootstrap(["a", "a", "b"], stat, repeats=30)
    assert result["clusters"] == 2 and result["valid_resamples"] == 30


def test_identical_baselines_have_zero_delta_and_p_one():
    r = paired_comparison([0, 1, 0, 1], [.1, .9, .3, .7], [.1, .9, .3, .7],
                          ["a", "a", "b", "b"], repeats=40)
    assert r["ci95"] == [0, 0] and r["paired_cluster_randomization_p"] == 1


def test_calibration_refuses_test_and_overlap():
    rows = [{"id": str(i), "group_id": str(i), "split": "calibration",
             "label": i%2, "score": .2 if i%2 == 0 else .8} for i in range(10)]
    b = fit_calibrator(rows)
    with pytest.raises(ValueError, match="overlap"):
        transform(b, rows)
    with pytest.raises(ValueError, match="calibration partition"):
        fit_calibrator([{**r, "split": "test"} for r in rows])


def test_annotation_export_separates_gold_and_preserves_official_partition():
    row = {**dataclasses.asdict(case()), "correct_answer": "two", "original_hallucinated": True,
           "original_correct": False, "hallucination_type": "counting", "hallucinated_spans": [[10,15]],
           "visual_evidence": [], "severity": "material", "corrected_answer": "There are two cats.",
           "source_dataset": "fixture", "source_id": "1", "source_license": "fixture-only",
           "generator_model": "fixture", "generator_revision": "1", "prompt_id": "1",
           "annotation_status": "adjudicated", "annotator_ids": ["A", "B"], "adjudicator": "C",
           "official_partition": "test"}
    inputs, gold = prepare([row], "frozen-salt")
    assert inputs[0]["split"] == "test" and "correct_answer" not in inputs[0]
    assert gold[0]["correct_answer"] == "two"
    with pytest.raises(ValueError, match="development-exposed"):
        prepare([row], "frozen-salt", [row["group_id"]])
