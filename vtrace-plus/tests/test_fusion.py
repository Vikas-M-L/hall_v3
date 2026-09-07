#!/usr/bin/env python3
"""
Fusion smoke tests. Pure logic — no models, no GPU, no downloads.

    pytest tests/test_fusion.py -v
    python tests/test_fusion.py          # also works without pytest

This is the only part of the repo that can be verified without a GPU, so it
carries the weight: orientation, rule firing, renormalization, missing-signal
handling, and aggregation are all checked here with hand-computed expectations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusion.vtrace_fusion import (  # noqa: E402
    ALL_SIGNALS,
    BASE_WEIGHTS,
    CED_THRESHOLD,
    DISAGREEMENT_THRESHOLD,
    SignalBundle,
    aggregate,
    compute_ced,
    compute_disagreement,
    fuse,
    to_risk,
)

ALL_ON = {s: True for s in ALL_SIGNALS}
TOL = 1e-6


# =========================================================================
# Orientation — the failure mode that produces plausible-looking wrong answers
# =========================================================================


def test_orientation_inverts_the_right_signals():
    # "higher = more faithful" signals must come out as LOW risk.
    # Tolerance, not ==: 1.0 - 0.9 is 2 ULP below the literal 0.1 in IEEE-754.
    assert abs(to_risk("confidence", 0.9) - 0.1) < TOL
    assert abs(to_risk("evidence", 0.9) - 0.1) < TOL
    assert abs(to_risk("clip_similarity", 0.9) - 0.1) < TOL
    assert abs(to_risk("counterfactual", 0.9) - 0.1) < TOL
    # UniProbe already reports hallucination probability — pass through, exactly
    assert to_risk("uniprobe", 0.9) == 0.9


def test_orientation_clamps_and_propagates_nan():
    assert to_risk("evidence", 1.5) == 0.0
    assert to_risk("evidence", -0.5) == 1.0
    assert np.isnan(to_risk("evidence", float("nan")))
    assert np.isnan(to_risk("evidence", None))


def test_a_perfectly_grounded_claim_scores_near_zero():
    b = SignalBundle(confidence=1.0, evidence=1.0, clip_similarity=1.0,
                     uniprobe=0.0, counterfactual=1.0)
    for mode in ("v1", "v2"):
        assert fuse(b, mode=mode, active_signals=ALL_ON).risk < TOL


def test_a_maximally_bad_claim_scores_near_one():
    b = SignalBundle(confidence=0.0, evidence=0.0, clip_similarity=0.0,
                     uniprobe=1.0, counterfactual=0.0)
    for mode in ("v1", "v2"):
        assert fuse(b, mode=mode, active_signals=ALL_ON).risk > 1 - TOL


# =========================================================================
# v1 — fixed equal weights
# =========================================================================


def test_v1_is_a_plain_average_of_risks():
    b = SignalBundle(confidence=0.8, evidence=0.6, clip_similarity=0.5,
                     uniprobe=0.3, counterfactual=0.4)
    # risks: 0.2, 0.4, 0.5, 0.3, 0.6  -> mean 0.4
    r = fuse(b, mode="v1", active_signals=ALL_ON)
    assert abs(r.risk - 0.4) < TOL
    assert all(abs(w - 0.2) < TOL for w in r.weights.values())


def test_v1_ignores_the_rules_entirely():
    """Signal values that fire both v2 rules must not change v1's weights."""
    b = SignalBundle(confidence=0.93, evidence=0.30, clip_similarity=0.45,
                     uniprobe=0.85, counterfactual=0.80)
    r = fuse(b, mode="v1", active_signals=ALL_ON)
    assert all(abs(w - 0.2) < TOL for w in r.weights.values())


# =========================================================================
# Rule 1 — confident but ungrounded
# =========================================================================


def test_ced_is_native_orientation():
    assert abs(compute_ced(0.9, 0.3) - 0.6) < TOL
    assert np.isnan(compute_ced(float("nan"), 0.3))
    assert np.isnan(compute_ced(0.9, float("nan")))


def test_rule1_fires_and_shifts_weight_off_confidence():
    b = SignalBundle(confidence=0.95, evidence=0.35, clip_similarity=0.40,
                     uniprobe=0.55, counterfactual=0.20)
    r = fuse(b, mode="v2", active_signals=ALL_ON)

    assert r.ced > CED_THRESHOLD
    assert any("CED" in f for f in r.rules_fired)
    assert not any("disagreement" in f for f in r.rules_fired)

    assert r.weights["evidence"] > BASE_WEIGHTS["evidence"]
    assert r.weights["counterfactual"] > BASE_WEIGHTS["counterfactual"]
    assert r.weights["confidence"] < BASE_WEIGHTS["confidence"]
    # hand-computed: 0.08/1.15, 0.40/1.15
    assert abs(r.weights["confidence"] - 0.08 / 1.15) < 1e-4
    assert abs(r.weights["evidence"] - 0.40 / 1.15) < 1e-4


def test_rule1_does_not_fire_when_evidence_supports_confidence():
    b = SignalBundle(confidence=0.90, evidence=0.85, clip_similarity=0.80,
                     uniprobe=0.20, counterfactual=0.70)
    r = fuse(b, mode="v2", active_signals=ALL_ON)
    assert r.ced < CED_THRESHOLD
    assert r.rules_fired == ["none (near-equal base weighting)"]


# =========================================================================
# Rule 2 — detectors contradict each other
# =========================================================================


def test_disagreement_needs_two_signals():
    assert np.isnan(compute_disagreement({"evidence": 0.5}))
    assert not np.isnan(compute_disagreement({"evidence": 0.5, "uniprobe": 0.9}))


def test_stubbed_signals_do_not_count_as_disagreeing():
    """
    A stub returns the same constant for every claim, so its "distance" from a
    real detector is not disagreement — it is the distance to an arbitrary
    placeholder. With two of three detectors stubbed, only one real value is
    left and Rule 2 must go quiet.
    """
    risks = {"evidence": 0.9, "uniprobe": 0.5, "counterfactual": 0.5}
    assert not np.isnan(compute_disagreement(risks))
    assert np.isnan(compute_disagreement(risks, stubbed=("uniprobe", "counterfactual")))


def test_rule2_cannot_fire_off_placeholder_constants():
    """
    The shipped default has uniprobe and counterfactual at 0.5. Without the stub
    exclusion, evidence=0.35 (risk 0.65) against two 0.5s gives std 0.0707 —
    below threshold here, but the point is that the number describes the gap to a
    placeholder, not a conflict between detectors. Push evidence further and the
    unguarded version fires on a single signal.
    """
    b = SignalBundle(confidence=0.95, evidence=0.05, clip_similarity=0.40,
                     uniprobe=0.50, counterfactual=0.50,
                     stubbed=("uniprobe", "counterfactual"))
    r = fuse(b, mode="v2", active_signals=ALL_ON, drop_stubbed=False)
    assert np.isnan(r.disagreement)
    assert not any("disagreement" in f for f in r.rules_fired)
    # Rule 1 still fires — it does not depend on the stubs.
    assert any("CED" in f for f in r.rules_fired)


def test_a_reported_rule_always_actually_changed_the_weights():
    """
    `rules_fired` must describe effects, not intentions. If a rule is listed, the
    weights have to differ from plain renormalized BASE_WEIGHTS; if none is
    listed, they have to match it exactly.

    With the current gating this holds structurally — Rule 1 needs confidence and
    evidence available, and both are then weightable; Rule 2 needs two detectors
    available, likewise. The check is a guard for future rules that touch signals
    they do not depend on, where a trigger could be met with nothing to move.
    """
    rng = np.random.default_rng(7)
    for _ in range(300):
        vals = rng.random(5)
        toggles = {s: bool(rng.integers(0, 2)) for s in ALL_SIGNALS}
        if not any(toggles.values()):
            continue
        r = fuse(SignalBundle(**dict(zip(ALL_SIGNALS, vals))),
                 mode="v2", active_signals=toggles)
        if not r.weights:
            continue
        base_total = sum(BASE_WEIGHTS[s] for s in r.weights)
        untouched = {s: BASE_WEIGHTS[s] / base_total for s in r.weights}
        is_base = all(abs(r.weights[s] - untouched[s]) < 1e-12 for s in r.weights)
        reported = [f for f in r.rules_fired if not f.startswith("none")]
        assert is_base == (not reported), (
            f"rules_fired={r.rules_fired} but weights "
            f"{'match' if is_base else 'differ from'} base"
        )


def test_rule2_fires_and_promotes_clip_as_tiebreaker():
    b = SignalBundle(confidence=0.60, evidence=0.85, clip_similarity=0.50,
                     uniprobe=0.90, counterfactual=0.15)
    r = fuse(b, mode="v2", active_signals=ALL_ON)

    assert r.disagreement > DISAGREEMENT_THRESHOLD
    assert any("disagreement" in f for f in r.rules_fired)
    assert not any("CED" in f for f in r.rules_fired)

    assert r.weights["clip_similarity"] > BASE_WEIGHTS["clip_similarity"]
    for s in ("evidence", "uniprobe", "counterfactual"):
        assert r.weights[s] < BASE_WEIGHTS[s]
    assert abs(r.weights["clip_similarity"] - 0.30 / 0.89) < 1e-4


def test_both_rules_can_fire_together():
    b = SignalBundle(confidence=0.93, evidence=0.30, clip_similarity=0.45,
                     uniprobe=0.85, counterfactual=0.80)
    r = fuse(b, mode="v2", active_signals=ALL_ON)
    assert len(r.rules_fired) == 2
    assert any("CED" in f for f in r.rules_fired)
    assert any("disagreement" in f for f in r.rules_fired)


# =========================================================================
# Invariants that must hold for every input
# =========================================================================


def test_weights_and_contributions_stay_consistent():
    rng = np.random.default_rng(0)
    for _ in range(500):
        vals = rng.random(5)
        b = SignalBundle(**dict(zip(ALL_SIGNALS, vals)))
        for mode in ("v1", "v2"):
            r = fuse(b, mode=mode, active_signals=ALL_ON)
            assert abs(sum(r.weights.values()) - 1.0) < 1e-9
            assert abs(sum(r.contributions.values()) - r.risk) < 1e-9
            assert 0.0 <= r.risk <= 1.0
            assert all(w > 0 for w in r.weights.values())


def test_missing_signals_are_dropped_and_weights_renormalize():
    b = SignalBundle(confidence=0.70, clip_similarity=0.55)  # other three nan
    r = fuse(b, mode="v2", active_signals=ALL_ON)

    assert set(r.weights) == {"confidence", "clip_similarity"}
    assert len(r.excluded) == 3
    assert abs(sum(r.weights.values()) - 1.0) < TOL
    # base 0.20 / 0.15 renormalized over 0.35
    assert abs(r.weights["confidence"] - 0.20 / 0.35) < TOL
    # both rules disabled: CED needs evidence, disagreement needs 2 detectors
    assert np.isnan(r.ced) and np.isnan(r.disagreement)


def test_disabled_signals_are_excluded():
    b = SignalBundle(confidence=0.8, evidence=0.6, clip_similarity=0.5,
                     uniprobe=0.3, counterfactual=0.4)
    toggles = {**ALL_ON, "uniprobe": False}
    r = fuse(b, mode="v1", active_signals=toggles)
    assert "uniprobe" not in r.weights
    assert any("uniprobe" in e for e in r.excluded)
    assert all(abs(w - 0.25) < TOL for w in r.weights.values())


def test_a_disabled_signal_cannot_still_drive_rule_1():
    """
    An excluded signal must not steer the weighting of the ones that remain.
    These values fire Rule 1 hard (CED = 0.60), but with `confidence` disabled
    the rule has no licence to run at all.
    """
    b = SignalBundle(confidence=0.95, evidence=0.35, clip_similarity=0.40,
                     uniprobe=0.55, counterfactual=0.20)
    r = fuse(b, mode="v2", active_signals={**ALL_ON, "confidence": False})

    assert r.ced > CED_THRESHOLD          # still reported as a diagnostic
    assert not any("CED" in f for f in r.rules_fired)
    # weights are just BASE_WEIGHTS renormalized over the remaining four
    remaining = sum(BASE_WEIGHTS[s] for s in r.weights)
    for s in r.weights:
        assert abs(r.weights[s] - BASE_WEIGHTS[s] / remaining) < TOL


def test_stubbed_signals_can_be_dropped():
    b = SignalBundle(confidence=0.8, evidence=0.6, clip_similarity=0.5,
                     uniprobe=0.5, counterfactual=0.5,
                     stubbed=("uniprobe", "counterfactual"))
    kept = fuse(b, mode="v1", active_signals=ALL_ON, drop_stubbed=False)
    dropped = fuse(b, mode="v1", active_signals=ALL_ON, drop_stubbed=True)
    assert len(kept.weights) == 5
    assert set(dropped.weights) == {"confidence", "evidence", "clip_similarity"}


def test_no_usable_signals_yields_nan_not_a_confident_zero():
    r = fuse(SignalBundle(), mode="v2", active_signals=ALL_ON)
    assert np.isnan(r.risk)
    assert r.weights == {}
    assert "No usable signals" in r.explain()


def test_bad_mode_is_rejected():
    try:
        fuse(SignalBundle(confidence=0.5), mode="v3", active_signals=ALL_ON)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown fusion mode")


# =========================================================================
# Monotonicity — v1 is monotone; v2 is NOT, and that is worth knowing
# =========================================================================


def _sweep(mode: str, n: int = 400) -> list[float]:
    """Risk as evidence sweeps 0 -> 1 with every other signal held fixed."""
    out = []
    for e in np.linspace(0.0, 1.0, n):
        b = SignalBundle(confidence=0.9, evidence=float(e), clip_similarity=0.5,
                         uniprobe=0.5, counterfactual=0.5)
        out.append(fuse(b, mode=mode, active_signals=ALL_ON).risk)
    return out


def test_v1_risk_decreases_monotonically_as_evidence_improves():
    risks = _sweep("v1")
    assert all(b <= a + 1e-12 for a, b in zip(risks, risks[1:]))


def test_v2_monotonicity_violation_is_bounded():
    """
    KNOWN BEHAVIOUR, not a bug being hidden: v2's hard thresholds make risk
    discontinuous in the signal values, so better evidence can briefly *raise*
    risk as a rule switches on or off.

    Along this sweep (confidence 0.9, the other three at 0.5) Rule 1 fires for
    e < 0.65 and Rule 2 for e < 0.0757 or e > 0.9243, giving four regimes. Risk
    falls monotonically inside each, so violations occur only at the boundaries:

        e ~ 0.0757  Rule 2 switches OFF   +0.0440   <- largest
        e ~ 0.6500  Rule 1 switches OFF   -0.0382   (downward, fine)
        e ~ 0.9243  Rule 2 switches ON    +0.0241

    The largest jump is where `evidence` stops being the high-risk outlier among
    the detectors: Rule 2 had been damping it, Rule 2 turns off, and the now
    full-weight evidence term dominates. The bound below is a regression guard —
    if a constant change makes the discontinuity much larger, this fails.
    """
    risks = _sweep("v2")
    worst = max((b - a for a, b in zip(risks, risks[1:])), default=0.0)
    assert worst > 0, "expected v2 to be non-monotone; did the rules change?"
    assert worst < 0.10, f"monotonicity violation grew to {worst:.4f}"


# =========================================================================
# Aggregation
# =========================================================================


def test_aggregation_methods():
    risks = [0.1, 0.9, 0.5, 0.3]
    assert abs(aggregate(risks, "max") - 0.9) < TOL
    assert abs(aggregate(risks, "mean") - 0.45) < TOL
    assert abs(aggregate(risks, "topk_mean", topk=2) - 0.7) < TOL
    # k larger than the list is clamped, not an error
    assert abs(aggregate(risks, "topk_mean", topk=99) - 0.45) < TOL


def test_aggregation_skips_nans_and_survives_an_empty_list():
    assert abs(aggregate([0.2, float("nan"), 0.8], "max") - 0.8) < TOL
    assert np.isnan(aggregate([], "max"))
    assert np.isnan(aggregate([float("nan")], "max"))


def test_unknown_aggregation_method_is_rejected():
    try:
        aggregate([0.5], "median")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown aggregation method")


# =========================================================================


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
