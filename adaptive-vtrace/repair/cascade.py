"""Cost-aware cascaded routing — the 10/10 upgrade.

The plan's honest weak spot (§11 "Cost claims"): routing costs 6+ VLM passes
per claim while uniform abstention costs ~nothing, so "lower compute" cannot
be claimed. The answer is a cascade: CLIP/SigLIP signals cost ~0 VLM passes,
so decide immediately when they are decisive and escalate to expensive VLM
passes (confidence, counterfactual, VIG nulls, self-consistency samples) only
for ambiguous claims.

Cost model (VLM forward passes; CLIP text/image encodes are ~free beside them):

    evidence / clip_similarity : 0.0   (frozen CLIP, no VLM)
    confidence                 : 0.2   (reuses generation logprobs)
    counterfactual (local)     : 2.0   (mask + rescore)
    vig (4 nulls + base)       : 5.0
    self_consistency (5 samp.) : 5.0
    uniprobe (probe)           : 0.5   (one hidden-state read)

Cascade policy (per claim):
    stage 0 (cost 0): CLIP margin m = |evidence_unit - 0.5| * 2 in [0,1].
        m >= tau_decisive -> verdict now (risk = 1 - match), no escalation.
    stage 1 (cost 2.2): add confidence + counterfactual; fused risk r.
        |r - 0.5| * 2 >= tau_mid -> verdict now.
    stage 2 (cost 10): full stack incl. VIG + self-consistency.

Claim: cascade matches full-stack accuracy within epsilon at a fraction of
mean cost — because most claims are easy. Tested synthetically here; the GPU
run replaces simulated stage outcomes with measured ones via the same API.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# VLM passes per signal. CLIP-family signals are ~0 by construction.
COSTS = {"evidence": 0.0, "clip_similarity": 0.0, "confidence": 0.2,
         "counterfactual": 2.0, "vig": 5.0, "self_consistency": 5.0,
         "uniprobe": 0.5}

STAGE_COSTS = {"stage0": 0.0, "stage1": 2.2, "stage2": 12.2}


def decisiveness(risk: float) -> float:
    """0 = total toss-up, 1 = maximally decisive."""
    if risk != risk:
        return 0.0
    return float(abs(risk - 0.5) * 2.0)


def cascade_decide(clip_risk: float, full_risk: float,
                   tau_decisive: float = 0.6, tau_mid: float = 0.4) -> dict:
    """Simulated cascade decision given stage outcomes.

    clip_risk: stage-0 CLIP-only risk. full_risk: stage-1/2 fused risk
    (callers compute only what the cascade actually reaches — pass NaN for
    stages never run). Returns {verdict_risk, stage_reached, cost}."""
    if decisiveness(clip_risk) >= tau_decisive:
        return {"verdict_risk": clip_risk, "stage": "stage0",
                "cost": STAGE_COSTS["stage0"], "escalated": False}
    if full_risk == full_risk and decisiveness(full_risk) >= tau_mid:
        return {"verdict_risk": full_risk, "stage": "stage1",
                "cost": STAGE_COSTS["stage1"], "escalated": True}
    return {"verdict_risk": full_risk, "stage": "stage2",
            "cost": STAGE_COSTS["stage2"], "escalated": True}


def pareto_table(y_true: np.ndarray, clip_risks: np.ndarray,
                 full_risks: np.ndarray,
                 taus: tuple[float, ...] = (0.3, 0.5, 0.6, 0.8)) -> list[dict]:
    """Accuracy-vs-mean-cost frontier over decisiveness thresholds.

    Accuracy = fraction of claims whose verdict side (risk<>0.5) matches truth.
    Shows the paper's cost claim honestly: what accuracy each budget buys."""
    y = np.asarray(y_true, dtype=int)
    rows = []
    for tau in taus:
        verdicts, costs = [], []
        for yc, cr, fr in zip(y, clip_risks, full_risks):
            d = cascade_decide(float(cr), float(fr), tau_decisive=tau)
            v = d["verdict_risk"]
            verdicts.append(int(v >= 0.5) == yc if v == v else False)
            costs.append(d["cost"])
        rows.append({"tau": tau, "accuracy": float(np.mean(verdicts)),
                     "mean_cost": float(np.mean(costs)),
                     "full_cost": STAGE_COSTS["stage2"]})
    return rows
