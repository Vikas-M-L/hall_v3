"""Rule-based mechanism diagnosis + repair routing (inference-time, CPU-only).

Positioning: V-TRACE+ is a diagnosis and repair-routing system —
Detect -> Explain -> Prove -> Route. This module is the Explain + Route step
for the live product. The *trained* two-stage classifier
(adaptive-vtrace/fusion/two_stage.py) is the research-grade version; this
rule-based one is transparent, instant, and always shows its reasons.

Taxonomy: M0 no fault · M1 prior override · M2 perceptual/count failure ·
M3 unsurfaced uncertainty · M4 binding/relation failure.

Repair router (cheap first):
    M0 -> accept (no action)
    M1 -> image-first re-prompt (cheap: new VLM call, no new signals)
    M2 -> detector recount / zoom re-query (cheap: OWL-ViT already loaded)
    M3 -> abstain or hedge + report uncertainty (free)
    M4 -> relation verification: VLM YES/NO on the isolated pair (1 call)
    count mismatch -> OWL-ViT recount (cheap)
    ocr/spatial claim + mid risk -> grounding re-check (cheap)

Every diagnosis returns its triggering reasons — never a bare label.
"""
from __future__ import annotations

MECHANISMS = ("M0", "M1", "M2", "M3", "M4")

MECHANISM_NAMES = {
    "M0": "No fault",
    "M1": "Prior override",
    "M2": "Perceptual / counting failure",
    "M3": "Unsurfaced uncertainty",
    "M4": "Binding / relation failure",
}

REPAIR_FOR = {
    "M0": ("accept", "No action — claim stands.", "none"),
    "M1": ("image-first re-prompt", "Re-ask the VLM with the image first and a "
           "narrow question; priors lose to fresh pixels.", "cheap"),
    "M2": ("detector recount / zoom re-query", "Re-check with OWL-ViT boxes or a "
           "cropped zoom; perception, not knowledge, failed.", "cheap"),
    "M3": ("abstain or hedge", "Report uncertainty instead of answering; "
           "do not decode further.", "free"),
    "M4": ("relation verification", "Identify the entities in this claim and "
           "verify their stated relationship separately from their existence.", "1 VLM call"),
}


def diagnose(risk: float, evidence_risk: float, similarity_risk: float,
             claim_type: str = "object", negated: bool = False,
             uncertainty: float | None = None,
             count_match: bool | None = None,
             vlm_supported: bool | None = None) -> dict:
    """Return {mechanism, name, repair, action, cost, reasons}.

    All inputs are risk-oriented ([0,1], higher = worse) except uncertainty
    (0..1 disagreement) which may be None when unsampled. NaN risk -> M0 with
    low confidence is WRONG, so NaN yields an explicit 'unknown' instead.
    """
    reasons: list[str] = []
    if risk != risk:
        return {"mechanism": "M?", "name": "Unknown (no usable signals)",
                "repair": "re-run verification", "action": "re-verify",
                "cost": "free", "reasons": ["all signals NaN"]}

    # Hard evidence channels first — they overrule heuristics.
    if count_match is False:
        reasons.append("detector count disagrees with claimed number")
        return _out("M2", reasons)
    if vlm_supported is False:
        reasons.append("VLM directly refuted the claim")
        if claim_type in ("count", "counting"):
            reasons.append("Count-bearing claim was refuted; verify both the number and any accompanying action.")
            result = _out("M2", reasons)
            result["action"] = "Recount the named objects and separately verify the claimed activity; the VLM explanation is evidence, not ground truth."
            return result
        if claim_type == "attribute":
            reasons.append("An attribute claim was refuted; similarity alone cannot identify the cause.")
            result = _out("M2", reasons)
            result["name"] = "Possible perceptual / attribute error"
            result["repair"] = "attribute verification"
            result["action"] = "Check the named object's visible property (such as color or clothing type) separately. Confirm any proposed correction against the image."
            return result
        # A relation hypothesis needs a relation-bearing claim, not merely
        # strong image/text compatibility. It is still not causal proof.
        if claim_type in ("spatial", "relation"):
            reasons.append("A relation-bearing claim was refuted; binding failure is a hypothesis requiring a focused check.")
            return _out("M4", reasons)
        reasons.append("Claim refuted; a perceptual error is a hypothesis, not an established cause.")
        return _out("M2", reasons)

    if risk < 0.4:
        reasons.append(f"risk {risk:.2f} below action threshold")
        return _out("M0", reasons)

    if (isinstance(uncertainty, (int, float)) and uncertainty == uncertainty
            and uncertainty > 0.5):
        reasons.append(f"sampler disagreement {uncertainty:.2f} despite decisive-ish score")
        return _out("M3", reasons)

    # CLIP split: regions match but whole doesn't (or vice versa) = parts vs
    # arrangement disagree -> binding suspect, esp. for relation-ish claims.
    if evidence_risk == evidence_risk and similarity_risk == similarity_risk:
        gap = abs(evidence_risk - similarity_risk)
        if gap > 0.25 and claim_type in ("relation", "spatial", "attribute", "action"):
            reasons.append(f"region/whole CLIP gap {gap:.2f} on a {claim_type} claim: "
                           "parts match, composition may not")
            return _out("M4", reasons)

    if negated:
        reasons.append("negated claim with residual risk after inversion")
        return _out("M1", reasons)

    if claim_type in ("count",) or count_match is not None:
        reasons.append("count-flavored claim above threshold")
        return _out("M2", reasons)

    # Default for confident-looking but unsupported: prior override.
    reasons.append(f"risk {risk:.2f} with no uncertainty/counter-evidence on record: "
                   "model asserted beyond its grounding")
    return _out("M1", reasons)


def _out(mech: str, reasons: list[str]) -> dict:
    repair, action, cost = REPAIR_FOR[mech]
    return {"mechanism": mech, "name": MECHANISM_NAMES[mech], "repair": repair,
            "action": action, "cost": cost, "reasons": reasons}
