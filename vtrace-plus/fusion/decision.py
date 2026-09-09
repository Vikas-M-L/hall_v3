"""Final evidence decision, separate from the uncalibrated embedding risk.

Model refutation is evidence, not ground truth. Conflicting direct checks or
unsupported claim types go to review; numeric risk/contributions stay intact.
"""
from __future__ import annotations

import math

from fusion.diagnose import diagnose
from fusion.negation import has_negation

PRESENTATION = {
    "supported": ("SUPPORTED", "badge-ok", "banner-safe"),
    "contradicted": ("CONTRADICTED", "badge-bad", "banner-risky"),
    "unresolved": ("UNRESOLVED — REVIEW", "badge-warn", "banner-review"),
}
SPECIALIST_TYPES = {"count", "counting", "action", "spatial", "relation", "ocr"}


def resolve(record):
    """Consume existing result JSON, including optional VLM/count checks."""
    risk = record.get("risk")
    vv = record.get("vlm_verdict") or {}
    cc = record.get("count_check") or {}
    vlm = vv.get("supported")
    count = cc.get("match")
    kind = record.get("claim_type", "object")
    reasons = []
    sources = []
    if type(vlm) is bool:
        sources.append("vlm_crosscheck")
        reasons.append("VLM cross-check " + ("supports" if vlm else "refutes") + " the claim.")
    if type(count) is bool:
        sources.append("object_counter")
        reasons.append("Object counter " + ("matches" if count else "disagrees with") + " the claimed count.")
    if vlm is True and count is False:
        verdict = "unresolved"
        reasons.append("Direct verification checks conflict; inspect the image manually.")
    elif vlm is False:
        # A matching count cannot validate action/attribute clauses in a compound
        # claim. Refutation of the complete assertion takes precedence.
        verdict = "contradicted"
        reasons.append("Embedding compatibility alone cannot override an explicit refutation.")
    elif count is False:
        verdict = "unresolved"
        reasons.append("Detector mismatch needs recount/review; missed or duplicate boxes are possible.")
    elif vlm is True:
        if record.get("verdict") == "contradicted":
            verdict = "unresolved"
            reasons.append("VLM and pairwise verifier disagree; review rather than accept.")
        else:
            verdict = "supported"
    elif kind in SPECIALIST_TYPES or record.get("decomposition") == "requires_full_claim_check":
        verdict = "unresolved"
        reasons.append(f"Embedding similarity alone does not verify {kind} claims.")
        if count is True:
            reasons.append("Count agreement does not verify the rest of a compound claim.")
        reasons.append("Run a full-claim visual cross-check or inspect the claim manually.")
    elif record.get("verdict") in PRESENTATION:
        verdict = record["verdict"]
        sources.append("pairwise_verifier")
        reasons.extend(record.get("reasons") or [])
    elif isinstance(risk, (int, float)) and not isinstance(risk, bool) and math.isfinite(risk):
        sources.append("embedding_heuristic")
        verdict = "supported" if risk < .4 else "unresolved"
        reasons.append("Embedding-only compatibility estimate; this is not a factual guarantee.")
    else:
        verdict = "unresolved"
        reasons.append("No usable verification result.")
    action = {"supported": "accept_with_evidence", "contradicted": "flag_and_recheck",
              "unresolved": "request_verification"}[verdict]
    return {"verdict": verdict, "reasons": reasons, "sources": sources, "action": action,
            "evidence_disagreement": verdict == "contradicted" and isinstance(risk, (int, float)) and risk < .4,
            "note": "Decision reflects available model evidence, not independently established ground truth."}


def finalize(record):
    """Single source of truth for Checker, table, JSON and Autopsy."""
    out = dict(record)
    # Preserve v3 result independently; final verdict must not overwrite its
    # pairwise reasoning or modify risk/contribution arithmetic.
    out["fusion_verdict"] = record.get("fusion_verdict", record.get("verdict"))
    out["final_decision"] = resolve({**record, "verdict": out["fusion_verdict"]})
    out["verdict"] = out["final_decision"]["verdict"]
    rr = record.get("risks") or {}
    vv, cc = record.get("vlm_verdict") or {}, record.get("count_check") or {}
    def numeric(value):
        return float(value) if isinstance(value, (int, float)) else float("nan")
    out["diagnosis"] = diagnose(numeric(record.get("risk")),
                                numeric(rr.get("evidence")), numeric(rr.get("clip_similarity")),
                                claim_type=record.get("claim_type", "object"),
                                negated=has_negation(record.get("claim_text", record.get("text", ""))),
                                uncertainty=record.get("consistency_uncertainty"),
                                count_match=cc.get("match"), vlm_supported=vv.get("supported"))
    if out["verdict"] == "unresolved" and out["diagnosis"]["mechanism"] == "M0":
        out["diagnosis"] = {"mechanism": "M?", "name": "Unverified claim",
                            "repair": "full-claim verification", "action": "Verify each count, action and relation separately.",
                            "cost": "not measured", "reasons": out["final_decision"]["reasons"]}
    return out


def summarize(records):
    counts = {v: sum(r["final_decision"]["verdict"] == v for r in records) for v in PRESENTATION}
    verdict = "contradicted" if counts["contradicted"] else (
        "unresolved" if counts["unresolved"] or not records else "supported")
    return {"verdict": verdict, "counts": counts, "total": len(records)}


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value
