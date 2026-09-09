"""Studio analysis orchestration. Raw scores and final evidence decisions stay separate."""
from __future__ import annotations

import re
import time
from collections import Counter

from fusion.decision import finalize, json_safe, summarize
from fusion.negation import apply_negation, make_counterclaim
from fusion.vtrace_fusion import SignalBundle, aggregate, fuse
from models.atomic_claims import extract_facts
from models.owl_wrapper import count_verdict, parse_count

ACTIVE = {"confidence": False, "evidence": True, "clip_similarity": True,
          "uniprobe": False, "counterfactual": False}


def overlap(claim, sample):
    a = Counter(re.findall(r"[a-z0-9]+", claim.lower()))
    b = Counter(re.findall(r"[a-z0-9]+", sample.lower()))
    return 2*sum((a & b).values())/(sum(a.values())+sum(b.values())) if a and b else 0.


def analyze(image, response, cfg, clip, gemini=None, detector=None):
    """Injected backends enable actual end-to-end tests without network spend.

    cfg selects explicit API use, decomposition and cross-check budget. Errors
    are surfaced in the saved result; missing observations are not fabricated.
    """
    start, events, warnings = time.perf_counter(), [], []

    def call(stage, fn):
        t = time.perf_counter()
        try:
            out = fn()
        except Exception as exc:
            events.append({"stage": stage, "success": False, "seconds": time.perf_counter()-t})
            warnings.append(f"{stage} failed ({type(exc).__name__}); evidence unavailable.")
            return None
        events.append({"stage": stage, "success": True, "seconds": time.perf_counter()-t})
        return out

    if cfg.get("generate"):
        if gemini is None:
            raise ValueError("Gemini backend is required to generate a description")
        response = call("generate", lambda: gemini.describe(image, cfg["question"]))
        if not response:
            raise ValueError("Description unavailable; enter text manually or retry")
    if not response or not response.strip():
        raise ValueError("Enter claims or enable Gemini description generation")
    entries = None
    if cfg.get("api_decomposition") and gemini is not None:
        entries = call("decompose", lambda: gemini.decompose(response))
    facts = extract_facts(response, entries=entries)
    if not facts:
        raise ValueError("No claims extracted")
    max_claims = cfg.get("max_claims", 20)
    if len(facts) > max_claims:
        raise ValueError(f"Extracted {len(facts)} facts; limit is {max_claims}. Shorten the text; no claims were silently dropped.")
    samples = []
    if cfg.get("consistency") and gemini is not None:
        samples = call("sample", lambda: gemini.sample(image, cfg["question"], n=3, temperature=.7)) or []
    t = time.perf_counter()
    regions = clip.encode_regions(image, grid=cfg["grid"])
    records = []
    for fact in facts:
        s, box, whole = clip.evidence(fact["text"], regions)
        sim = clip.similarity(fact["text"], regions)
        pw = None
        if cfg["fusion_mode"] == "v3":
            counter = make_counterclaim(fact["text"])
            pw = clip.pairwise_scores(fact["text"], counter["counterclaim_text"], regions)
            pw["counterclaim_text"] = counter["counterclaim_text"]
        b = SignalBundle(claim_id=fact["claim_id"], claim_text=fact["text"],
                         claim_type=fact["claim_type"], evidence=s, clip_similarity=sim)
        r = fuse(b, mode=cfg["fusion_mode"], active_signals=ACTIVE, drop_stubbed=True, pairwise=pw)
        if cfg["fusion_mode"] != "v3":
            corr, neg = apply_negation(fact["text"], r.risks.get("evidence", float("nan")),
                                       r.risks.get("clip_similarity", float("nan")))
            if neg and corr == corr:
                r.risk = corr
                r.rules_fired.append("negation-inverted (legacy heuristic)")
        record = {**fact, **r.to_dict(), "vlm_verdict": None, "count_check": None,
                  "detector_evidence": None,
                  "visual_evidence": {"source": "similarity_crop", "box": None if whole else box,
                                      "whole_image_won": whole, "checkpoint": cfg["clip_model"],
                                      "note": "Matching crop, not an object bounding box or proof."},
                  "consistency_uncertainty": 1-sum(overlap(fact["text"], s) for s in samples)/len(samples) if samples else None}
        records.append(record)
    events.append({"stage": "local_scoring", "success": True, "seconds": time.perf_counter()-t})

    if detector is not None:
        labels = {}
        expected = {}
        for r in records:
            n, lab = parse_count(r["text"])
            if n is not None:
                expected[r["claim_id"]] = (n, lab)
            label = r.get("detector_label") or lab
            if label:
                labels[r["claim_id"]] = label
        if labels:
            detected = call("object_detection", lambda: detector.detect(image, sorted(set(labels.values()))))
            if detected is not None:
                for r in records:
                    label = labels.get(r["claim_id"])
                    observation = detected.get(label) if label else None
                    if observation is not None:
                        r["detector_evidence"] = {"label": label, **observation}
                        if r["claim_id"] in expected:
                            n, lab = expected[r["claim_id"]]
                            r["count_check"] = {"expected_num": n, "expected_label": lab,
                                                **observation, **count_verdict(n, lab, observation)}
    if gemini is not None and cfg.get("crosscheck"):
        priority = {"count": 0, "action": 1, "attribute": 2, "spatial": 3, "relation": 3, "ocr": 4}
        candidates = [r for r in records if cfg.get("verify_all") or r["claim_type"] in priority
                      or r["decomposition"] == "requires_full_claim_check"
                      or (r["risk"] == r["risk"] and .35 <= r["risk"] < .65)
                      or r.get("verdict") == "unresolved"]
        candidates.sort(key=lambda r: priority.get(r["claim_type"], 5))
        limit = len(records) if cfg.get("verify_all") else cfg.get("crosscheck_limit", 3)
        for r in candidates[:limit]:
            r["vlm_verdict"] = call("crosscheck", lambda r=r: gemini.verify(image, r["text"]))
        if len(candidates) > limit:
            warnings.append(f"Cross-check budget reached: {len(candidates)-limit} candidate facts were not sent to the VLM.")
    finalized = [finalize(r) for r in records]
    summary = summarize(finalized)
    return json_safe({"response": response, "question": cfg["question"], "config": cfg,
                      "image_risk": aggregate([r["risk"] for r in records], method=cfg["aggregation"], topk=3),
                      "fusion_mode": cfg["fusion_mode"], "aggregation": cfg["aggregation"],
                      "final_verdict": summary["verdict"], "decision_counts": summary["counts"],
                      "claims": finalized, "events": events, "warnings": warnings,
                      "total_seconds": time.perf_counter()-start,
                      "timing_note": "Analysis wall time includes optional API stages; model loading is measured by the UI separately."})


def repair_and_verify(image, original, question, gemini, clip, cfg):
    """Explicit user-triggered candidate generation, all-fact and relevance checks.

    A model-approved candidate is not independent ground truth. An inconclusive
    or contradicted fact or failed relevance check rejects the candidate.
    """
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from research.preservation import ledger, alignment_inputs, assess
    started = time.perf_counter()
    audit_cfg = {**cfg, "question": question, "generate": False, "api_decomposition": True,
                 "consistency": False, "crosscheck": True, "verify_all": True}
    # Recheck original facts on this image. Stored low-risk claims are not
    # automatically promoted to immutable gold.
    baseline = analyze(image, original, audit_cfg, clip, gemini)
    original_ledger = ledger(original, baseline["claims"])
    candidate = gemini.repair(image, original, question)
    report = analyze(image, candidate, audit_cfg, clip, gemini)
    candidate_ledger = ledger(candidate, report["claims"], "c")
    relevance = gemini.verify_answer(image, candidate, question)
    alignment, audit_error = None, None
    try:
        alignment = gemini.audit_preservation(alignment_inputs(original_ledger, candidate_ledger, candidate))
    except Exception as exc:
        audit_error = type(exc).__name__
    gate = assess(original_ledger, candidate_ledger, candidate, alignment, relevance)
    if audit_error:
        gate["audit_error"] = audit_error
    accepted = gate["accepted"]
    return {"status": "model_verified_candidate" if accepted else "rejected_candidate",
            "original": original, "candidate": candidate, "accepted": accepted,
            "final_response": candidate if accepted else None, "relevance_check": relevance,
            "claim_checks": report, "original_checks": baseline,
            "original_ledger": original_ledger, "candidate_ledger": candidate_ledger,
            "preservation": gate, "seconds": time.perf_counter()-started,
            "note": "Original preserved. Model verification is fallible; independent evaluation remains required."}
