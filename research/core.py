"""Strict data contracts, group isolation and label-free closed-loop routing."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

TYPES = ("object", "attribute", "relation", "counting", "spatial", "ocr",
         "reasoning", "knowledge", "unsupported_inference", "grounding", "unknown")
ACTIONS = {"object": "visual_recheck", "attribute": "attribute_recheck",
           "relation": "relation_recheck", "counting": "recount",
           "spatial": "spatial_recheck", "ocr": "read_text",
           "reasoning": "reasoning_recheck", "knowledge": "abstain",
           "unsupported_inference": "abstain", "grounding": "visual_recheck",
           "unknown": "abstain"}
STRATEGIES = ("original", "detector_only", "uniform_self_correction",
              "detect_then_fixed", "type_routing", "random_router",
              "routed_no_verifier", "full", "no_diagnosis",
              "preservation_full", "preservation_fixed", "preservation_gate_disabled")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(clean(r), ensure_ascii=False, allow_nan=False)
                            + "\n" for r in rows), encoding="utf-8")


@dataclass(frozen=True)
class Case:
    """Only model-visible inputs. Gold and source error types are forbidden here."""
    id: str
    group_id: str
    image: str
    image_sha256: str
    question: str
    response: str
    split: str

    @classmethod
    def parse(cls, row):
        fields = set(cls.__dataclass_fields__)
        if set(row) != fields:
            raise ValueError(f"input fields must be {sorted(fields)}; gold belongs separately")
        case = cls(**row)
        if any(not isinstance(v, str) or not v.strip() for v in asdict(case).values()):
            raise ValueError("all input fields must be nonempty strings")
        if case.split not in ("train", "calibration", "test", "development", "fixture"):
            raise ValueError("unknown split")
        if len(case.image_sha256) != 64 or any(c not in "0123456789abcdef" for c in case.image_sha256):
            raise ValueError("image_sha256 must be a lowercase SHA256 digest")
        return case


def validate_cases(cases, verify_images=False):
    ids, groups, images = set(), {}, {}
    for c in cases:
        if c.id in ids:
            raise ValueError(f"duplicate id: {c.id}")
        ids.add(c.id)
        for key, seen in ((c.group_id, groups), (c.image_sha256, images)):
            if key in seen and seen[key] != c.split:
                raise ValueError("image/group crosses partitions")
            seen[key] = c.split
        if verify_images:
            if hashlib.sha256(Path(c.image).read_bytes()).hexdigest() != c.image_sha256:
                raise ValueError(f"image hash mismatch: {c.id}")
    return {"n": len(cases), "groups": len(groups), "images": len(images)}


def assign_split(group_id, salt, development_groups=()):
    """Protocol helper, not a substitute for official dataset partitions."""
    if group_id in development_groups:
        return "development"
    x = int(hashlib.sha256(f"{salt}:{group_id}".encode()).hexdigest()[:8], 16) / 2**32
    return "train" if x < .6 else "calibration" if x < .8 else "test"


def validate_detection(d):
    if not isinstance(d, dict) or not isinstance(d.get("claims"), list) or not d["claims"]:
        raise ValueError("detector must return nonempty claims")
    for c in d["claims"]:
        if not isinstance(c, dict) or not isinstance(c.get("text"), str) or not c["text"].strip():
            raise ValueError("invalid claim")
        if c.get("type") not in TYPES:
            raise ValueError("invalid predicted error type")
        p = c.get("risk")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("risk must be finite [0,1]; missing is not zero")
        if c.get("verdict") not in ("supported", "contradicted", "unresolved"):
            raise ValueError("invalid verifier verdict")
    return d


def run_case(case, backend, strategy="full", threshold=.5, seed=42):
    """One bounded repair attempt. Never sees gold; preserves originals in trace.

    backend.call(stage, inputs, action) returns JSON. Failed verification abstains
    rather than silently accepting a rewrite. Absent modalities route to review.
    """
    if strategy not in STRATEGIES or not 0 <= threshold <= 1:
        raise ValueError("invalid strategy/threshold")
    inputs = {k: getattr(case, k) for k in ("image", "image_sha256", "question", "response")}
    trace = {"id": case.id, "group_id": case.group_id, "split": case.split,
             "strategy": strategy, "seed": seed, "original_response": case.response,
             "final_response": case.response, "status": "kept_original",
             "action": "accept", "candidate": None, "detection": None, "verification": None}
    if strategy == "original":
        return trace
    detection = validate_detection(backend.call("detect", inputs, "none"))
    trace["detection"] = detection
    for c in detection["claims"]:
        # Exact source offsets only. Fuzzy alignment would overstate span evidence.
        source = c.get("source", "")
        start = case.response.find(source) if source else -1
        c["span"] = [start, start + len(source)] if start >= 0 else None
    trace["risk"] = max(c["risk"] for c in detection["claims"])
    suspect = max(detection["claims"], key=lambda c: c["risk"])
    requires_action = trace["risk"] >= threshold or any(
        c["verdict"] != "supported" for c in detection["claims"])
    if strategy == "detector_only" or (not requires_action and strategy != "uniform_self_correction"):
        return trace
    action = ACTIONS[suspect["type"]]
    if strategy in ("uniform_self_correction", "detect_then_fixed", "no_diagnosis", "preservation_fixed"):
        action = "visual_recheck"
    elif strategy == "random_router":
        rng = random.Random(f"{seed}:{case.id}")
        action = rng.choice(sorted(set(ACTIONS.values())))
    elif strategy == "type_routing":
        # Null taxonomy: surface claim content, not gold or detector cause.
        import re
        action = "recount" if re.search(r"\b\d+\b", suspect["text"]) else "visual_recheck"
    trace["action"] = action
    if action == "abstain":
        trace.update(status="abstained", final_response=None)
        return trace
    repair_evidence = detection
    if strategy == "no_diagnosis":
        # Remove diagnosis and diagnostic rationale, retain only the detector's
        # factual spans/verdicts. Otherwise this ablation leaks type via evidence.
        repair_evidence = {"claims": [{k: v for k, v in c.items()
                                      if k in ("text", "source", "span", "risk", "verdict")}
                                     for c in detection["claims"]]}
    candidate = backend.call("repair", {**inputs, "evidence": repair_evidence}, action)
    text = candidate.get("response") if isinstance(candidate, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("repair must return a nonempty response")
    trace["candidate"] = text.strip()
    if strategy == "routed_no_verifier":
        trace.update(status="accepted_unverified", final_response=text.strip())
        return trace
    # Verifier receives question and candidate, not diagnosis or suggested verdict.
    verification = backend.call("verify", {**inputs, "response": text.strip()}, "none")
    if not isinstance(verification, dict):
        raise ValueError("invalid verification response")
    if any(type(verification.get(k)) not in (bool, type(None))
           for k in ("supported", "answers_question")):
        raise ValueError("verification flags must be bool or null")
    trace["verification"] = verification
    accepted = verification.get("supported") is True and verification.get("answers_question") is True
    if strategy.startswith("preservation_"):
        from research.preservation import ledger, alignment_inputs, assess
        before = ledger(case.response, detection["claims"])
        after_detection = validate_detection(backend.call("detect", {**inputs, "response": text.strip()}, "none"))
        after = ledger(text.strip(), after_detection["claims"], "c")
        # Gate-disabled arm still collects the same audit: isolates acceptance
        # policy from verification compute. Compare logical costs separately.
        alignment = backend.call("preservation", {"image": inputs["image"],
                                  "image_sha256": inputs["image_sha256"],
                                  **alignment_inputs(before, after, text.strip())}, "none")
        trace["original_ledger"], trace["candidate_ledger"] = before, after
        trace["preservation"] = assess(before, after, text.strip(), alignment, verification,
                                        enabled=strategy != "preservation_gate_disabled")
        accepted = trace["preservation"]["accepted"]
    if accepted:
        trace.update(status="accepted_repair", final_response=text.strip())
    else:
        trace.update(status="rejected_repair_abstained", final_response=None)
    return trace
