"""Negation handling for CLIP-based verification.

CLIP is bag-of-words-ish: "no dog in the room" embeds near "dog in the room",
so a FALSE negated claim ("there is no dog" said of a dog photo) scores a
strong match — i.e. low risk, exactly backwards. This module detects explicit
negation and inverts the reading: for a negated claim, a strong match of the
*negated content* means the claim is FALSE (high risk).

    risk_negated = match  (not 1 - match)

where match is the [0,1] evidence/similarity unit score. Applied as a
post-fusion correction so the core fusion math (and its 27 tests) is untouched.
The correction is flagged in `rules_fired` as "negation-inverted" — always
visible, never silent.
"""
from __future__ import annotations

import re

NEG_PATTERN = re.compile(
    r"(?:\b(no|not|never|nothing|none|nobody|nowhere|neither|nor|without|"
    r"absent|missing|lack|lacks|lacking)\b|n't\b|n['\u2019]t\b|free of|clear of)",
    re.IGNORECASE,
)


def has_negation(claim: str) -> bool:
    return bool(NEG_PATTERN.search(claim or ""))


HEDGE_PATTERN = re.compile(
    r"\b(maybe|perhaps|possibly|probably|appears?|seems?|looks?\s+like|"
    r"might|could\s+be|suggests?|likely|uncertain)\b",
    re.IGNORECASE,
)

# Leading-negation strippers, ordered most-specific first. Each turns a denied
# claim into its positive counterclaim for pairwise scoring.
_NEG_STRIP_RES = [
    re.compile(r"^\s*there\s+is\s+no\s+(?P<body>.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*there\s+are\s+no\s+(?P<body>.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*no\s+(?P<body>.+?)\s*$", re.IGNORECASE),
]


def has_hedge(claim: str) -> bool:
    """Uncertain language — never gets a hard negation, only abstention."""
    return bool(HEDGE_PATTERN.search(claim or ""))


def make_counterclaim(claim: str) -> dict:
    """Build the minimal counterclaim for pairwise verification.

    Returns {is_negated, claim_text, counterclaim_text or None, reason}.
    Positive claim -> "No <claim-stripped>". Negated claim -> stripped
    positive. Hedged or empty claims -> counterclaim None (abstain path).
    """
    text = (claim or "").strip().rstrip(".")
    if not text:
        return {"is_negated": False, "claim_text": claim,
                "counterclaim_text": None, "reason": "empty claim"}
    if has_hedge(text):
        return {"is_negated": False, "claim_text": claim,
                "counterclaim_text": None,
                "reason": "hedged language: no hard negation generated"}
    for rx in _NEG_STRIP_RES:
        m = rx.match(text)
        if m:
            body = m.group("body").strip()
            if body:
                return {"is_negated": True, "claim_text": claim,
                        "counterclaim_text": body,
                        "reason": "negated claim: counterclaim is the positive form"}
    stripped = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE).strip()
    return {"is_negated": False, "claim_text": claim,
            "counterclaim_text": f"No {stripped or text}",
            "reason": "positive claim: counterclaim is the denial"}


def invert_for_negation(match_unit: float) -> float:
    """[0,1] match of the negated content -> risk. NaN propagates."""
    if match_unit != match_unit:
        return float("nan")
    return float(min(1.0, max(0.0, match_unit)))


def apply_negation(bundle_claim_text: str, risk_evidence: float,
                   risk_similarity: float) -> tuple[float, bool]:
    """Given risk-oriented evidence/similarity, return (corrected_mean_risk, applied).

    Converts back to match units, averages the available ones, and returns the
    match itself as the risk for negated claims. Non-negated claims pass
    through untouched (applied=False).
    """
    if not has_negation(bundle_claim_text):
        return float("nan"), False
    matches = []
    if risk_evidence == risk_evidence:
        matches.append(1.0 - risk_evidence)
    if risk_similarity == risk_similarity:
        matches.append(1.0 - risk_similarity)
    if not matches:
        return float("nan"), True
    return float(sum(matches) / len(matches)), True
