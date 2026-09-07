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
