"""Sampling-based self-consistency (plan sections 3, 6a).

SelfCheckGPT-style: sample k responses at temperature T, measure each claim's
agreement against the samples. Operationalizes "unsurfaced internal uncertainty"
without a trained probe, and triples as: M3 label source, external baseline
(ladder rung 10), and UniProbe hedge.

Model-free core: token-overlap F1 agreement. The VLM-dependent sampler
(`sample_responses`) is injected; tests use canned samples. Never use the
hidden-state probe as a feature AND self-consistency from the same trace as
its label — that is circular (plan section 6a).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def f1_overlap(a: str, b: str) -> float:
    """Token F1 between two strings. 1 = identical bags, 0 = disjoint."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    from collections import Counter

    ca, cb = Counter(ta), Counter(tb)
    inter = sum((ca & cb).values())
    if inter == 0:
        return 0.0
    prec = inter / sum(ca.values())
    rec = inter / sum(cb.values())
    return 2 * prec * rec / (prec + rec)


def claim_consistency(claim: str, samples: list[str]) -> dict:
    """Agreement of `claim` against each sampled response.

    Returns {agreement (mean F1), min_f1, n_samples, uncertainty = 1 - agreement}.
    High uncertainty + high output confidence = M3 signature.
    """
    if not samples:
        return {"agreement": float("nan"), "min_f1": float("nan"),
                "n_samples": 0, "uncertainty": float("nan")}
    scores = [f1_overlap(claim, s) for s in samples]
    mean = sum(scores) / len(scores)
    return {"agreement": mean, "min_f1": min(scores),
            "n_samples": len(samples), "uncertainty": 1.0 - mean}


def sample_agreement_matrix(samples: list[str]) -> list[list[float]]:
    """Pairwise F1 matrix — semantic-entropy-lite without an NLI model."""
    return [[f1_overlap(a, b) for b in samples] for a in samples]


def score_claims_consistency(claims: list[str], sampler, n_samples: int = 5,
                             temperature: float = 0.7) -> list[dict]:
    """Score each claim. `sampler(prompt, n, temperature) -> list[str]`
    is VLM-backed on GPU; tests inject a fake."""
    try:
        samples = list(sampler(n_samples, temperature))
    except Exception as exc:
        logger.warning("consistency sampling failed: %s", exc)
        samples = []
    return [claim_consistency(c, samples) for c in claims]
