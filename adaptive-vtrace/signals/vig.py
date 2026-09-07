"""Visual Information Gain (V-TRACE+ reframed plan, section 3).

    VIG(c) = log P(c | I, x) - log E_{I'}[ P(c | I', x) ]

The pointwise mutual information between claim and image, in nats. This is a
*repurposed* signal (M3ID PMI / VCD contrast), not a novel one — the novelty is
using it jointly with confidence + prior + uncertainty for repair routing.

Estimator rules enforced here (see plan for why):
  * log E[...] = logsumexp(log p_i) - log N, NEVER mean(log p_i) (Jensen bias).
  * The prior term log E[P(c|I',x)] is a first-class output: M1 is *high prior*,
    not merely low gain. Low VIG alone has a large false-positive basin.
  * `x` (scoring template) is an explicit argument — preregister it per run.

The numpy core is model-free and fully tested. The VLM-dependent part
(`score_claim_vig`) takes an injected `scorer` callable so tests use a fake.
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SCORING_TEMPLATE = "Describe what you see. {claim}"


def log_expected_prob(null_lps: list[float] | np.ndarray, use_logsumexp: bool = True) -> float:
    """log E[P] over nulls from per-null mean log-probs.

    Correct: logsumexp(lp) - log N. Setting use_logsumexp=False reproduces the
    *biased* mean-of-logs estimator, for the ablation only.
    """
    arr = np.asarray(null_lps, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    if use_logsumexp:
        m = float(np.max(arr))
        return float(m + math.log(float(np.mean(np.exp(arr - m)))))
    return float(np.mean(arr))


def vig_from_logprobs(base_lp: float, null_lps: list[float] | np.ndarray,
                      use_logsumexp: bool = True) -> tuple[float, float]:
    """Returns (vig_nats, prior_term_nats). NaN propagates if inputs are NaN."""
    if base_lp != base_lp:
        return float("nan"), log_expected_prob(null_lps, use_logsumexp)
    prior = log_expected_prob(null_lps, use_logsumexp)
    if prior != prior:
        return float("nan"), float("nan")
    return float(base_lp - prior), prior


def per_token_vig(base_tok_lps: list[float], null_tok_lps: list[list[float]],
                  use_logsumexp: bool = True) -> tuple[float, float]:
    """Length-normalized VIG: mean over tokens of per-token VIG.

    Null token sequences must align positionally with the base sequence
    (same scoring template => same claim tokens). Returns (vig_per_tok, prior_per_tok).
    """
    base = np.asarray(base_tok_lps, dtype=float)
    if base.size == 0:
        return float("nan"), float("nan")
    per_tok = []
    for t in range(base.size):
        col = [seq[t] for seq in null_tok_lps if t < len(seq)]
        v, _ = vig_from_logprobs(float(base[t]), col, use_logsumexp)
        per_tok.append(v)
    per_tok = np.asarray(per_tok, dtype=float)
    valid = per_tok[~np.isnan(per_tok)]
    if valid.size == 0:
        return float("nan"), float("nan")
    prior_vals = []
    for t in range(base.size):
        col = [seq[t] for seq in null_tok_lps if t < len(seq)]
        prior_vals.append(log_expected_prob(col, use_logsumexp))
    pa = np.asarray(prior_vals, dtype=float)
    return float(np.mean(valid)), float(np.nanmean(pa))


def score_claim_vig(scorer, claim: str, image, null_images: list,
                    template: str = DEFAULT_SCORING_TEMPLATE,
                    use_logsumexp: bool = True, per_token: bool = True) -> dict:
    """Full VIG scoring with an injected scorer.

    scorer(prompt_text, image) -> list[float] of token log-probs for the claim
    span under `template`. The real implementation backs this with
    VLMWrapper teacher-forced scoring (GPU); tests inject a fake.
    """
    prompt = template.format(claim=claim)
    try:
        base_tok = list(scorer(prompt, image))
    except Exception as exc:
        logger.warning("VIG base scoring failed: %s", exc)
        return {"vig": float("nan"), "prior": float("nan"),
                "vig_per_token": float("nan"), "n_nulls": 0}
    null_tok: list[list[float]] = []
    for nim in null_images:
        try:
            null_tok.append(list(scorer(prompt, nim)))
        except Exception as exc:
            logger.warning("VIG null scoring failed: %s", exc)
    base_mean = float(np.mean(base_tok)) if base_tok else float("nan")
    null_means = [float(np.mean(s)) for s in null_tok if s]
    vig, prior = vig_from_logprobs(base_mean, null_means, use_logsumexp)
    out = {"vig": vig, "prior": prior, "n_nulls": len(null_means)}
    if per_token:
        vpt, _ = per_token_vig(base_tok, null_tok, use_logsumexp)
        out["vig_per_token"] = vpt
    else:
        out["vig_per_token"] = float("nan")
    return out
