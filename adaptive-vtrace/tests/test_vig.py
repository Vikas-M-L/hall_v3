#!/usr/bin/env python3
"""Tests for signals/vig.py — pure numpy, no models, no GPU."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.vig import (  # noqa: E402
    log_expected_prob,
    per_token_vig,
    score_claim_vig,
    vig_from_logprobs,
)


def test_logsumexp_matches_definition():
    nulls = [-2.0, -3.0, -2.5, -4.0]
    got = log_expected_prob(nulls, True)
    want = math.log(sum(math.exp(v) for v in nulls) / len(nulls))
    assert abs(got - want) < 1e-9, (got, want)


def test_mean_of_logs_is_biased_upward():
    # Jensen (log concave): mean(log) <= log(mean). The biased estimator
    # *underestimates* the prior term, hence *overestimates* VIG — worst for
    # rare claims where null variance is high (plan section 3).
    rng = np.random.default_rng(0)
    nulls = list(rng.normal(-3.0, 1.5, size=200))
    assert log_expected_prob(nulls, False) < log_expected_prob(nulls, True)


def test_vig_is_base_minus_prior():
    v, p = vig_from_logprobs(-1.0, [-2.0, -3.0], True)
    assert abs(v - (-1.0 - p)) < 1e-9
    # M1 signature: high prior (common claim), low gain
    assert p > -3.0


def test_nan_propagates():
    v, p = vig_from_logprobs(float("nan"), [-2.0])
    assert np.isnan(v)
    v, p = vig_from_logprobs(-1.0, [float("nan")])
    assert np.isnan(v) and np.isnan(p)
    assert np.isnan(log_expected_prob([]))


def test_per_token_averages_positions():
    base = [-1.0, -2.0]
    nulls = [[-2.0, -3.0], [-3.0, -4.0]]
    vpt, ppt = per_token_vig(base, nulls, True)
    v0, _ = vig_from_logprobs(-1.0, [-2.0, -3.0], True)
    v1, _ = vig_from_logprobs(-2.0, [-3.0, -4.0], True)
    assert abs(vpt - (v0 + v1) / 2) < 1e-9


def test_score_claim_with_fake_scorer():
    # Fake scorer: claim tokens depend on image id — real image helps.
    def fake(prompt, image):
        return [-0.5, -0.7] if image == "real" else [-2.0, -2.5]

    out = score_claim_vig(fake, "a fork", "real", ["n1", "n2", "n3", "n4"])
    assert out["n_nulls"] == 4
    assert out["vig"] > 1.0, out  # strong image dependence
    assert out["prior"] < -1.5, out  # rare-ish claim


def test_prior_term_separates_common_from_rare():
    # Same VIG-ish setup, common claim has HIGHER prior term than rare claim.
    def fake_common(prompt, image):
        return [-0.4, -0.5]

    def fake_rare(prompt, image):
        return [-0.4, -0.5] if image == "real" else [-5.0, -5.0]

    c = score_claim_vig(fake_common, "a table", "real", ["n1", "n2"])
    r = score_claim_vig(fake_rare, "14 km sign", "real", ["n1", "n2"])
    assert c["prior"] > r["prior"], (c["prior"], r["prior"])


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
