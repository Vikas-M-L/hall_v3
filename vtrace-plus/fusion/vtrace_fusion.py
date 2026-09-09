"""
V-TRACE+ fusion — deterministic, rule-based, no learned parameters.

Combines per-claim signals into a single hallucination risk in [0, 1], plus an
explanation of how that number was reached.

Two modes
---------
v1  fixed equal-weight average across all active signals
v2  rule-based adaptive weighting — weights are a deterministic function of the
    signal values themselves. Nothing here is fitted; there is no training loop,
    no dataset, and no gradient anywhere in this file.

Every threshold and weight is a named constant in the block below, so behaviour
is tunable without touching logic.

--------------------------------------------------------------------------
SIGNAL ORIENTATION — the single most important thing in this file
--------------------------------------------------------------------------
Signals arrive in their own natural directions. Some are "higher is better",
some are "higher is worse". They are all converted to RISK orientation
(higher = more likely hallucinated) before any weighting happens:

    confidence      high = model is sure           -> risk = 1 - confidence
    evidence        high = visually grounded       -> risk = 1 - evidence
    clip_similarity high = image/text match        -> risk = 1 - similarity
    counterfactual  high = claim depended on region-> risk = 1 - sensitivity
    uniprobe        high = hallucinated            -> risk = uniprobe (as-is)

Getting one of these backwards inverts your results while producing numbers that
look entirely reasonable. If you swap in a model with a different orientation,
fix it in that model's wrapper so this table stays true.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# =========================================================================
# TUNABLE CONSTANTS — no magic numbers below this block
# =========================================================================

ALL_SIGNALS: tuple[str, ...] = (
    "confidence",
    "evidence",
    "clip_similarity",
    "uniprobe",
    "counterfactual",
)

# Signals whose natural direction is "higher = more faithful" and therefore get
# inverted into risk orientation.
INVERTED_SIGNALS: frozenset[str] = frozenset(
    {"confidence", "evidence", "clip_similarity", "counterfactual"}
)

# Base weights for v2 before any rule fires. Must sum to 1.0 over ALL_SIGNALS.
BASE_WEIGHTS: dict[str, float] = {
    "confidence": 0.20,
    "evidence": 0.25,
    "clip_similarity": 0.15,
    "uniprobe": 0.20,
    "counterfactual": 0.20,
}

# --- Rule 1: confidence-evidence discrepancy ------------------------------
# CED = confidence - evidence, in native (not risk) orientation, range [-1, 1].
# High CED means the model is confident while visual evidence is weak — the
# classic "sure but ungrounded" signature. When it fires, stop trusting the
# model's own confidence and lean on the signals that look at the image.
#
# CED IS NOT SCALE-FREE, and this threshold is meaningless until the two inputs
# are on comparable scales. `confidence` is a geometric-mean token probability,
# which under greedy decoding piles up in 0.85-0.98. `evidence` is a rescaled
# CLIP cosine whose spread depends entirely on clip.cos_min / clip.cos_max. Pick
# that window badly and CED is roughly "0.9 minus a constant" for every claim, so
# this rule fires on all claims or none, and the v2/v1 comparison measures the
# rescaling constant rather than anything about hallucination. Calibrate CLIP
# first (see models/clip_wrapper.py), then set this from the observed CED
# distribution in the run summary.
CED_THRESHOLD: float = 0.25
CED_UPWEIGHT_EVIDENCE: float = 1.60
CED_UPWEIGHT_COUNTERFACTUAL: float = 1.60
CED_DOWNWEIGHT_CONFIDENCE: float = 0.40

# --- Rule 2: cross-detector disagreement ----------------------------------
# Standard deviation across the risk-oriented values of the detector-style
# signals. High disagreement means they are contradicting each other, so weight
# shifts onto clip_similarity, which is independent of the VLM's internals and
# of the probes.
#
# CAVEAT, and it is a real one: clip_similarity is NOT independent of `evidence`
# — both come out of the same CLIP encoder (whole image vs. max over regions).
# So when disagreement is driven by `evidence` being the outlier, this rule
# downweights evidence and promotes the signal most correlated with it. See the
# "Known design concerns" section of the README before trusting Rule 2.
#
# The std is a population std over however many detectors are available, so the
# threshold's meaning shifts with that count: with 2 signals std = |a-b|/2, so
# 0.20 means |a-b| > 0.40; with 3 and one outlier it means |delta| > ~0.43.
DISAGREEMENT_SIGNALS: tuple[str, ...] = ("evidence", "uniprobe", "counterfactual")
DISAGREEMENT_THRESHOLD: float = 0.20
DISAGREEMENT_DOWNWEIGHT_DETECTORS: float = 0.60
DISAGREEMENT_UPWEIGHT_CLIP: float = 2.00

# --- v3: pairwise contradiction + calibrated abstention -------------------
# v3 keeps the v1 equal-weight base risk and adds two deterministic terms from
# the claim-vs-counterclaim margin (match units, same calibration window):
#   contra_strength = clip(-margin / 0.5) in [0,1] — counterclaim winning
#   risk_v3 = (1-a)*base + a*0.5 + c*contra, a = AMBIGUITY_WEIGHT*ambiguity
# Plus a three-state verdict from the margin alone, and a reasons list naming
# every trigger. v1/v2 code paths are untouched by this block.
PAIRWISE_MARGIN_THRESHOLD: float = 0.12
PAIRWISE_AMBIGUITY_THRESHOLD: float = 0.35
ABSTENTION_THRESHOLD: float = 0.60
CONTRADICTION_WEIGHT: float = 0.20
AMBIGUITY_WEIGHT: float = 0.15
LOCAL_GLOBAL_GAP_THRESHOLD: float = 0.35
# Negated claims ("There is no X") are verified through the POSITIVE form's
# match m: m high means X is present so the denial is contradicted; m low
# means X is absent so the denial is supported. Margin math cannot do this —
# both texts share the content words, so the margin mostly measures fluency.
PAIRWISE_NEG_HIGH: float = 0.60
PAIRWISE_NEG_LOW: float = 0.40
# When the pairwise margin ties, the base visual match decides if IT is
# decisive (outside the uncertain band); otherwise the claim stays unresolved.
BASE_DECISIVE_LOW: float = 0.35
BASE_DECISIVE_HIGH: float = 0.65

# --- Aggregation ----------------------------------------------------------
DEFAULT_TOPK: int = 3

# Floor applied to each weight BEFORE renormalization, so a signal cannot be
# driven to zero by composed multipliers. With the shipped constants the lowest
# reachable pre-normalization weight is 0.08, so this never binds today; it is
# insurance for future multipliers, not an active guarantee about the final
# normalized weights (those can land below MIN_WEIGHT after dividing).
MIN_WEIGHT: float = 0.01


# =========================================================================
# Data structures
# =========================================================================


@dataclass
class SignalBundle:
    """Raw per-claim signal values, each in its own natural orientation."""

    confidence: float = float("nan")       # [0,1] geometric-mean token probability
    evidence: float = float("nan")         # [0,1] max claim-to-region CLIP score
    clip_similarity: float = float("nan")  # [0,1] claim-to-whole-image CLIP score
    uniprobe: float = float("nan")         # [0,1] hallucination probability
    counterfactual: float = float("nan")   # [0,1] masking sensitivity

    claim_id: str | None = None
    claim_text: str = ""
    claim_type: str = ""
    stubbed: tuple[str, ...] = ()          # signals currently returning placeholders

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in ALL_SIGNALS:
            v = getattr(self, s)
            out[s] = float("nan") if v is None else float(v)
        return out


@dataclass
class FusionResult:
    risk: float
    weights: dict[str, float]
    risks: dict[str, float]                # risk-oriented signal values
    contributions: dict[str, float]        # weight * risk, sums to `risk`
    ced: float
    disagreement: float
    rules_fired: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    mode: str = "v2"
    claim_id: str | None = None
    claim_text: str = ""
    verdict: str | None = None               # v3 only: supported|contradicted|unresolved
    reasons: list[str] = field(default_factory=list)   # v3 only: why this verdict
    pairwise: dict = field(default_factory=dict)       # v3 only: margin/ambiguity detail

    def top_signal(self) -> str | None:
        """The signal contributing most to this risk score."""
        if not self.contributions:
            return None
        return max(self.contributions.items(), key=lambda kv: kv[1])[0]

    def explain(self) -> str:
        if np.isnan(self.risk):
            return "No usable signals — risk undefined."

        lines = [f"risk={self.risk:.3f} (mode={self.mode})"]
        if self.claim_text:
            lines.insert(0, f'claim: "{self.claim_text}"')

        ordered = sorted(self.contributions.items(), key=lambda kv: -kv[1])
        for name, contrib in ordered:
            lines.append(
                f"  {name:<16} risk={self.risks[name]:.3f}  "
                f"w={self.weights[name]:.3f}  ->{contrib:.3f}"
            )

        lines.append(f"  CED={self.ced:+.3f}   disagreement={self.disagreement:.3f}")
        lines.append(
            "  rules fired: " + (", ".join(self.rules_fired) if self.rules_fired else "none")
        )
        if self.excluded:
            lines.append("  excluded: " + ", ".join(self.excluded))
        top = self.top_signal()
        if top:
            lines.append(f"  dominant signal: {top}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "risk": self.risk,
            "mode": self.mode,
            "weights": self.weights,
            "risks": self.risks,
            "contributions": self.contributions,
            "ced": self.ced,
            "disagreement": self.disagreement,
            "rules_fired": self.rules_fired,
            "excluded": self.excluded,
            "top_signal": self.top_signal(),
            "verdict": self.verdict,
            "reasons": self.reasons,
            "pairwise": self.pairwise,
        }


def v3_verdict_and_risk(base_risk: float, risks: dict[str, float],
                        disagreement: float, pairwise: dict | None,
                        is_negated: bool = False) -> tuple:
    """v3: three-state verdict + adjusted risk + reasons.

    Positive claims use the claim-vs-counterclaim margin (same-window match
    units): margin > +thr -> supported; < -thr -> contradicted; else
    unresolved. Negated claims are verified through the POSITIVE form's match
    m instead (the margin is fluency noise when both texts share content
    words): m > NEG_HIGH -> contradicted, m < NEG_LOW -> supported.
    Risk starts at the v1 base and is pulled toward 0.5 by ambiguity and
    pushed up by contradiction strength. Reasons name triggers: near-tie,
    weak localization, local/global gap, detector disagreement,
    confident-model-vs-uncertain-verifier, abstention-level risk.
    """
    pw = pairwise or {}
    margin = pw.get("margin", float("nan"))
    amb = pw.get("ambiguity", float("nan"))
    reasons: list[str] = []
    if is_negated:
        m = pw.get("counter_match", float("nan"))
        if m != m:
            verdict = "unresolved"
            reasons.append("no positive-form match available")
        elif m > PAIRWISE_NEG_HIGH:
            verdict = "contradicted"
            reasons.append(f"positive form matches at {m:.2f}: the denial is false")
        elif m < PAIRWISE_NEG_LOW:
            verdict = "supported"
            reasons.append(f"positive form matches at {m:.2f}: the denial holds")
        else:
            verdict = "unresolved"
            reasons.append(f"positive-form match {m:.2f} is inconclusive")
    elif margin != margin:
        verdict = "unresolved"
        reasons.append("no pairwise margin available (hedged, empty, or failed)")
    elif margin > PAIRWISE_MARGIN_THRESHOLD:
        verdict = "supported"
        reasons.append(f"claim wins by margin {margin:+.2f}")
    elif margin < -PAIRWISE_MARGIN_THRESHOLD:
        verdict = "contradicted"
        reasons.append(f"counterclaim wins by margin {margin:+.2f}")
    else:
        # Pairwise tie. Bare-noun claims ("a cow") produce tiny margins because
        # the denial shares every content word; fall back to the base match
        # when IT is decisive, and reserve 'unresolved' for genuine toss-ups.
        if base_risk == base_risk and base_risk < BASE_DECISIVE_LOW:
            verdict = "supported"
            reasons.append("pairwise tied; base visual match decisively supports "
                           f"(risk {base_risk:.2f})")
        elif base_risk == base_risk and base_risk >= BASE_DECISIVE_HIGH:
            verdict = "contradicted"
            reasons.append("pairwise tied; base visual match decisively refutes "
                           f"(risk {base_risk:.2f})")
        else:
            verdict = "unresolved"
            reasons.append("positive and negative hypotheses are nearly tied")

    if amb == amb and amb > PAIRWISE_AMBIGUITY_THRESHOLD:
        if verdict == "unresolved":
            reasons.append(f"ambiguity {amb:.2f} above threshold")
    er, sr = risks.get("evidence", float("nan")), risks.get("clip_similarity", float("nan"))
    if er == er and sr == sr and abs(er - sr) > LOCAL_GLOBAL_GAP_THRESHOLD:
        reasons.append("local and global evidence disagree "
                       f"(gap {abs(er - sr):.2f})")
        if verdict == "supported":
            verdict = "unresolved"
    if disagreement == disagreement and disagreement > DISAGREEMENT_THRESHOLD:
        reasons.append("detectors disagree beyond Rule 2 threshold")
    cr = risks.get("confidence", float("nan"))
    if (cr == cr and cr < 0.2 and base_risk == base_risk
            and 0.35 <= base_risk <= 0.65):
        reasons.append("VLM is confident but the verifier is uncertain")

    if is_negated:
        m = pw.get("counter_match", float("nan"))
        contra = float(max(0.0, min(1.0, (m - 0.5) / 0.5))) if m == m else 0.0
    else:
        contra = (float(max(0.0, min(1.0, -margin / 0.5)))
                  if margin == margin else 0.0)
    a = AMBIGUITY_WEIGHT * (amb if amb == amb else 0.0)
    if base_risk != base_risk:
        risk = float("nan")
    else:
        risk = float(np.clip((1 - a) * base_risk + a * 0.5
                             + CONTRADICTION_WEIGHT * contra, 0.0, 1.0))
    if risk == risk and risk >= ABSTENTION_THRESHOLD and verdict != "contradicted":
        reasons.append("risk above abstention threshold — route to human/GPU review")
    return verdict, risk, reasons


# =========================================================================
# Core helpers
# =========================================================================


def to_risk(name: str, value: float) -> float:
    """Convert a signal from its natural orientation into risk orientation."""
    if value is None or np.isnan(value):
        return float("nan")
    v = float(np.clip(value, 0.0, 1.0))
    return 1.0 - v if name in INVERTED_SIGNALS else v


def compute_ced(confidence: float, evidence: float) -> float:
    """
    Confidence minus evidence, in NATIVE orientation, range [-1, 1].

    Positive and large = confident but poorly grounded.
    Returns nan if either input is missing, which disables Rule 1.
    """
    if np.isnan(confidence) or np.isnan(evidence):
        return float("nan")
    return float(np.clip(confidence, 0, 1) - np.clip(evidence, 0, 1))


def compute_disagreement(
    risks: dict[str, float], stubbed: tuple[str, ...] | frozenset[str] = ()
) -> float:
    """
    Standard deviation across available detector signals, in risk orientation.

    Needs at least two available signals to mean anything; returns nan otherwise,
    which disables Rule 2.

    Stubbed signals are excluded even when they are still being fused. A stub
    returns the same constant for every claim, so the "spread" between it and a
    real detector measures nothing but the distance to an arbitrary placeholder —
    with uniprobe and counterfactual both stubbed at 0.5, the std would track
    |evidence_risk - 0.5| and Rule 2 would fire on evidence alone.
    """
    stub = set(stubbed)
    vals = [
        risks[s]
        for s in DISAGREEMENT_SIGNALS
        if s in risks and s not in stub and not np.isnan(risks[s])
    ]
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals))


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    """Renormalize to sum to 1.0, with a floor so nothing is fully silenced."""
    if not weights:
        return {}
    floored = {k: max(v, MIN_WEIGHT) for k, v in weights.items()}
    total = sum(floored.values())
    if total <= 0:
        n = len(floored)
        return {k: 1.0 / n for k in floored}
    return {k: v / total for k, v in floored.items()}


# =========================================================================
# Weighting modes
# =========================================================================


def v1_weights(available: list[str]) -> tuple[dict[str, float], list[str]]:
    """Fixed equal weighting across available signals. Rules are not consulted."""
    if not available:
        return {}, []
    w = 1.0 / len(available)
    return {s: w for s in available}, ["v1: fixed equal weights"]


def v2_weights(
    available: list[str], risks: dict[str, float], ced: float, disagreement: float
) -> tuple[dict[str, float], list[str]]:
    """
    Rule-based adaptive weighting. Deterministic function of the signal values.

    Rules are independent and may both fire; multipliers compose, then the whole
    vector is renormalized. If neither fires, weights stay at BASE_WEIGHTS
    restricted to the available signals — near-equal, which is what v1 always does.
    """
    weights = {s: BASE_WEIGHTS[s] for s in available}
    fired: list[str] = []

    # --- Rule 1: confident but ungrounded --------------------------------
    # Gated on both inputs being AVAILABLE, not just present in the bundle. A
    # signal the caller disabled must not steer the weighting of the others.
    #
    # A rule is only recorded as "fired" if it actually moved a weight. Its
    # trigger condition can be met while every signal it would touch is absent,
    # and reporting that as a firing makes `rules_fired` describe an intent
    # rather than an effect — which is misleading in exactly the cases where you
    # are reading the explanation to work out why a score came out odd.
    ced_usable = "confidence" in risks and "evidence" in risks
    if ced_usable and not np.isnan(ced) and ced > CED_THRESHOLD:
        touched = False
        if "evidence" in weights:
            weights["evidence"] *= CED_UPWEIGHT_EVIDENCE
            touched = True
        if "counterfactual" in weights:
            weights["counterfactual"] *= CED_UPWEIGHT_COUNTERFACTUAL
            touched = True
        if "confidence" in weights:
            weights["confidence"] *= CED_DOWNWEIGHT_CONFIDENCE
            touched = True
        if touched:
            fired.append(f"CED>{CED_THRESHOLD} (confident but ungrounded)")

    # --- Rule 2: detectors contradict each other -------------------------
    # compute_disagreement already only sees available, non-stubbed signals, so
    # this is automatically gated. It needs >= 2 of them or it returns nan.
    if not np.isnan(disagreement) and disagreement > DISAGREEMENT_THRESHOLD:
        touched = False
        for s in DISAGREEMENT_SIGNALS:
            if s in weights:
                weights[s] *= DISAGREEMENT_DOWNWEIGHT_DETECTORS
                touched = True
        if "clip_similarity" in weights:
            weights["clip_similarity"] *= DISAGREEMENT_UPWEIGHT_CLIP
            touched = True
        if touched:
            fired.append(f"disagreement>{DISAGREEMENT_THRESHOLD} (detectors conflict)")

    if not fired:
        fired.append("none (near-equal base weighting)")
        return _normalize(weights), fired

    # A trigger being met is not the same as the weighting changing. Multipliers
    # that hit every available signal uniformly vanish under renormalization —
    # e.g. Rule 2 downweights all three detectors by 0.60 and upweights
    # clip_similarity, so if clip_similarity is unavailable AND confidence is
    # unavailable, the only remaining signals are the three detectors, they are
    # all scaled by the same 0.60, and the normalized result is exactly the base
    # weighting. Reporting "fired" there would send you looking for an effect on
    # the score that provably is not present.
    normalized = _normalize(weights)
    base_total = sum(BASE_WEIGHTS[s] for s in available)
    if all(
        abs(normalized[s] - BASE_WEIGHTS[s] / base_total) < 1e-12 for s in available
    ):
        return normalized, [
            "none (triggers met but uniform over the available signals, so the "
            "weighting is unchanged: " + "; ".join(fired) + ")"
        ]

    return normalized, fired


# =========================================================================
# Public API
# =========================================================================


def fuse(
    bundle: SignalBundle,
    mode: str = "v2",
    active_signals: dict[str, bool] | None = None,
    drop_stubbed: bool = False,
    pairwise: dict | None = None,
) -> FusionResult:
    """
    Fuse one claim's signals into a risk score.

    Parameters
    ----------
    mode           "v1" (equal weights), "v2" (rule-based adaptive) or
                   "v3" (pairwise contradiction + calibrated abstention;
                   needs `pairwise` from CLIPWrapper.pairwise_scores)
    active_signals config toggles; a signal set False is excluded entirely
    drop_stubbed   if True, signals listed in `bundle.stubbed` are excluded.
                    A stubbed signal returns the same constant for every claim,
                    so it adds no information and only drags scores toward that
                    constant. Worth enabling once you know which are stubbed.
    pairwise       v3 only: {margin, ambiguity, ...} in same-window match units

    Signals that are NaN (a model failed, or was never wired in) are dropped and
    the remaining weights renormalize over what is left.
    """
    if mode not in ("v1", "v2", "v3"):
        raise ValueError(f"fusion mode must be 'v1', 'v2' or 'v3', got {mode!r}")

    toggles = active_signals or {s: True for s in ALL_SIGNALS}
    raw = bundle.as_dict()

    excluded: list[str] = []
    available: list[str] = []
    risks: dict[str, float] = {}

    for s in ALL_SIGNALS:
        if not toggles.get(s, True):
            excluded.append(f"{s} (disabled)")
            continue
        if drop_stubbed and s in bundle.stubbed:
            excluded.append(f"{s} (stubbed)")
            continue
        r = to_risk(s, raw[s])
        if np.isnan(r):
            excluded.append(f"{s} (unavailable)")
            continue
        risks[s] = r
        available.append(s)

    # CED uses native orientation, and is computed even if one of the two is
    # later excluded from weighting — it is a diagnostic, not a fused signal.
    ced = compute_ced(raw["confidence"], raw["evidence"])
    disagreement = compute_disagreement(risks, bundle.stubbed)

    if not available:
        return FusionResult(
            risk=float("nan"),
            weights={},
            risks={},
            contributions={},
            ced=ced,
            disagreement=disagreement,
            rules_fired=[],
            excluded=excluded,
            mode=mode,
            claim_id=bundle.claim_id,
            claim_text=bundle.claim_text,
        )

    if mode == "v1":
        weights, fired = v1_weights(available)
    elif mode == "v3":
        weights, fired = v1_weights(available)
        fired = ["v3: equal-weight base + pairwise adjustment"]
    else:
        weights, fired = v2_weights(available, risks, ced, disagreement)

    contributions = {s: weights[s] * risks[s] for s in available}
    risk = float(np.clip(sum(contributions.values()), 0.0, 1.0))

    verdict, reasons, pw = None, [], {}
    if mode == "v3":
        from fusion.negation import make_counterclaim

        is_neg = make_counterclaim(bundle.claim_text)["is_negated"]
        verdict, risk, reasons = v3_verdict_and_risk(risk, risks, disagreement,
                                                    pairwise, is_negated=is_neg)
        pw = dict(pairwise or {})

    return FusionResult(
        risk=risk,
        weights=weights,
        risks=risks,
        contributions=contributions,
        ced=ced,
        disagreement=disagreement,
        rules_fired=fired,
        excluded=excluded,
        mode=mode,
        claim_id=bundle.claim_id,
        claim_text=bundle.claim_text,
        verdict=verdict,
        reasons=reasons,
        pairwise=pw,
    )


def aggregate(claim_risks: list[float], method: str = "max", topk: int = DEFAULT_TOPK) -> float:
    """
    Aggregate per-claim risks into one image-level risk.

    max        one bad claim makes the response risky. Sensitive, and the usual
               default for "should a human look at this?"
    topk_mean  mean of the k riskiest claims. Less jumpy than max on long
               responses, where max saturates on a single outlier.
    mean       mean over all claims. Dilutes a single severe hallucination in a
               long response; report it only alongside one of the others.
    """
    vals = [r for r in claim_risks if r is not None and not np.isnan(r)]
    if not vals:
        return float("nan")

    if method == "max":
        return float(max(vals))
    if method == "mean":
        return float(np.mean(vals))
    if method == "topk_mean":
        k = max(1, min(topk, len(vals)))
        return float(np.mean(sorted(vals, reverse=True)[:k]))
    raise ValueError(f"unknown aggregation method: {method}")


def evidence_coverage(results: list[FusionResult],
                      exclude_hedged: bool = True) -> dict:
    """Image-level decisiveness: decisive claims / total claims.

    Decisive = verdict is supported or contradicted (v3), or — for v1/v2
    results without verdicts — risk outside the [0.35, 0.65) band. Hedged
    claims (rules mention hedging/no-counterclaim) are excluded when
    exclude_hedged=True: they were never verifiable, so they must not dilute
    coverage. Separates 'high risk because contradicted' from 'high
    uncertainty because unverifiable'.
    """
    tot, dec = 0, 0
    for r in results:
        text = " ".join(r.rules_fired) + " " + " ".join(r.reasons)
        if exclude_hedged and ("hedg" in text or "no counterclaim" in text):
            continue
        tot += 1
        if r.verdict is not None:
            if r.verdict in ("supported", "contradicted"):
                dec += 1
        elif r.risk == r.risk and (r.risk < 0.35 or r.risk >= 0.65):
            dec += 1
    return {"decisive": dec, "total": tot,
            "coverage": (dec / tot) if tot else float("nan")}
