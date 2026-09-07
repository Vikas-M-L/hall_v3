"""
Counterfactual visual-grounding wrapper.

=============================================================================
THE EXTERNAL MODEL IS A SCAFFOLD — the repo ID "Counterfactual-Hallucination-Detect"
could not be independently verified. Confirm the real Hugging Face model ID and
its API before enabling `mode: model`.

Before implementing, confirm:
  1. the actual repo ID (config key `counterfactual.checkpoint`)
  2. what it consumes — image + masked image? image + region box? image + claim?
  3. output orientation: is a HIGH score hallucinated, or well-grounded?
  4. whether it expects a specific masking style (blur / black / inpaint)
Item 3 silently inverts your results if you get it wrong.

A working LOCAL mode is provided below and is what configs/default.yaml selects,
because the counterfactual signal does not actually require an external model —
you can compute it with the VLM you have already loaded. (The in-code fallback
when no mode is set is `placeholder`, so an incomplete config fails safe rather
than silently doubling your inference cost.)
=============================================================================

Contract
--------
    input:   image, region box, claim text
    output:  sensitivity score in [0, 1]

ORIENTATION, read this before wiring into fusion
------------------------------------------------
HIGH sensitivity = masking the region changed the model's confidence a lot
                 = the claim genuinely depended on that region
                 = WELL GROUNDED
                 = LOW hallucination risk

So the fusion layer inverts this signal (risk = 1 - sensitivity). If you swap in
an external model that reports the opposite orientation, fix it HERE so that the
contract above still holds, rather than special-casing it in the fusion logic.

=============================================================================
CALIBRATION WARNING — the same trap as clip_wrapper, for the same reason

The raw quantity here is a difference of mean log-probs, typically on the order
of 0.01-0.05 nats. Two things follow, and both were bugs in the first draft:

  * The scale that maps delta -> [0,1] has to match that magnitude. Too large and
    every sensitivity lands near 0, i.e. every claim looks maximally ungrounded.
  * Clamping negative deltas to 0 is worse than it looks. Roughly half of all
    deltas are negative pure noise, and clamping maps every one of them to
    sensitivity exactly 0.0, i.e. risk exactly 1.0. Half your claims would be
    flagged at maximum risk by a coin flip.

So: the map is a symmetric tanh centred on delta = 0, where delta = 0 gives
sensitivity 0.5 — "this region made no measurable difference", which is genuine
ignorance, not evidence of hallucination. Negative deltas land below 0.5 without
piling up on the boundary.

DELTA_SCALE is still a guess. Run once, read `calibration_report()`, set
`counterfactual.delta_scale` from the observed spread.
=============================================================================
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]

_WARNED = False

# Nats of mean-log-prob change that count as "substantial dependence on this
# region". delta = +DELTA_SCALE maps to sensitivity ~0.88, -DELTA_SCALE to ~0.12.
# A GUESS — see the calibration warning above.
DELTA_SCALE = 0.05

# Masking a box this large is not a region test, it is an image-ablation test:
# every claim becomes sensitive and the signal stops discriminating. Above this
# fraction of image area, return nan instead.
MAX_MASK_AREA_FRACTION = 0.60

VALID_MASK_MODES = ("blur", "grey", "noise")
_MASK_MODE_ALIASES = {"gray": "grey", "black": "grey", "gaussian": "blur"}


def mask_region(
    image: Image.Image,
    box: Box,
    mode: str = "blur",
    blur_radius: int = 25,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Occlude `box` = (left, top, right, bottom) in a copy of the image."""
    out = image.copy()
    left, top, right, bottom = box
    if right <= left or bottom <= top:
        return out

    crop = out.crop(box)
    if mode == "blur":
        patch = crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    elif mode == "grey":
        patch = Image.new("RGB", crop.size, (128, 128, 128))
    elif mode == "noise":
        gen = rng if rng is not None else np.random.default_rng(0)
        arr = gen.integers(0, 256, (crop.size[1], crop.size[0], 3), dtype=np.uint8)
        patch = Image.fromarray(arr)
    else:
        raise ValueError(f"unknown mask mode: {mode}")

    out.paste(patch, box)
    return out


def sensitivity_from_delta(delta: float, scale: float = DELTA_SCALE) -> float:
    """
    Map a mean-log-prob delta to sensitivity in (0, 1), symmetric about 0.

        delta  0     -> 0.5   masking changed nothing measurable
        delta +scale -> ~0.88 claim depended on the region  (grounded)
        delta -scale -> ~0.12 claim got *more* likely without it

    tanh rather than clip: nothing piles up on the boundaries, so a noisy zero
    stays distinguishable from real anti-grounding.
    """
    if np.isnan(delta):
        return float("nan")
    return float(0.5 * (1.0 + np.tanh(float(delta) / scale)))


class CounterfactualWrapper:
    """
    Modes
    -----
    placeholder  returns a fixed constant so the pipeline runs end-to-end
    local        (recommended default) mask the region, re-score the claim with
                 the already-loaded VLM, and measure the confidence shift.
                 Needs no external model and no training.
    model        the unverified external model — TODO, see the header
    """

    VALID_MODES = ("placeholder", "local", "model")

    def __init__(self, cfg: dict, vlm=None):
        self.cfg = cfg
        self.ccfg = cfg["counterfactual"]
        self.vlm = vlm
        self.mode = self.ccfg.get("mode", "placeholder")
        self.placeholder = float(self.ccfg.get("placeholder_value", 0.5))
        self.blur_radius = self.ccfg.get("blur_radius", 25)
        self.delta_scale = float(self.ccfg.get("delta_scale", DELTA_SCALE))
        self._rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self._deltas: list[float] = []

        # Validate config BEFORE anything downloads a 7B checkpoint: a typo here
        # should cost a second, not twenty minutes.
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"counterfactual.mode must be one of {self.VALID_MODES}, "
                f"got {self.mode!r}"
            )
        raw_mask = self.ccfg.get("mask_mode", "blur")
        self.mask_mode = _MASK_MODE_ALIASES.get(raw_mask, raw_mask)
        if self.mask_mode not in VALID_MASK_MODES:
            raise ValueError(
                f"counterfactual.mask_mode must be one of {VALID_MASK_MODES} "
                f"(aliases: {sorted(_MASK_MODE_ALIASES)}), got {raw_mask!r}"
            )
        if self.delta_scale <= 0:
            raise ValueError("counterfactual.delta_scale must be > 0")

        if self.mode == "model":
            if not self.ccfg.get("checkpoint"):
                raise ValueError(
                    "counterfactual.mode == 'model' but no checkpoint set. "
                    "Confirm the real repo ID first."
                )
            raise NotImplementedError(
                "External counterfactual model not implemented — repo ID unverified. "
                "Use mode: local, which needs no external model."
            )
        if self.mode == "local" and vlm is None:
            raise ValueError("counterfactual.mode == 'local' requires a VLMWrapper")

    def sensitivity(
        self,
        image: Image.Image,
        claim_text: str,
        box: Box | None = None,
        prompt: str = "Describe this image.",
    ) -> float:
        """
        Returns sensitivity in [0, 1]. HIGH = claim depended on the region = grounded.

        `box` should be the best-matching region from CLIPWrapper.evidence(). If it
        is None — no localized referent was found — this returns nan rather than
        masking the whole image, because a whole-image mask measures something
        different and would make every claim look sensitive.
        """
        global _WARNED

        if self.mode == "placeholder":
            if not _WARNED:
                logger.warning(
                    "Counterfactual is STUBBED — returning constant %.2f for every claim.",
                    self.placeholder,
                )
                _WARNED = True
            return self.placeholder

        if self.mode == "local":
            return self._local_sensitivity(image, claim_text, box, prompt)

        raise NotImplementedError("external counterfactual model not implemented")

    # ------------------------------------------------------------------------

    def _local_sensitivity(
        self, image: Image.Image, claim_text: str, box: Box | None, prompt: str
    ) -> float:
        """
        Score the claim against the original image and against the masked image;
        the normalized drop in mean log-prob is the sensitivity.

        Uses mean (not summed) log-prob so the result is length-invariant —
        otherwise long claims would appear systematically more sensitive.
        """
        if box is None:
            logger.debug("no region box for claim; counterfactual unavailable")
            return float("nan")

        w, h = image.size
        box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        if box_area <= 0:
            return float("nan")
        if box_area > MAX_MASK_AREA_FRACTION * w * h:
            logger.debug(
                "region covers %.0f%% of the image; too large for a region test",
                100.0 * box_area / (w * h),
            )
            return float("nan")

        try:
            base = self.vlm.score_continuation(image, prompt, claim_text)
            masked_img = mask_region(
                image, box, self.mask_mode, self.blur_radius, self._rng
            )
            masked = self.vlm.score_continuation(masked_img, prompt, claim_text)
        except Exception as exc:  # noqa: BLE001 - a failed score must not kill the run
            logger.warning("local counterfactual scoring failed: %s", exc)
            return float("nan")

        if np.isnan(base.mean_logprob) or np.isnan(masked.mean_logprob):
            return float("nan")

        # Positive delta = confidence dropped when the region was hidden.
        delta = base.mean_logprob - masked.mean_logprob
        self._deltas.append(float(delta))
        return sensitivity_from_delta(delta, self.delta_scale)

    def calibration_report(self) -> str:
        if not self._deltas:
            return "Counterfactual: no deltas computed."
        a = np.asarray(self._deltas)
        pos = float((a > 0).mean()) * 100.0
        msg = (
            f"Counterfactual deltas over {len(a)} claims: "
            f"min={a.min():+.4f} median={np.median(a):+.4f} max={a.max():+.4f} "
            f"p90|delta|={np.percentile(np.abs(a), 90):.4f}; {pos:.0f}% positive. "
            f"delta_scale={self.delta_scale:.4f}"
        )
        p90 = float(np.percentile(np.abs(a), 90))
        if p90 < 0.2 * self.delta_scale:
            msg += (
                "  <-- delta_scale is far too LARGE for this spread: sensitivities "
                "are all bunched near 0.5 and the signal is inert. Set "
                "counterfactual.delta_scale near the p90 above."
            )
        elif p90 > 5.0 * self.delta_scale:
            msg += (
                "  <-- delta_scale is too SMALL: tanh is saturating and most "
                "sensitivities are pinned near 0 or 1."
            )
        if 40.0 < pos < 60.0:
            msg += (
                "  NOTE: deltas are ~50/50 positive/negative, which is what pure "
                "noise looks like. Check that masking is actually changing the "
                "region before trusting this signal at all."
            )
        return msg
