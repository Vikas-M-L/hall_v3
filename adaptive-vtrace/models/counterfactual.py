"""
Counterfactual visual-grounding module.

Two components:

  1. LOCAL SENSITIVITY — mask the claim's best-matching region, re-score the
     claim with the already-loaded VLM, and measure the confidence shift.
     Needs no external model and no training.

  2. CF-DETECT — optional external "Counterfactual-Hallucination-Detect" model.
     The repo ID could not be independently verified.  Scaffolded for future
     wiring; returns NaN when checkpoint is null.

=============================================================================
ORIENTATION — read this before wiring into fusion

HIGH sensitivity = masking the region changed the model's confidence a lot
                 = the claim genuinely depended on that region
                 = WELL GROUNDED
                 = LOW hallucination risk

So the fusion layer inverts this signal (risk = 1 - sensitivity). If you swap
in an external model with the opposite orientation, fix it HERE so that the
contract still holds.
=============================================================================

=============================================================================
CALIBRATION WARNING — same trap as clip_wrapper, same reason

The raw quantity here is a difference of mean log-probs, typically 0.01–0.05
nats.  Two things follow:

  * The scale mapping delta → [0,1] must match that magnitude. Too large and
    every sensitivity lands near 0, i.e. every claim looks maximally ungrounded.
  * Clamping negative deltas to 0 maps ~half of all deltas (negative noise) to
    sensitivity exactly 0.0, i.e. risk exactly 1.0.

So: the map is a symmetric tanh centred on delta=0, where delta=0 gives
sensitivity 0.5.  DELTA_SCALE is still a guess. Run once, read
``calibration_report()``, set ``counterfactual.delta_scale`` from the observed
spread.
=============================================================================
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]

# Nats of mean-log-prob change that count as "substantial dependence".
# delta = +DELTA_SCALE maps to sensitivity ~0.88, -DELTA_SCALE to ~0.12.
DELTA_SCALE = 0.05

# Masking a box larger than this fraction of the image is not a region test —
# every claim becomes sensitive and the signal stops discriminating.
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
    """Occlude ``box`` = (left, top, right, bottom) in a copy of the image."""
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

        delta  0     → 0.5   masking changed nothing measurable
        delta +scale → ~0.88 claim depended on the region  (grounded)
        delta -scale → ~0.12 claim got *more* likely without it

    tanh rather than clip: nothing piles up on the boundaries, so a noisy zero
    stays distinguishable from real anti-grounding.
    """
    if np.isnan(delta):
        return float("nan")
    return float(0.5 * (1.0 + np.tanh(float(delta) / scale)))


class CounterfactualModule:
    """
    Counterfactual visual-grounding signals.

    Provides two independent signals:

    1. ``sensitivity()`` — local mask-and-rescore using the loaded VLM.
    2. ``cf_detect_score()`` — external CF-Detect model (scaffolded).
    """

    def __init__(self, cfg: dict, vlm=None, grounding=None):
        """
        Parameters
        ----------
        cfg       : full config dict
        vlm       : VLMWrapper instance for local counterfactual scoring
        grounding : GroundingModule instance for finding claim-relevant regions
                    (used by control_sensitivity to pick random non-relevant
                    regions)
        """
        self.cfg = cfg
        self.ccfg = cfg["counterfactual"]
        self.vlm = vlm
        self.grounding = grounding

        # Mask settings
        raw_mask = self.ccfg.get("mask_mode", "blur")
        self.mask_mode = _MASK_MODE_ALIASES.get(raw_mask, raw_mask)
        if self.mask_mode not in VALID_MASK_MODES:
            raise ValueError(
                f"counterfactual.mask_mode must be one of {VALID_MASK_MODES} "
                f"(aliases: {sorted(_MASK_MODE_ALIASES)}), got {raw_mask!r}"
            )
        self.blur_radius = self.ccfg.get("blur_radius", 25)
        self.delta_scale = float(self.ccfg.get("delta_scale", DELTA_SCALE))
        if self.delta_scale <= 0:
            raise ValueError("counterfactual.delta_scale must be > 0")
        self.n_control_regions = self.ccfg.get("n_control_regions", 1)

        self._rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self._deltas: list[float] = []

        # CF-Detect external model
        self._cf_detect_cfg = cfg.get("cf_detect", {})
        self._cf_detect_model = None
        self._cf_detect_processor = None
        cf_enabled = self._cf_detect_cfg.get("enabled", True)
        cf_checkpoint = self._cf_detect_cfg.get("checkpoint")

        if cf_enabled and cf_checkpoint is not None:
            self._load_cf_detect(cf_checkpoint)

    # ------------------------------------------------------------------ init

    def _load_cf_detect(self, checkpoint: str) -> None:
        """
        Load the external CF-Detect model.

        NOT IMPLEMENTED — the repo ID "Counterfactual-Hallucination-Detect"
        could not be independently verified.  Confirm before enabling.
        """
        logger.warning(
            "cf_detect.checkpoint is set to %r but the external model is not "
            "implemented — the repo ID could not be verified.  cf_detect_score() "
            "will return NaN.  Use the local counterfactual signal instead.",
            checkpoint,
        )
        # TODO: Once verified, load the model here:
        #
        #   from transformers import AutoModelForXxx, AutoProcessor
        #   self._cf_detect_processor = AutoProcessor.from_pretrained(checkpoint)
        #   self._cf_detect_model = AutoModelForXxx.from_pretrained(checkpoint)
        #       .to(self.cfg.get("device", "cuda")).eval()

    # -------------------------------------------------------- local sensitivity

    def sensitivity(
        self,
        image: Image.Image,
        claim_text: str,
        box: Box | None = None,
        prompt: str = "Describe this image.",
    ) -> float:
        """
        Returns sensitivity in [0, 1]. HIGH = claim depended on the region =
        well grounded.

        ``box`` should be the best-matching region from the grounding module.
        If it is None — no localized referent was found — this returns NaN
        rather than masking the whole image, because a whole-image mask measures
        something different and would make every claim look sensitive.
        """
        if self.vlm is None:
            logger.warning(
                "local counterfactual requires a VLMWrapper; returning NaN"
            )
            return float("nan")

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
        except Exception as exc:
            logger.warning("local counterfactual scoring failed: %s", exc)
            return float("nan")

        if np.isnan(base.mean_logprob) or np.isnan(masked.mean_logprob):
            return float("nan")

        # Positive delta = confidence dropped when the region was hidden.
        delta = base.mean_logprob - masked.mean_logprob
        self._deltas.append(float(delta))
        return sensitivity_from_delta(delta, self.delta_scale)

    # -------------------------------------------------------- control sensitivity

    def control_sensitivity(
        self,
        image: Image.Image,
        claim_text: str,
        prompt: str = "Describe this image.",
    ) -> float:
        """
        Mask a random region (not the claim's referent) and measure sensitivity.

        If the claim's sensitivity is high but control sensitivity is also high,
        the VLM is sensitive to *any* region being masked — which is a sign of
        fragility rather than genuine grounding (the "binding error" signature
        from M4 in the reframed plan).

        Returns the mean control sensitivity over ``n_control_regions`` random
        masks.
        """
        if self.vlm is None:
            return float("nan")

        w, h = image.size
        if w < MIN_CONTROL_SIZE or h < MIN_CONTROL_SIZE:
            return float("nan")

        sensitivities: list[float] = []
        for _ in range(self.n_control_regions):
            box = self._random_box(w, h)
            s = self.sensitivity(image, claim_text, box=box, prompt=prompt)
            if not np.isnan(s):
                sensitivities.append(s)

        return float(np.mean(sensitivities)) if sensitivities else float("nan")

    def _random_box(self, img_w: int, img_h: int) -> Box:
        """Generate a random bounding box covering 5–15% of the image area."""
        frac = self._rng.uniform(0.05, 0.15)
        box_w = int(img_w * np.sqrt(frac))
        box_h = int(img_h * np.sqrt(frac))
        box_w = max(MIN_CROP_PX, min(box_w, img_w))
        box_h = max(MIN_CROP_PX, min(box_h, img_h))

        left = int(self._rng.integers(0, max(1, img_w - box_w)))
        top = int(self._rng.integers(0, max(1, img_h - box_h)))
        return (left, top, left + box_w, top + box_h)

    # -------------------------------------------------------- CF-Detect external

    def cf_detect_score(
        self,
        image: Image.Image,
        claim_text: str,
    ) -> float:
        """
        External CF-Detect model score.

        Returns NaN when the model is not loaded (the default, since the repo
        ID is unverified).

        When implemented, this should return a score in [0, 1] where
        HIGH = well grounded (matching the orientation contract).
        """
        if self._cf_detect_model is None:
            return float("nan")

        # TODO: Once the model is verified and loaded, implement inference here:
        #
        #   inputs = self._cf_detect_processor(
        #       images=image, text=claim_text, return_tensors="pt"
        #   )
        #   inputs = {k: v.to(device) for k, v in inputs.items()}
        #   with torch.no_grad():
        #       outputs = self._cf_detect_model(**inputs)
        #   score = ... # extract and orient the score
        #   return float(score)
        raise NotImplementedError(
            "CF-Detect external model inference not implemented — repo ID "
            "unverified.  Use the local counterfactual signal instead."
        )

    # -------------------------------------------------------- calibration

    def calibration_report(self) -> str:
        """
        Summary of observed deltas for calibrating ``delta_scale``.

        Print this at the end of a run and set ``counterfactual.delta_scale``
        near the reported p90|delta|.
        """
        if not self._deltas:
            return "Counterfactual: no deltas computed."
        a = np.asarray(self._deltas)
        pos = float((a > 0).mean()) * 100.0
        p90 = float(np.percentile(np.abs(a), 90))
        msg = (
            f"Counterfactual deltas over {len(a)} claims: "
            f"min={a.min():+.4f} median={np.median(a):+.4f} max={a.max():+.4f} "
            f"p90|delta|={p90:.4f}; {pos:.0f}% positive. "
            f"delta_scale={self.delta_scale:.4f}"
        )
        if p90 < 0.2 * self.delta_scale:
            msg += (
                "  <-- delta_scale is far too LARGE for this spread: "
                "sensitivities are all bunched near 0.5 and the signal is "
                "inert. Set counterfactual.delta_scale near the p90 above."
            )
        elif p90 > 5.0 * self.delta_scale:
            msg += (
                "  <-- delta_scale is too SMALL: tanh is saturating and most "
                "sensitivities are pinned near 0 or 1."
            )
        if 40.0 < pos < 60.0:
            msg += (
                "  NOTE: deltas are ~50/50 positive/negative, which is what "
                "pure noise looks like. Check that masking is actually changing "
                "the region before trusting this signal at all."
            )
        return msg


# Minimum image dimension for control-region masking to be meaningful
MIN_CROP_PX = 8
MIN_CONTROL_SIZE = 32
