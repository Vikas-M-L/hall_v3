"""
CLIP / SigLIP wrapper — frozen, inference only.

Two signals come out of here:

  evidence     - max similarity between the claim and any candidate image region
  similarity   - similarity between the claim and the whole image

=============================================================================
CALIBRATION WARNING — read before interpreting any number from this file

Raw CLIP image-text cosines do NOT span [-1, 1]. In practice they occupy a
narrow band, roughly 0.15-0.35 for CLIP ViT-L/14. Mapping them with the obvious
(cos + 1) / 2 crushes every claim into ~0.575-0.675, which:

  * makes `evidence` and `clip_similarity` near-constant, so they carry almost
    no information while still consuming fusion weight, and
  * makes CED = confidence - evidence meaningless, because it subtracts a token
    probability (typically 0.85-0.98 under greedy decoding) from a number pinned
    near 0.62. CED would then sit around 0.23-0.36 for every claim, straddling
    CED_THRESHOLD = 0.25 by accident of the rescaling constant rather than
    because anything is ungrounded.

So the mapping here is an explicit window [cos_min, cos_max] from config, not an
affine map of the full cosine range. THE SHIPPED DEFAULTS ARE A GUESS. Run once,
read the observed-cosine range this wrapper reports at the end of the run, set
`clip.cos_min` / `clip.cos_max` from it, and only then look at CED or set
CED_THRESHOLD. Until you have done that, treat every fused score as unmeaning.
=============================================================================

Region proposals are grid crops rather than GroundingDINO output. That is a
deliberate choice: grid crops need no extra model, no extra download, and no
guessing at a third-party API. They are coarser than real grounding boxes, so if
you later wire in a detector, replace `propose_regions` and nothing else changes.

Each region carries its own box, so there is no separate index-to-box inverse to
keep in sync. Region 0 is always the full image, which is also what `similarity`
uses — so a claim costs one text encode, and an image costs one batch of image
encodes no matter how many claims it has.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]

# Minimum crop side in pixels. Anything smaller is dropped as uninformative.
# Because each region carries its own box, dropping one cannot desynchronize
# anything downstream.
MIN_CROP_PX = 8


def propose_regions(
    image: Image.Image, grid: int = 3, overlap: float = 0.2
) -> list[tuple[Image.Image, Box]]:
    """
    Returns [(full_image, full_box)] followed by overlapping grid crops, each
    paired with its own (left, top, right, bottom) box in original-image coords.

    Overlap matters: a claim's referent that straddles a hard grid boundary would
    otherwise be split across two crops and score poorly in both.
    """
    w, h = image.size
    out: list[tuple[Image.Image, Box]] = [(image, (0, 0, w, h))]
    if grid <= 1:
        return out

    step_x, step_y = w / grid, h / grid
    pad_x, pad_y = step_x * overlap, step_y * overlap

    for i in range(grid):
        for j in range(grid):
            box: Box = (
                max(0, int(i * step_x - pad_x)),
                max(0, int(j * step_y - pad_y)),
                min(w, int((i + 1) * step_x + pad_x)),
                min(h, int((j + 1) * step_y + pad_y)),
            )
            if box[2] - box[0] > MIN_CROP_PX and box[3] - box[1] > MIN_CROP_PX:
                out.append((image.crop(box), box))
    return out


@dataclass
class RegionSet:
    """Regions for one image, encoded once and reused across all its claims."""

    boxes: list[Box]
    embeds: torch.Tensor = field(repr=False)  # (n_regions, dim), unit-normed

    def __len__(self) -> int:
        return len(self.boxes)


class CLIPWrapper:
    def __init__(self, cfg: dict):
        from transformers import AutoModel, AutoProcessor

        self.cfg = cfg
        ccfg = cfg["clip"]
        self.backend = ccfg.get("backend", "transformers")
        self.device = cfg.get("device", "cuda")
        self.grid = ccfg.get("region_grid", 3)
        self.overlap = ccfg.get("region_overlap", 0.2)
        self.cos_min = float(ccfg.get("cos_min", 0.10))
        self.cos_max = float(ccfg.get("cos_max", 0.40))
        if self.cos_max <= self.cos_min:
            raise ValueError("clip.cos_max must exceed clip.cos_min")
        # Prompt ensembling: each claim is embedded under every template and the
        # text embeddings are averaged before the cosine. Cost is a few extra
        # *text* encodes (cheap); image encodes are untouched. "{}" = raw claim.
        self.templates: list[str] = list(
            ccfg.get("templates", ["{}", "a photo of {}", "a picture showing {}"])
        ) or ["{}"]
        checkpoint = ccfg["checkpoint"]

        # Honour the global dtype, and force fp32 on CPU: most CPU kernels have
        # no Half implementation, so a hardcoded fp16 CLIP makes device: cpu
        # crash with "addmm_impl_cpu_ not implemented for 'Half'".
        requested = cfg.get("dtype", "float16")
        if str(self.device).startswith("cpu"):
            self.dtype = torch.float32
        else:
            self.dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[requested]

        # Observed raw-cosine range across the whole run, for calibration.
        self._cos_seen: list[float] = []

        if self.backend == "transformers":
            self.model = (
                AutoModel.from_pretrained(checkpoint, torch_dtype=self.dtype)
                .to(self.device)
                .eval()
            )
            self.processor = AutoProcessor.from_pretrained(checkpoint)
            # SigLIP was trained with fixed-length padding; padding=True quietly
            # degrades its text embeddings.
            self._text_padding = (
                "max_length" if "siglip" in str(checkpoint).lower() else True
            )
        elif self.backend == "open_clip":
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                checkpoint, pretrained="openai"
            )
            self.model = self.model.to(self.device).to(self.dtype).eval()
            self.tokenizer = open_clip.get_tokenizer(checkpoint)
            self._text_padding = True
        else:
            raise ValueError(f"unknown clip backend: {self.backend}")

    # ------------------------------------------------------------------ encode

    @staticmethod
    def _as_tensor(out) -> "torch.Tensor":
        """Unwrap transformers model outputs (BaseModelOutputWithPooling, etc.)
        to a plain tensor. `get_image_features` returns a tensor on some
        versions and an output object on others."""
        import torch as _torch

        if isinstance(out, _torch.Tensor):
            return out
        for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            if hasattr(out, attr):
                val = getattr(out, attr)
                if isinstance(val, _torch.Tensor):
                    # last_hidden_state needs pooling; embeds do not.
                    if attr == "last_hidden_state":
                        return val[:, 0, :]
                    return val
        try:
            return out[0]
        except Exception:
            raise TypeError(f"cannot unwrap CLIP output of type {type(out)}")

    @torch.no_grad()
    def _embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        if self.backend == "transformers":
            inputs = self.processor(images=images, return_tensors="pt")
            px = inputs["pixel_values"].to(self.device, dtype=self.dtype)
            feats = self._as_tensor(self.model.get_image_features(pixel_values=px))
        else:
            px = torch.stack([self.preprocess(im) for im in images])
            px = px.to(self.device, dtype=self.dtype)
            feats = self.model.encode_image(px)
        feats = feats.float()
        return feats / feats.norm(dim=-1, keepdim=True)

    def _expand_templates(self, texts: list[str]) -> tuple[list[str], int, int]:
        """Expand texts x templates. Returns (expanded, n_texts, n_templates)."""
        expanded: list[str] = []
        for t in texts:
            for tpl in self.templates:
                expanded.append(tpl.format(t) if "{}" in tpl else tpl)
        return expanded, len(texts), len(self.templates)

    @torch.no_grad()
    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        expanded, n, k = self._expand_templates(texts)
        if self.backend == "transformers":
            inputs = self.processor(
                text=expanded,
                return_tensors="pt",
                padding=self._text_padding,
                truncation=True,
            )
            feats = self._as_tensor(
                self.model.get_text_features(
                    input_ids=inputs["input_ids"].to(self.device),
                    attention_mask=(
                        inputs["attention_mask"].to(self.device)
                        if "attention_mask" in inputs
                        else None
                    ),
                )
            )
        else:
            feats = self.model.encode_text(self.tokenizer(expanded).to(self.device))
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        if k > 1:
            # Mean-pool the k template variants of each text, then renormalize.
            feats = feats.view(n, k, -1).mean(dim=1)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_regions(
        self, image: Image.Image, grid: int | None = None, overlap: float | None = None
    ) -> RegionSet:
        """
        Propose and encode all regions for one image. Call this ONCE per image and
        pass the result to every claim — it is the whole reason the per-claim cost
        is one text encode instead of twenty-one image encodes.
        """
        pairs = propose_regions(
            image,
            self.grid if grid is None else grid,
            self.overlap if overlap is None else overlap,
        )
        crops = [p[0] for p in pairs]
        boxes = [p[1] for p in pairs]
        return RegionSet(boxes=boxes, embeds=self._embed_images(crops))

    # ------------------------------------------------------------- normalization

    def _to_unit(self, cosine: float) -> float:
        """
        Raw cosine -> [0, 1] via the configured window. See the calibration
        warning at the top of this file: the window, not the [-1,1] range, is
        what determines whether these scores have any spread.
        """
        self._cos_seen.append(float(cosine))
        scaled = (float(cosine) - self.cos_min) / (self.cos_max - self.cos_min)
        return float(np.clip(scaled, 0.0, 1.0))

    def observed_cosine_range(self) -> tuple[float, float, float] | None:
        """(min, median, max) of every raw cosine seen so far, or None."""
        if not self._cos_seen:
            return None
        a = np.asarray(self._cos_seen)
        return float(a.min()), float(np.median(a)), float(a.max())

    def suggested_window(self) -> tuple[float, float] | None:
        """Data-driven [cos_min, cos_max]: 5th/95th percentile of observed raw
        cosines, with a small margin. Returns None until 4+ comparisons seen."""
        if len(self._cos_seen) < 4:
            return None
        a = np.asarray(self._cos_seen)
        lo, hi = float(np.percentile(a, 5)), float(np.percentile(a, 95))
        if hi - lo < 0.02:  # degenerate: everything identical, keep current window
            return None
        return (round(lo - 0.01, 4), round(hi + 0.01, 4))

    def calibration_report(self) -> str:
        rng = self.observed_cosine_range()
        if rng is None:
            return "CLIP: no cosines computed."
        lo, mid, hi = rng
        clipped = sum(
            1 for c in self._cos_seen if c <= self.cos_min or c >= self.cos_max
        )
        pct = 100.0 * clipped / len(self._cos_seen)
        msg = (
            f"CLIP raw cosines over {len(self._cos_seen)} comparisons: "
            f"min={lo:.4f} median={mid:.4f} max={hi:.4f}. "
            f"Configured window [{self.cos_min:.4f}, {self.cos_max:.4f}] "
            f"saturated {pct:.1f}% of them."
        )
        if pct > 5.0:
            msg += (
                "  <-- RECALIBRATE: set clip.cos_min/cos_max from the observed "
                "range or your scores are being flattened at the ends."
            )
        span = hi - lo
        if span < 0.5 * (self.cos_max - self.cos_min):
            msg += (
                f"  <-- Observed span {span:.4f} is much narrower than the "
                "window, so scores are compressed into the middle. Narrow the "
                "window to restore spread."
            )
        return msg

    # ------------------------------------------------------------------ signals

    def similarity(self, claim: str, regions: RegionSet) -> float:
        """
        Claim vs. the WHOLE image. Higher = better semantic match.

        Region 0 is the full image by construction, so this reuses the already
        computed embedding rather than encoding the image again.
        """
        try:
            txt = self._embed_texts([claim])
            return self._to_unit(float(regions.embeds[0] @ txt[0]))
        except Exception as exc:
            logger.warning("CLIP similarity failed: %s", exc)
            return float("nan")

    def evidence(self, claim: str, regions: RegionSet) -> tuple[float, Box | None, bool]:
        """
        Max claim-to-region similarity, the box that achieved it, and whether
        the WHOLE image won (third element).

        Higher = stronger visual evidence that the claim's referent is actually
        present somewhere in the image. Returns (nan, None, False) on failure.

        The box is the best-matching CROP where one exists. If the full image
        wins (region 0), the box falls back to the best actual crop — the
        counterfactual module needs *something* to mask — but whole_won=True
        tells display consumers NOT to draw it as "the" referent box: the claim
        matched globally, and the fallback crop would be a misleading ROI.
        """
        try:
            txt = self._embed_texts([claim])
            cos = (regions.embeds @ txt[0]).cpu().numpy()
            idx = int(np.argmax(cos))
            score = self._to_unit(float(cos[idx]))

            if idx == 0 and len(regions) > 1:
                # Whole image won. Fall back to the best actual crop for the box,
                # so the counterfactual signal still has something to mask.
                crop_idx = int(np.argmax(cos[1:])) + 1
                return score, regions.boxes[crop_idx], True
            if idx == 0:
                return score, None, True
            return score, regions.boxes[idx], False
        except Exception as exc:
            logger.warning("CLIP evidence failed: %s", exc)
            return float("nan"), None, False
