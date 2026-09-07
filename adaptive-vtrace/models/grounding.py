"""
Region-level visual grounding for claim localisation.

Two backends:

  attention_rollout  (default) — uses the VLM's own attention to identify
                     which image patches a claim refers to, then cuts bounding
                     boxes around high-attention clusters.  No extra model.
  groundingdino      — uses IDEA-Research/grounding-dino-base to get phrase-
                     grounded bounding boxes.  Requires the `groundingdino`
                     package (~900 MB download on first run).

The module exposes a uniform API regardless of backend:

    regions = grounding.propose_regions(image, claim_text, prompt)
    crop, box = grounding.best_region(image, claim_text, prompt)

Every returned region carries its own (left, top, right, bottom) box so
nothing downstream has to reconstruct geometry from an index.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]

# Minimum crop side in pixels.  Anything smaller is dropped as uninformative.
MIN_CROP_PX = 8


@dataclass
class Region:
    """A crop of the source image together with its bounding box."""

    crop: Image.Image
    box: Box  # (left, top, right, bottom) in original-image coordinates
    score: float = 0.0  # relevance score; higher = better match to the claim


class GroundingModule:
    """Claim-to-region localisation, backend-agnostic."""

    VALID_BACKENDS = ("attention_rollout", "groundingdino")

    def __init__(self, cfg: dict, vlm=None):
        self.cfg = cfg
        gcfg = cfg["grounding"]
        self.backend = gcfg.get("backend", "attention_rollout")
        if self.backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"grounding.backend must be one of {self.VALID_BACKENDS}, "
                f"got {self.backend!r}"
            )
        self.vlm = vlm
        self.n_proposals = gcfg.get("n_region_proposals", 8)

        # GroundingDINO-specific
        self._gdino_model = None
        self._gdino_processor = None
        if self.backend == "groundingdino":
            self._load_groundingdino(gcfg)

    # ------------------------------------------------------------------ init

    def _load_groundingdino(self, gcfg: dict) -> None:
        checkpoint = gcfg.get(
            "groundingdino_checkpoint", "IDEA-Research/grounding-dino-base"
        )
        self.box_threshold = float(gcfg.get("box_threshold", 0.25))
        self.text_threshold = float(gcfg.get("text_threshold", 0.20))

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            logger.info("loading GroundingDINO from %s ...", checkpoint)
            device = self.cfg.get("device", "cuda")
            self._gdino_processor = AutoProcessor.from_pretrained(checkpoint)
            self._gdino_model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
                .to(device)
                .eval()
            )
            self._gdino_device = device
        except Exception as exc:
            logger.warning(
                "GroundingDINO failed to load (%s); falling back to "
                "attention_rollout for this run",
                exc,
            )
            self.backend = "attention_rollout"

    # --------------------------------------------------------- attention rollout

    def _rollout_regions(
        self, image: Image.Image, claim_text: str, prompt: str
    ) -> list[Region]:
        """
        Propose regions from the VLM's own visual attention.

        The VLM's attention_rollout method returns a 1-D array of attention
        weights over visual token positions.  We reshape it to a spatial grid
        (assuming square-ish patch layout), threshold, and cut bounding boxes
        around connected components of high attention.
        """
        if self.vlm is None:
            logger.warning("attention_rollout requires a VLMWrapper; returning []")
            return []

        attn = self.vlm.attention_rollout(image, prompt, claim_text)
        if attn is None or len(attn) == 0:
            return self._grid_fallback(image)

        # Reshape 1-D attention into a 2-D spatial grid.  Qwen2.5-VL uses
        # dynamic resolution, so the number of visual tokens varies.  Assume a
        # roughly square grid; the aspect-ratio error is tolerable for crop
        # proposals.
        n = len(attn)
        h = int(math.isqrt(n))
        w = h if h * h == n else n // max(h, 1)
        if h * w != n:
            # Non-square: fall back to the closest rectangle
            h = int(math.sqrt(n))
            w = n // max(h, 1)
            attn = attn[: h * w]

        if h == 0 or w == 0:
            return self._grid_fallback(image)

        attn_map = attn.reshape(h, w)

        # Normalize to [0, 1]
        a_min, a_max = float(attn_map.min()), float(attn_map.max())
        if a_max > a_min:
            attn_map = (attn_map - a_min) / (a_max - a_min)
        else:
            return self._grid_fallback(image)

        # Threshold at the 75th percentile to find high-attention clusters
        threshold = float(np.percentile(attn_map, 75))
        mask = attn_map >= threshold

        regions = self._boxes_from_mask(image, mask, attn_map, h, w)
        if not regions:
            return self._grid_fallback(image)

        # Sort by score descending, keep top N
        regions.sort(key=lambda r: r.score, reverse=True)
        return regions[: self.n_proposals]

    @staticmethod
    def _boxes_from_mask(
        image: Image.Image,
        mask: np.ndarray,
        attn_map: np.ndarray,
        grid_h: int,
        grid_w: int,
    ) -> list[Region]:
        """
        Cut bounding boxes from a binary attention mask using connected
        components (flood fill).  Each box is mapped back to pixel coordinates.
        """
        img_w, img_h = image.size
        cell_w = img_w / grid_w
        cell_h = img_h / grid_h

        visited = np.zeros_like(mask, dtype=bool)
        regions: list[Region] = []

        for i in range(grid_h):
            for j in range(grid_w):
                if mask[i, j] and not visited[i, j]:
                    # Flood-fill to find the connected component
                    component: list[tuple[int, int]] = []
                    stack = [(i, j)]
                    while stack:
                        ci, cj = stack.pop()
                        if (
                            0 <= ci < grid_h
                            and 0 <= cj < grid_w
                            and mask[ci, cj]
                            and not visited[ci, cj]
                        ):
                            visited[ci, cj] = True
                            component.append((ci, cj))
                            stack.extend(
                                [(ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)]
                            )

                    if not component:
                        continue

                    # Bounding box of the component in grid coords
                    rows = [c[0] for c in component]
                    cols = [c[1] for c in component]
                    r_min, r_max = min(rows), max(rows)
                    c_min, c_max = min(cols), max(cols)

                    # Map to pixel coords with a small margin
                    margin = 0.5  # half-cell margin
                    left = max(0, int((c_min - margin) * cell_w))
                    top = max(0, int((r_min - margin) * cell_h))
                    right = min(img_w, int((c_max + 1 + margin) * cell_w))
                    bottom = min(img_h, int((r_max + 1 + margin) * cell_h))

                    if right - left < MIN_CROP_PX or bottom - top < MIN_CROP_PX:
                        continue

                    box: Box = (left, top, right, bottom)
                    score = float(np.mean([attn_map[r, c] for r, c in component]))
                    regions.append(
                        Region(crop=image.crop(box), box=box, score=score)
                    )

        return regions

    def _grid_fallback(self, image: Image.Image, grid: int = 3) -> list[Region]:
        """
        When attention rollout fails or is unavailable, fall back to a simple
        overlapping grid — the same approach vtrace-plus uses by default.
        """
        w, h = image.size
        regions: list[Region] = []
        overlap = 0.2

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
                    regions.append(Region(crop=image.crop(box), box=box, score=0.0))

        return regions

    # --------------------------------------------------------- GroundingDINO

    def _gdino_regions(
        self, image: Image.Image, claim_text: str
    ) -> list[Region]:
        """
        Use GroundingDINO to propose phrase-grounded bounding boxes for the
        claim text.
        """
        import torch

        if self._gdino_model is None or self._gdino_processor is None:
            logger.warning("GroundingDINO not loaded; falling back to grid")
            return self._grid_fallback(image)

        try:
            inputs = self._gdino_processor(
                images=image, text=claim_text, return_tensors="pt"
            )
            inputs = {k: v.to(self._gdino_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._gdino_model(**inputs)

            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]],  # (height, width)
            )[0]

            regions: list[Region] = []
            img_w, img_h = image.size

            for box_tensor, score_tensor in zip(
                results["boxes"], results["scores"]
            ):
                # post_process returns xyxy format
                coords = box_tensor.cpu().tolist()
                box: Box = (
                    max(0, int(coords[0])),
                    max(0, int(coords[1])),
                    min(img_w, int(coords[2])),
                    min(img_h, int(coords[3])),
                )
                if box[2] - box[0] < MIN_CROP_PX or box[3] - box[1] < MIN_CROP_PX:
                    continue
                regions.append(
                    Region(
                        crop=image.crop(box),
                        box=box,
                        score=float(score_tensor.cpu()),
                    )
                )

            if not regions:
                logger.debug(
                    "GroundingDINO found no boxes for %r; falling back to grid",
                    claim_text[:60],
                )
                return self._grid_fallback(image)

            regions.sort(key=lambda r: r.score, reverse=True)
            return regions[: self.n_proposals]

        except Exception as exc:
            logger.warning("GroundingDINO inference failed (%s); using grid", exc)
            return self._grid_fallback(image)

    # --------------------------------------------------------- public API

    def propose_regions(
        self, image: Image.Image, claim_text: str, prompt: str = ""
    ) -> list[Region]:
        """
        Return candidate regions for the claim, sorted by relevance (best first).

        Always includes the full image as region 0 so that callers needing a
        whole-image fallback don't have to special-case it.
        """
        # Full image is always region 0
        w, h = image.size
        full = Region(crop=image, box=(0, 0, w, h), score=0.0)

        if self.backend == "groundingdino":
            regions = self._gdino_regions(image, claim_text)
        else:
            regions = self._rollout_regions(image, claim_text, prompt)

        # Deduplicate: don't include the full image twice
        return [full] + [r for r in regions if r.box != (0, 0, w, h)]

    def best_region(
        self, image: Image.Image, claim_text: str, prompt: str = ""
    ) -> Region | None:
        """
        Return the single best-matching region crop for the claim, or None if
        only the full image matched (meaning no localized referent was found).

        Returning None rather than the full image is deliberate: masking the
        entire image in the counterfactual module tests something different from
        masking a localized region, and the caller needs to distinguish the two.
        """
        regions = self.propose_regions(image, claim_text, prompt)
        # Skip region 0 (the full image)
        crops = [r for r in regions[1:] if r.score > 0]
        if crops:
            return crops[0]  # already sorted by score
        # If attention rollout gave us grid regions (score=0), return the first
        non_full = [r for r in regions[1:]]
        return non_full[0] if non_full else None
