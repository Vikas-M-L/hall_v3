"""CLIP / SigLIP image-text similarity — global and region-level."""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPScorer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        ccfg = cfg["clip"]
        self.backend = ccfg.get("backend", "transformers")
        self.device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        checkpoint = ccfg["checkpoint"]

        # CPU has no Half kernels; force float32 there
        clip_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if self.backend == "transformers":
            from transformers import AutoModel, AutoProcessor

            self.model = AutoModel.from_pretrained(checkpoint, torch_dtype=clip_dtype)
            self.model = self.model.to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(checkpoint)
            self.is_siglip = "siglip" in checkpoint.lower()
        elif self.backend == "open_clip":
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                checkpoint, pretrained="openai"
            )
            self.model = self.model.to(self.device).eval()
            self.tokenizer = open_clip.get_tokenizer(checkpoint)
            self.is_siglip = False
        else:
            raise ValueError(f"unknown clip backend {self.backend}")

    @torch.no_grad()
    def _embed(self, images: list[Image.Image], texts: list[str]):
        if self.backend == "transformers":
            inputs = self.processor(
                text=texts, images=images, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            img = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            txt = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        else:
            px = torch.stack([self.preprocess(im) for im in images]).to(self.device)
            tok = self.tokenizer(texts).to(self.device)
            img = self.model.encode_image(px)
            txt = self.model.encode_text(tok)

        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return img.float(), txt.float()

    def similarity(self, image: Image.Image, text: str) -> float:
        """Global claim-to-whole-image similarity, mapped to [0, 1]."""
        try:
            img, txt = self._embed([image], [text])
            cos = float((img @ txt.T)[0, 0])
            return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))
        except Exception as exc:
            logger.warning("CLIP similarity failed: %s", exc)
            return float("nan")

    def region_similarities(self, regions: list[Image.Image], text: str) -> list[float]:
        """Similarity of one claim against each candidate region crop."""
        if not regions:
            return []
        try:
            img, txt = self._embed(regions, [text])
            cos = (img @ txt.T).squeeze(-1)
            return [float(np.clip((float(c) + 1.0) / 2.0, 0.0, 1.0)) for c in cos]
        except Exception as exc:
            logger.warning("CLIP region similarity failed: %s", exc)
            return [float("nan")] * len(regions)

    def evidence_grounding(self, regions: list[Image.Image], text: str) -> float:
        """Max over region similarities — the visual evidence grounding signal."""
        sims = [s for s in self.region_similarities(regions, text) if not np.isnan(s)]
        return max(sims) if sims else float("nan")

    @torch.no_grad()
    def text_similarity(self, a: str, b: str) -> float:
        """Text-text cosine, used by the self-consistency signal."""
        try:
            _, txt = self._embed([Image.new("RGB", (224, 224))], [a, b])
            return float(np.clip((float(txt[0] @ txt[1]) + 1.0) / 2.0, 0.0, 1.0))
        except Exception as exc:
            logger.warning("CLIP text similarity failed: %s", exc)
            return float("nan")
