"""Open-vocabulary detection + counting (fixes CLIP count/category blindness).

CLIP matches "2 dogs" to a photo of 2 cats (animal-ness, number-blindness).
OWL-ViT draws one box per instance: compare the *counted* boxes against the
*claimed* number and category instead of comparing embedding soup.

Number parsing is regex over words one..twelve + digits. Category guess strips
the leading count and articles ("2 dogs" -> "dogs", "a red car" -> "red car").
Both are deliberately dumb and fully tested; the detector does the seeing.
"""
from __future__ import annotations

import logging
import re

import torch

logger = logging.getLogger(__name__)

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
             "twelve": 12, "a": 1, "an": 1}

_COUNT_RE = re.compile(
    r"\b(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
    r"\s+(?P<label>[a-z][a-z\- ]*?)(?=\s+(?:in|on|at|near|by|with|and|or)\b|[.,;!?]|$)",
    re.IGNORECASE)


def parse_count(claim: str) -> tuple[int | None, str | None]:
    """("2 dogs in the room") -> (2, "dogs"). Returns (None, None) when no
    countable noun phrase is found."""
    m = _COUNT_RE.search(claim or "")
    if not m:
        return None, None
    raw = m.group("num").lower()
    num = int(raw) if raw.isdigit() else NUM_WORDS.get(raw)
    label = re.sub(r"^(a|an|the)\s+", "", m.group("label").strip(), flags=re.I)
    if not label:
        return None, None
    return num, label


class OWLDetector:
    """Thin OWL-ViT wrapper. Loads once (~180s first CPU load, cached after);
    detection itself is seconds per image."""

    def __init__(self, checkpoint: str = "google/owlvit-base-patch32"):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.processor = OwlViTProcessor.from_pretrained(checkpoint)
        self.model = OwlViTForObjectDetection.from_pretrained(
            checkpoint, dtype=torch.float32).eval()

    @torch.no_grad()
    def detect(self, image, labels: list[str],
               threshold: float = 0.15) -> dict[str, dict]:
        """labels -> {count, top_score, boxes}. Boxes in image pixel coords."""
        if not labels:
            return {}
        inputs = self.processor(text=[[f"a {lab}" for lab in labels]],
                                images=image, return_tensors="pt")
        out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, threshold=threshold, target_sizes=[image.size[::-1]])[0]
        n_lab = len(labels)
        per = {lab: {"count": 0, "top_score": 0.0, "boxes": []} for lab in labels}
        for box, score, lab_idx in zip(res["boxes"], res["scores"], res["labels"]):
            lab = labels[int(lab_idx) % n_lab]
            x0, y0, x1, y1 = (int(v) for v in box.tolist())
            per[lab]["count"] += 1
            per[lab]["top_score"] = max(per[lab]["top_score"], float(score))
            per[lab]["boxes"].append((x0, y0, x1, y1))
        return per


def count_verdict(expected_num: int, expected_label: str,
                  detected: dict) -> dict:
    """Compare claimed count vs detected boxes. Returns a verdict dict with a
    plain-English reason — shown in the UI next to the CLIP score."""
    got = detected.get("count", 0)
    if got == expected_num and expected_num > 0:
        return {"match": True,
                "reason": f"found {got} x '{expected_label}' as claimed"}
    if got == 0:
        return {"match": False,
                "reason": f"no '{expected_label}' detected (claimed {expected_num})"}
    return {"match": False,
            "reason": f"found {got} x '{expected_label}', claimed {expected_num}"}
