"""POPE benchmark loader (plan sections 6a, 10).

POPE's adversarial split queries frequently co-occurring but absent objects —
a prior-override (M1) probe by construction. Run the THREE-way ordering
random < popular < adversarial (monotone trend, plan section 6a), and report
yes-bias: a "yes" may reflect answer-format prior, not visual co-occurrence
prior — different mechanism, same label. Say so in the paper.

Expected entry (tolerant to key variants):
  {"question_id": ..., "image": "COCO_val2014_0000.jpg",
   "text": "Is there a fork in the image?", "label": "yes"/"no",
   "category": "random"/"popular"/"adversarial"}
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

CATEGORIES = ("random", "popular", "adversarial")


def _norm_label(entry: dict):
    for k in ("label", "answer", "a"):
        if k in entry:
            v = str(entry[k]).strip().lower()
            return 1 if v in ("yes", "1", "true") else 0
    return None


def _norm_category(entry: dict) -> str:
    for k in ("category", "split", "subset"):
        if k in entry:
            v = str(entry[k]).strip().lower()
            for c in CATEGORIES:
                if c in v:
                    return c
    return "unknown"


def load_pope(path: str | Path) -> list[dict]:
    """Load + normalize POPE entries. Skips entries missing question/image."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for k in ("data", "questions", "annotations"):
            if k in raw:
                raw = raw[k]
                break
    out = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        q = e.get("text", e.get("question", ""))
        img = e.get("image", e.get("image_id", e.get("filename", "")))
        if not q or not img:
            logger.debug("skipping entry without question/image: %s", e.get("question_id"))
            continue
        out.append({"question_id": e.get("question_id", e.get("id")),
                    "image": str(img), "question": str(q),
                    "label": _norm_label(e), "category": _norm_category(e)})
    return out


def three_way_summary(entries: list[dict]) -> dict:
    """Counts + yes-rate per split. Yes-rate far from 0.5 => yes-bias warning."""
    by = {c: [e for e in entries if e["category"] == c] for c in CATEGORIES}
    rep = {}
    for c, es in by.items():
        labs = [e["label"] for e in es if e["label"] is not None]
        rep[c] = {"n": len(es),
                  "yes_rate": (sum(labs) / len(labs)) if labs else None}
    yes_all = [e["label"] for e in entries if e["label"] is not None]
    rep["overall_yes_rate"] = (sum(yes_all) / len(yes_all)) if yes_all else None
    rep["category_counts"] = dict(Counter(e["category"] for e in entries))
    return rep
