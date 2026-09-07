"""LoRA data builder: (image, claim, frozen signals, mechanism) -> JSONL.

The finetune teaches Qwen to *diagnose*, not to see: the base model stays
FROZEN, small LoRA adapters learn the M0-M4 mapping from the claim plus the
already-extracted signal vector. Input format keeps the image (so the adapter
can ground words to pixels) and verbalizes the signals (so it need not
rediscover CLIP from scratch):

    <image> Claim: "There are 3 cats."
    Signals: evidence=0.81 similarity=0.77 uncertainty=0.12 count=3/3 vlm=YES
    Diagnose the failure mechanism with exactly one token: M0 M1 M2 M3 M4.
    Answer:

Target: the single mechanism token. Mechanism labels come from (in order of
trust): human adjudication > repair-outcome oracle (which repair actually
worked) > split prior (POPE adversarial ~ M1) > rule-based diagnose().
The source is recorded per row so training can weight or filter by provenance.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Claim: \"{claim}\"\n"
    "Signals: evidence={evidence} similarity={similarity}"
    " uncertainty={uncertainty} count={count} vlm={vlm}\n"
    "Diagnose the failure mechanism with exactly one token: M0 M1 M2 M3 M4.\n"
    "Answer:"
)

MECHANISMS = ("M0", "M1", "M2", "M3", "M4")


def _fmt_float(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if f != f else f"{f:.2f}"


def row_to_record(image: str, claim: str, mechanism: str, provenance: str,
                  evidence: float = float("nan"),
                  similarity: float = float("nan"),
                  uncertainty: float | None = None,
                  count: str = "n/a", vlm: str = "n/a") -> dict:
    """One instruction-tuning record. Raises on bad mechanism (fail loud)."""
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}, got {mechanism!r}")
    unc = "n/a" if uncertainty is None else _fmt_float(uncertainty)
    prompt = PROMPT_TEMPLATE.format(
        claim=claim.strip(), evidence=_fmt_float(evidence),
        similarity=_fmt_float(similarity), uncertainty=unc,
        count=count, vlm=vlm)
    return {"image": str(image), "prompt": prompt, "label": mechanism,
            "provenance": provenance}


def build_jsonl(rows: list[dict], out_path: str | Path) -> int:
    """rows: kwargs for row_to_record. Writes JSONL, returns row count.
    Balances classes by downsampling to the minority count and reports it."""
    from collections import Counter

    recs = [row_to_record(**r) for r in rows]
    counts = Counter(r["label"] for r in recs)
    floor = min(counts.values())
    logger.info("class counts %s -> balancing to %d", dict(counts), floor)
    by_label: dict[str, list[dict]] = {m: [] for m in MECHANISMS}
    for r in recs:
        if len(by_label[r["label"]]) < floor:
            by_label[r["label"]].append(r)
    balanced = [r for m in MECHANISMS for r in by_label[m]]
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in balanced) + "\n", encoding="utf-8")
    return len(balanced)
