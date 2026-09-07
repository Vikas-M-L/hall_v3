#!/usr/bin/env python3
"""
V-TRACE+ end-to-end pipeline. Inference only.

    image -> VLM response -> claims -> per-claim signals -> fusion -> risk

Usage
-----
    # single image
    python scripts/run_pipeline.py --image path/to/img.jpg

    # a directory of images
    python scripts/run_pipeline.py --images path/to/dir --out results/run1.json

    # compare fusion modes on the same signals (no model reload)
    python scripts/run_pipeline.py --image img.jpg --fusion-mode v1

    # exercise the fusion logic with no models and no GPU
    python scripts/run_pipeline.py --dry-run

Read the run summary, not just the scores. Every signal here can fail silently to
NaN, and a run where 80% of one signal is NaN produces confident-looking risk
numbers computed from whatever survived. The summary reports per-signal NaN rates
and the calibration ranges you need in order to set the CLIP window, delta_scale,
and CED_THRESHOLD. Until those are calibrated the absolute risk values mean
nothing — see README "Calibrate before you interpret".
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fusion.vtrace_fusion import (  # noqa: E402
    ALL_SIGNALS,
    SignalBundle,
    aggregate,
    fuse,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("vtrace")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# A signal missing this often is not "occasionally unavailable", it is broken.
NAN_RATE_ALARM = 0.25


# =========================================================================
# Signal collection
# =========================================================================


def collect_signals(image, claims, generation, cfg, models) -> list[SignalBundle]:
    """
    One SignalBundle per claim.

    Regions are proposed AND encoded once per image, not once per claim. With a
    3x3 grid that is 10 CLIP image encodes per image instead of 10 per claim,
    which is the difference between CLIP being free and CLIP dominating runtime.
    Each region carries its own box, so nothing downstream has to reconstruct
    geometry from an index.
    """
    clip = models["clip"]
    uniprobe = models["uniprobe"]
    counterfactual = models["counterfactual"]
    toggles = cfg["signals"]

    stubbed: list[str] = []
    if uniprobe is not None and uniprobe.is_stubbed:
        stubbed.append("uniprobe")
    if counterfactual is not None and counterfactual.mode == "placeholder":
        stubbed.append("counterfactual")

    regions = clip.encode_regions(image) if clip is not None else None

    want_evidence = toggles.get("evidence", True)
    want_similarity = toggles.get("clip_similarity", True)
    want_cf = toggles.get("counterfactual", True) and counterfactual is not None
    # The counterfactual signal needs the best-matching box even when the
    # `evidence` score itself is switched off, so the region lookup is driven by
    # either consumer rather than by the evidence toggle alone.
    need_box = want_cf and counterfactual.mode == "local"

    bundles: list[SignalBundle] = []
    for claim in claims:
        b = SignalBundle(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            claim_type=claim.claim_type,
            stubbed=tuple(stubbed),
        )

        # Character span in the response -> token indices in the generation.
        # Computed once; both `confidence` and UniProbe need it. claim.span is
        # None when the extractor could not locate the claim in the response, and
        # that must stay NaN rather than silently scoring the whole response.
        tok_span = None
        if generation is not None and claim.span is not None:
            tok_span = models["vlm"].char_span_to_token_indices(generation, *claim.span)

        # --- confidence: the VLM's own certainty over the claim's tokens ----
        if toggles.get("confidence", True) and tok_span is not None:
            b.confidence = generation.confidence_span(*tok_span)

        # --- CLIP signals ---------------------------------------------------
        box = None
        if clip is not None and regions is not None:
            if want_evidence or need_box:
                score, box, _whole_won = clip.evidence(claim.text, regions)
                if want_evidence:
                    b.evidence = score
            if want_similarity:
                b.clip_similarity = clip.similarity(claim.text, regions)

        # --- UniProbe -------------------------------------------------------
        if toggles.get("uniprobe", True) and uniprobe is not None:
            b.uniprobe = uniprobe.score(
                image=image,
                claim_text=claim.text,
                generation=generation,
                tok_span=tok_span,
            )

        # --- Counterfactual -------------------------------------------------
        # Mask the region CLIP thinks the claim refers to. `box` is None when no
        # localized referent was found; the wrapper returns NaN in that case
        # rather than masking the whole image, which measures something else.
        if want_cf:
            b.counterfactual = counterfactual.sensitivity(
                image, claim.text, box=box, prompt=cfg["prompt"]
            )

        bundles.append(b)

    return bundles


# =========================================================================
# Model loading
# =========================================================================


def validate_model_configs(cfg: dict) -> None:
    """
    Check the enum-ish config strings before anything downloads weights.

    The wrappers validate their own modes, but two of them are constructed only
    after the VLM is loaded, so a typo in `mask_mode` would surface ~16 GB and
    several minutes into the run. Importing these modules is free — they do no
    model loading at import time.
    """
    from models.counterfactual_wrapper import (
        _MASK_MODE_ALIASES,
        VALID_MASK_MODES,
        CounterfactualWrapper,
    )
    from models.uniprobe_wrapper import UniProbeWrapper

    u_raw = cfg["uniprobe"].get("mode", "placeholder")
    u_mode = UniProbeWrapper._ALIASES.get(u_raw, u_raw)
    if u_mode not in UniProbeWrapper.VALID_MODES:
        raise ValueError(
            f"uniprobe.mode must be one of {UniProbeWrapper.VALID_MODES}, got {u_raw!r}"
        )

    c_mode = cfg["counterfactual"].get("mode", "placeholder")
    if c_mode not in CounterfactualWrapper.VALID_MODES:
        raise ValueError(
            f"counterfactual.mode must be one of {CounterfactualWrapper.VALID_MODES}, "
            f"got {c_mode!r}"
        )

    m_raw = cfg["counterfactual"].get("mask_mode", "blur")
    if _MASK_MODE_ALIASES.get(m_raw, m_raw) not in VALID_MASK_MODES:
        raise ValueError(
            f"counterfactual.mask_mode must be one of {VALID_MASK_MODES} "
            f"(aliases: {sorted(_MASK_MODE_ALIASES)}), got {m_raw!r}"
        )


def load_models(cfg: dict, skip_vlm: bool = False) -> dict:
    from models.claim_extractor import ClaimExtractor
    from models.clip_wrapper import CLIPWrapper
    from models.counterfactual_wrapper import CounterfactualWrapper
    from models.uniprobe_wrapper import UniProbeWrapper
    from models.vlm_wrapper import VLMWrapper

    validate_model_configs(cfg)

    vlm = None
    if not skip_vlm:
        logger.info("loading VLM ...")
        vlm = VLMWrapper(cfg)
    else:
        logger.info("skipping VLM (--text mode): confidence + local counterfactual disabled")

    needs_clip = any(
        cfg["signals"].get(s, True) for s in ("evidence", "clip_similarity")
    ) or cfg["signals"].get("counterfactual", True)
    clip = None
    if needs_clip:
        logger.info("loading CLIP ...")
        clip = CLIPWrapper(cfg)

    # In text-only mode the VLM-dependent signals cannot run. Force them off
    # so fusion renormalizes onto CLIP only instead of scoring NaNs.
    if skip_vlm:
        for sig in ("confidence", "counterfactual"):
            if cfg["signals"].get(sig, True):
                logger.warning("disabling signal '%s': needs VLM, none loaded", sig)
                cfg["signals"][sig] = False

    return {
        "vlm": vlm,
        "clip": clip,
        "extractor": ClaimExtractor(cfg, vlm=vlm),
        "uniprobe": (
            UniProbeWrapper(cfg, vlm=vlm) if cfg["signals"].get("uniprobe", True) else None
        ),
        "counterfactual": (
            CounterfactualWrapper(cfg, vlm=vlm)
            if cfg["signals"].get("counterfactual", True)
            else None
        ),
    }


# =========================================================================
# Per-image driver
# =========================================================================


def process_image(image_path: Path, cfg: dict, models: dict, stats: Counter,
                    response_text: str | None = None) -> dict:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    prompt = cfg["prompt"]

    if response_text is not None:
        # Text-only mode: skip VLM generation entirely.
        generation = None
        response = response_text
        logger.info("[%s] using provided text: %s", image_path.name, response[:120])
    else:
        if models["vlm"] is None:
            raise ValueError("no --text given and no VLM loaded; provide --text or remove --no-vlm")
        generation = models["vlm"].generate(image, prompt)
        response = generation.text
        logger.info("[%s] response: %s", image_path.name, response[:120])

    claims = models["extractor"].extract(response, response_id=image_path.stem)
    logger.info("[%s] %d claims", image_path.name, len(claims))
    if not claims:
        stats["images_with_no_claims"] += 1
        return {
            "image": str(image_path),
            "response": response,
            "claims": [],
            "image_risk": float("nan"),
        }

    unlocated = sum(1 for c in claims if c.span is None)
    stats["claims_unlocated"] += unlocated
    stats["claims_total"] += len(claims)

    bundles = collect_signals(image, claims, generation, cfg, models)

    # Per-signal availability accounting. Without this you cannot distinguish
    # "the fusion says 0.6" from "four of five signals were NaN and the fusion
    # renormalized onto whatever was left".
    for b in bundles:
        raw = b.as_dict()
        for s in ALL_SIGNALS:
            if raw[s] != raw[s]:  # NaN
                stats[f"nan_{s}"] += 1

    results = [
        fuse(
            b,
            mode=cfg["fusion_mode"],
            active_signals=cfg["signals"],
            drop_stubbed=cfg.get("drop_stubbed", True),
        )
        for b in bundles
    ]
    # Negation correction (CLIP matches "no dog" to dogs: invert for negated
    # claims). Post-fusion so core weights are untouched; always flagged.
    from fusion.negation import apply_negation

    for b, r in zip(bundles, results):
        corr, neg = apply_negation(b.claim_text, r.risks.get("evidence", float("nan")),
                                   r.risks.get("clip_similarity", float("nan")))
        if neg and corr == corr:
            r.risk = corr
            r.rules_fired = [*r.rules_fired,
                             "negation-inverted (negated claim: match = risk)"]

    for r in results:
        if r.risk != r.risk:
            stats["claims_unscored"] += 1
        for rule in r.rules_fired:
            if rule.startswith("CED"):
                stats["rule1_fired"] += 1
            elif rule.startswith("disagreement"):
                stats["rule2_fired"] += 1

    image_risk = aggregate(
        [r.risk for r in results],
        method=cfg["aggregation"]["method"],
        topk=cfg["aggregation"]["topk"],
    )

    if cfg["output"].get("save_explanations", True):
        for r in results:
            print(r.explain(), "\n")

    return {
        "image": str(image_path),
        "response": response,
        "fusion_mode": cfg["fusion_mode"],
        "aggregation": cfg["aggregation"]["method"],
        "image_risk": image_risk,
        "claims": [{**c.to_dict(), **r.to_dict()} for c, r in zip(claims, results)],
    }


# =========================================================================
# Run summary
# =========================================================================


def run_summary(stats: Counter, models: dict | None, n_ok: int, n_failed: int) -> dict:
    """Everything you need to decide whether the numbers above are worth reading."""
    total = stats.get("claims_total", 0)
    lines = [
        "=" * 74,
        "RUN SUMMARY — read this before the scores",
        "=" * 74,
        f"images: {n_ok} processed, {n_failed} failed, "
        f"{stats.get('images_with_no_claims', 0)} yielded no claims",
        f"claims: {total} total, {stats.get('claims_unscored', 0)} unscored (all "
        f"signals NaN), {stats.get('claims_unlocated', 0)} could not be located in "
        f"the response",
    ]

    summary: dict = {
        "images_ok": n_ok,
        "images_failed": n_failed,
        "claims_total": total,
        "claims_unscored": stats.get("claims_unscored", 0),
        "claims_unlocated": stats.get("claims_unlocated", 0),
        "nan_rates": {},
        "rule1_fired": stats.get("rule1_fired", 0),
        "rule2_fired": stats.get("rule2_fired", 0),
    }

    if total:
        lines.append("per-signal NaN rate:")
        for s in ALL_SIGNALS:
            n = stats.get(f"nan_{s}", 0)
            rate = n / total
            summary["nan_rates"][s] = rate
            flag = "  <-- BROKEN?" if rate > NAN_RATE_ALARM else ""
            lines.append(f"    {s:<16} {n:>5}/{total}  ({rate:5.1%}){flag}")

        r1 = stats.get("rule1_fired", 0) / total
        r2 = stats.get("rule2_fired", 0) / total
        lines.append(f"rule 1 (CED) fired on {r1:.1%} of claims")
        lines.append(f"rule 2 (disagreement) fired on {r2:.1%} of claims")
        for name, rate in (("rule 1", r1), ("rule 2", r2)):
            if rate > 0.9 or (0 < rate < 0.02):
                lines.append(
                    f"    <-- {name} fires on {rate:.1%} of claims, so it is acting "
                    "as a constant, not a rule. v2 is then just a relabelled v1 "
                    "with different fixed weights. Recalibrate its threshold."
                )

    if models:
        for key in ("clip", "counterfactual"):
            m = models.get(key)
            if m is not None and hasattr(m, "calibration_report"):
                lines.append(m.calibration_report())
                if key == "clip" and hasattr(m, "suggested_window"):
                    sug = m.suggested_window()
                    if sug is not None:
                        lines.append(
                            f"SUGGESTED WINDOW: clip.cos_min={sug[0]} clip.cos_max={sug[1]} "
                            f"(from {len(m._cos_seen)} comparisons; set in config and re-run)"
                        )
                        summary["suggested_window"] = list(sug)

    lines.append("=" * 74)
    print("\n".join(lines))
    return summary


# =========================================================================
# Dry run — fusion only, no models
# =========================================================================


def dry_run(cfg: dict) -> None:
    """
    Exercise the fusion logic on hand-written signal values. No GPU, no model
    downloads. Verifies that orientation, rule firing, and renormalization all
    behave, and shows what each mode does with the same inputs.
    """
    cases = [
        (
            "grounded claim — everything agrees it is fine",
            SignalBundle(confidence=0.92, evidence=0.88, clip_similarity=0.81,
                         uniprobe=0.10, counterfactual=0.75),
        ),
        (
            "confident but ungrounded — Rule 1 should fire",
            SignalBundle(confidence=0.95, evidence=0.35, clip_similarity=0.40,
                         uniprobe=0.55, counterfactual=0.20),
        ),
        (
            "detectors contradict each other — Rule 2 should fire",
            SignalBundle(confidence=0.60, evidence=0.85, clip_similarity=0.50,
                         uniprobe=0.90, counterfactual=0.15),
        ),
        (
            "both rules fire",
            SignalBundle(confidence=0.93, evidence=0.30, clip_similarity=0.45,
                         uniprobe=0.85, counterfactual=0.80),
        ),
        (
            "half the signals missing",
            SignalBundle(confidence=0.70, evidence=float("nan"), clip_similarity=0.55,
                         uniprobe=float("nan"), counterfactual=float("nan")),
        ),
        (
            "stubs still fused — note Rule 2 is disabled, the stubs are constants",
            SignalBundle(confidence=0.95, evidence=0.35, clip_similarity=0.40,
                         uniprobe=0.50, counterfactual=0.50,
                         stubbed=("uniprobe", "counterfactual")),
        ),
    ]

    for label, bundle in cases:
        print("=" * 74)
        print(label)
        print("=" * 74)
        for mode in ("v1", "v2"):
            r = fuse(bundle, mode=mode, active_signals=cfg["signals"], drop_stubbed=False)
            print(r.explain())
            print()


# =========================================================================
# Config
# =========================================================================


REQUIRED_KEYS = (
    "fusion_mode",
    "signals",
    "vlm",
    "clip",
    "uniprobe",
    "counterfactual",
    "claim_extractor",
    "aggregation",
    "output",
)


def load_config(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    if cfg is None:
        raise ValueError(f"config {path} is empty")
    if not isinstance(cfg, dict):
        raise TypeError(f"config {path} must be a mapping, got {type(cfg).__name__}")

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"config {path} is missing required keys: {missing}")

    cfg.setdefault("prompt", "Describe this image in detail.")
    cfg.setdefault("drop_stubbed", True)
    cfg.setdefault("seed", 0)
    if cfg["fusion_mode"] not in ("v1", "v2"):
        raise ValueError(f"fusion_mode must be v1 or v2, got {cfg['fusion_mode']!r}")
    for key in ("method", "topk"):
        if key not in cfg["aggregation"]:
            raise KeyError(f"config {path}: aggregation.{key} is required")
    return cfg


def seed_everything(seed: int) -> None:
    """
    Greedy decoding is deterministic, but mask noise, any sampling temperature,
    and library-internal RNGs are not. Seed them so two runs of the same config
    are comparable — otherwise a v1-vs-v2 delta cannot be told from run noise.
    """
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def json_safe(obj):
    """Replace NaN with null. json.dumps emits bare `NaN`, which is not valid JSON."""
    if isinstance(obj, float):
        return None if obj != obj else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description="V-TRACE+ inference pipeline")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--image", help="single image path")
    ap.add_argument("--images", help="directory of images")
    ap.add_argument("--prompt", help="override the prompt in config")
    ap.add_argument("--fusion-mode", choices=["v1", "v2"], help="override config")
    ap.add_argument("--limit", type=int, help="stop after N images")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--dry-run", action="store_true", help="fusion only, no models")
    ap.add_argument("--text", default=None,
                    help="response text to score; skips VLM generation (Image -> Claims -> CLIP -> fusion -> JSON)")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="override config device; use cpu for lightweight run")
    ap.add_argument("--clip-model", default=None,
                    help="override clip.checkpoint, e.g. openai/clip-vit-base-patch32 for CPU speed")
    ap.add_argument("--cos-window", nargs=2, type=float, metavar=("MIN", "MAX"),
                    help="override clip.cos_min/cos_max, e.g. --cos-window -0.10 0.12 for SigLIP")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.prompt:
        cfg["prompt"] = args.prompt
    if args.fusion_mode:
        cfg["fusion_mode"] = args.fusion_mode
    if args.device:
        cfg["device"] = args.device
    if args.clip_model:
        cfg["clip"]["checkpoint"] = args.clip_model
    if args.cos_window:
        lo, hi = args.cos_window
        if hi <= lo:
            ap.error("cos-window MAX must exceed MIN")
        cfg["clip"]["cos_min"], cfg["clip"]["cos_max"] = lo, hi
    # Auto CPU fallback: default config says cuda but most laptops have none.
    if cfg.get("device") == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("cuda unavailable, falling back to cpu")
                cfg["device"] = "cpu"
        except ImportError:
            cfg["device"] = "cpu"

    if args.dry_run:
        dry_run(cfg)
        return 0

    if not args.image and not args.images:
        ap.error("need --image, --images, or --dry-run")

    skip_vlm = args.text is not None

    if args.image:
        paths = [Path(args.image)]
    else:
        paths = sorted(
            p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        logger.error("no images found")
        return 1

    seed_everything(int(cfg["seed"]))
    models = load_models(cfg, skip_vlm=skip_vlm)

    stats: Counter = Counter()
    results = []
    n_failed = 0
    for path in paths:
        try:
            results.append(process_image(path, cfg, models, stats,
                                         response_text=args.text))
        except Exception:
            n_failed += 1
            logger.exception("failed on %s", path)

    summary = run_summary(stats, models, len(results), n_failed)

    out_path = (
        Path(args.out) if args.out else Path(cfg["output"]["results_dir"]) / "results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            json_safe({"summary": summary, "config": cfg, "results": results}),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %d results to %s", len(results), out_path)

    scored = [r["image_risk"] for r in results if r["image_risk"] == r["image_risk"]]
    if scored:
        logger.info(
            "image risk: mean=%.3f  min=%.3f  max=%.3f",
            sum(scored) / len(scored), min(scored), max(scored),
        )

    # Exit non-zero when nothing usable came out. Returning 0 after every image
    # failed makes the pipeline look healthy to any caller that checks the exit
    # code, and produces an empty results file that reads as "no risk found".
    if not results:
        logger.error("every image failed; nothing was written but the summary")
        return 1
    if not scored:
        logger.error("no image produced a usable risk score")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
