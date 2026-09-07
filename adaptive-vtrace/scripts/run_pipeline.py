#!/usr/bin/env python3
"""
Adaptive V-TRACE+ pipeline — feature extraction and inference.

    image → VLM response → claims → per-claim signals → feature vector → output

Usage
-----
    # Single image — extract all signals and print per-claim features
    python scripts/run_pipeline.py --image path/to/img.jpg

    # Directory of images
    python scripts/run_pipeline.py --images path/to/dir --out results/run1.json

    # Dry run — test model loading and signal extraction with no image
    python scripts/run_pipeline.py --dry-run

    # Override config
    python scripts/run_pipeline.py --image img.jpg --config configs/custom.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("adaptive-vtrace")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================================
# Model loading
# =========================================================================


def load_models(cfg: dict) -> dict:
    """Load all models needed for the configured signals."""
    from models.vlm_wrapper import VLMWrapper

    logger.info("loading VLM ...")
    vlm = VLMWrapper(cfg)

    # CLIP
    clip = None
    needs_clip = any(
        cfg["signals"].get(s, True)
        for s in ("evidence_grounding", "clip_similarity", "self_consistency")
    )
    if needs_clip:
        from models.clip_similarity import CLIPScorer

        logger.info("loading CLIP ...")
        clip = CLIPScorer(cfg)

    # Claim extractor
    from models.claim_extractor import ClaimExtractor

    extractor = ClaimExtractor(cfg, vlm=vlm)

    # Grounding
    grounding = None
    if cfg["signals"].get("evidence_grounding", True) or cfg["signals"].get(
        "counterfactual", True
    ):
        from models.grounding import GroundingModule

        logger.info("loading grounding (%s) ...", cfg["grounding"]["backend"])
        grounding = GroundingModule(cfg, vlm=vlm)

    # UniProbe
    uniprobe = None
    if cfg["signals"].get("uniprobe", True):
        from models.uniprobe_wrapper import UniProbeWrapper

        uniprobe = UniProbeWrapper(cfg, vlm=vlm)
        logger.info("uniprobe: mode=%s, stubbed=%s", uniprobe.mode, uniprobe.is_stubbed)

    # Counterfactual
    counterfactual = None
    if cfg["signals"].get("counterfactual", True):
        from models.counterfactual import CounterfactualModule

        counterfactual = CounterfactualModule(cfg, vlm=vlm, grounding=grounding)
        logger.info("counterfactual: mask_mode=%s", counterfactual.mask_mode)

    return {
        "vlm": vlm,
        "clip": clip,
        "extractor": extractor,
        "grounding": grounding,
        "uniprobe": uniprobe,
        "counterfactual": counterfactual,
    }


# =========================================================================
# Signal extraction
# =========================================================================


def extract_signals(image, claim, generation, cfg, models) -> dict:
    """
    Extract all configured signals for a single claim.

    Returns a flat dict of signal_name → float (or NaN if unavailable).
    """
    vlm = models["vlm"]
    clip = models["clip"]
    grounding = models["grounding"]
    uniprobe = models["uniprobe"]
    counterfactual = models["counterfactual"]
    signals = cfg["signals"]
    prompt = cfg["prompt"]

    feats: dict[str, float] = {}

    # Token span for this claim
    tok_span = None
    if generation is not None and claim.span is not None:
        tok_span = vlm.char_span_to_token_indices(generation, *claim.span)

    # --- VLM confidence ---
    if signals.get("vlm_confidence", True) and tok_span is not None:
        lps = generation.token_logprobs[tok_span[0] : tok_span[1]]
        if lps:
            feats["vlm_confidence"] = float(np.clip(np.exp(np.mean(lps)), 0.0, 1.0))
        else:
            feats["vlm_confidence"] = float("nan")
    elif signals.get("vlm_confidence", True):
        feats["vlm_confidence"] = float("nan")

    # --- CLIP similarity ---
    if signals.get("clip_similarity", True) and clip is not None:
        feats["clip_similarity"] = clip.similarity(image, claim.text)

    # --- Evidence grounding ---
    best_box = None
    if grounding is not None:
        best_region = grounding.best_region(image, claim.text, prompt)
        if best_region is not None:
            best_box = best_region.box

    if signals.get("evidence_grounding", True) and clip is not None and grounding is not None:
        regions = grounding.propose_regions(image, claim.text, prompt)
        crops = [r.crop for r in regions]
        feats["evidence_grounding"] = clip.evidence_grounding(crops, claim.text)
    elif signals.get("evidence_grounding", True):
        feats["evidence_grounding"] = float("nan")

    # --- CED (Confidence - Evidence Delta) ---
    if signals.get("ced", True):
        conf = feats.get("vlm_confidence", float("nan"))
        evid = feats.get("evidence_grounding", float("nan"))
        if not (np.isnan(conf) or np.isnan(evid)):
            feats["ced"] = conf - evid
        else:
            feats["ced"] = float("nan")

    # --- UniProbe ---
    if signals.get("uniprobe", True) and uniprobe is not None:
        feats["uniprobe"] = uniprobe.score(
            image=image,
            claim_text=claim.text,
            generation=generation,
            tok_span=tok_span,
        )

    # --- Counterfactual sensitivity ---
    if signals.get("counterfactual", True) and counterfactual is not None:
        feats["counterfactual"] = counterfactual.sensitivity(
            image, claim.text, box=best_box, prompt=prompt
        )

    # --- CF-Detect ---
    if signals.get("cf_detect", True) and counterfactual is not None:
        feats["cf_detect"] = counterfactual.cf_detect_score(image, claim.text)

    # --- VIG (Visual Information Gain) ---
    if signals.get("vig", True):
        feats.update(_compute_vig(image, claim, cfg, models))

    # --- Prior term ---
    if signals.get("prior_term", True):
        feats["prior_term"] = _compute_prior_term(image, claim, cfg, models)

    # --- Self-consistency ---
    if signals.get("self_consistency", True):
        feats["self_consistency"] = _compute_self_consistency(
            image, claim, cfg, models
        )

    # --- Cross-detector disagreement ---
    if signals.get("cross_detector_disagreement", True):
        feats["cross_detector_disagreement"] = _compute_disagreement(feats)

    # --- Claim metadata features ---
    if signals.get("claim_type", True):
        feats["claim_type"] = claim.claim_type

    if signals.get("claim_length", True):
        feats["claim_length"] = len(claim.text.split())

    return feats


def _compute_vig(image, claim, cfg, models) -> dict:
    """
    Visual Information Gain: log P(claim | image) - log E[P(claim | null)].

    Uses logsumexp for the correct PMI estimator per the reframed plan.
    """
    from scipy.special import logsumexp

    vlm = models["vlm"]
    vig_cfg = cfg.get("vig", {})
    prompt = vig_cfg.get("scoring_template", "Describe what you see. {claim}")
    scoring_prompt = prompt.replace("{claim}", "")
    n_null = vig_cfg.get("n_null_samples", 4)
    use_lse = vig_cfg.get("use_logsumexp", True)

    try:
        # Score claim against real image
        base = vlm.score_continuation(image, scoring_prompt, claim.text)
        base_lp = base.mean_logprob
        if np.isnan(base_lp):
            return {"vig": float("nan"), "vig_per_token": float("nan")}

        # Score claim against null images
        null_lps = []
        null_mode = vig_cfg.get("null_construction", "random_images")
        null_pool_path = cfg.get("paths", {}).get("null_image_pool", "")

        for _ in range(n_null):
            null_img = _make_null_image(image, null_mode, null_pool_path)
            null_score = vlm.score_continuation(null_img, scoring_prompt, claim.text)
            if not np.isnan(null_score.mean_logprob):
                null_lps.append(null_score.mean_logprob)

        if not null_lps:
            return {"vig": float("nan"), "vig_per_token": float("nan")}

        # Correct estimator: logsumexp(log p_i) - log N
        if use_lse:
            log_e_null = float(logsumexp(null_lps) - np.log(len(null_lps)))
        else:
            log_e_null = float(np.mean(null_lps))

        vig = base_lp - log_e_null
        vig_per_token = vig / max(base.n_tokens, 1)

        return {"vig": float(vig), "vig_per_token": float(vig_per_token)}

    except Exception as exc:
        logger.warning("VIG computation failed: %s", exc)
        return {"vig": float("nan"), "vig_per_token": float("nan")}


def _make_null_image(image, mode: str, null_pool_path: str):
    """Create a null image for VIG computation."""
    from PIL import Image as PILImage

    w, h = image.size

    if mode == "grey":
        return PILImage.new("RGB", (w, h), (128, 128, 128))
    elif mode == "gaussian_noise":
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return PILImage.fromarray(arr)
    elif mode == "random_images":
        pool = Path(null_pool_path)
        if pool.exists() and pool.is_dir():
            imgs = [p for p in pool.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
            if imgs:
                chosen = random.choice(imgs)
                return PILImage.open(chosen).convert("RGB").resize((w, h))
        # Fall back to grey if no pool available
        logger.debug("null_image_pool not found; using grey null")
        return PILImage.new("RGB", (w, h), (128, 128, 128))
    elif mode == "token_removal":
        # Cannot match visual token count — see reframed plan §3
        return PILImage.new("RGB", (w, h), (128, 128, 128))
    else:
        return PILImage.new("RGB", (w, h), (128, 128, 128))


def _compute_prior_term(image, claim, cfg, models) -> float:
    """
    log E[P(claim | null image)] — the prior term.

    Separate from VIG so it enters the feature vector independently.
    M1 is characterized by HIGH prior, not merely low VIG.
    """
    from scipy.special import logsumexp

    vlm = models["vlm"]
    vig_cfg = cfg.get("vig", {})
    prompt = vig_cfg.get("scoring_template", "Describe what you see. {claim}")
    scoring_prompt = prompt.replace("{claim}", "")
    n_null = vig_cfg.get("n_null_samples", 4)
    null_mode = vig_cfg.get("null_construction", "random_images")
    null_pool_path = cfg.get("paths", {}).get("null_image_pool", "")

    try:
        null_lps = []
        for _ in range(n_null):
            null_img = _make_null_image(image, null_mode, null_pool_path)
            null_score = vlm.score_continuation(null_img, scoring_prompt, claim.text)
            if not np.isnan(null_score.mean_logprob):
                null_lps.append(null_score.mean_logprob)

        if not null_lps:
            return float("nan")

        return float(logsumexp(null_lps) - np.log(len(null_lps)))
    except Exception as exc:
        logger.warning("prior_term computation failed: %s", exc)
        return float("nan")


def _compute_self_consistency(image, claim, cfg, models) -> float:
    """
    Sampling-based self-consistency: generate N responses at temperature,
    check how many agree with the claim (via CLIP text similarity or NLI).
    """
    vlm = models["vlm"]
    clip = models["clip"]
    sc_cfg = cfg.get("self_consistency", {})
    n_samples = sc_cfg.get("n_samples", 5)
    temperature = sc_cfg.get("temperature", 0.7)
    agreement_method = sc_cfg.get("agreement", "embedding")
    prompt = cfg["prompt"]

    try:
        agreements = []
        for _ in range(n_samples):
            gen = vlm.generate(image, prompt, temperature=temperature)
            if agreement_method == "embedding" and clip is not None:
                sim = clip.text_similarity(claim.text, gen.text)
                agreements.append(sim)
            else:
                # Simple substring check fallback
                match = 1.0 if claim.text.lower() in gen.text.lower() else 0.0
                agreements.append(match)

        return float(np.mean(agreements)) if agreements else float("nan")

    except Exception as exc:
        logger.warning("self_consistency computation failed: %s", exc)
        return float("nan")


def _compute_disagreement(feats: dict) -> float:
    """
    Cross-detector disagreement: how much the detectors disagree.

    Measured as the range (max - min) of the grounding-oriented signals.
    High disagreement → uncertain diagnosis → mechanism ambiguous.
    """
    detector_signals = []
    for key in ("vlm_confidence", "evidence_grounding", "counterfactual", "uniprobe"):
        val = feats.get(key)
        if val is not None and isinstance(val, float) and not np.isnan(val):
            detector_signals.append(val)

    if len(detector_signals) < 2:
        return float("nan")

    return float(max(detector_signals) - min(detector_signals))


# =========================================================================
# Per-image driver
# =========================================================================


def process_image(
    image_path: Path, cfg: dict, models: dict, stats: Counter
) -> dict:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    prompt = cfg["prompt"]

    t0 = time.time()
    generation = models["vlm"].generate(image, prompt)
    logger.info("[%s] response (%0.1fs): %s", image_path.name, time.time() - t0, generation.text[:120])

    claims = models["extractor"].extract(generation.text, response_id=image_path.stem)
    logger.info("[%s] %d claims extracted", image_path.name, len(claims))
    if not claims:
        stats["images_with_no_claims"] += 1
        return {
            "image": str(image_path),
            "response": generation.text,
            "claims": [],
        }

    stats["claims_total"] += len(claims)

    claim_results = []
    for claim in claims:
        t1 = time.time()
        feats = extract_signals(image, claim, generation, cfg, models)
        elapsed = time.time() - t1

        # Count NaNs per signal
        for k, v in feats.items():
            if isinstance(v, float) and np.isnan(v):
                stats[f"nan_{k}"] += 1

        claim_results.append({
            **claim.to_dict(),
            "signals": feats,
            "extraction_time_s": round(elapsed, 2),
        })

        logger.info(
            "  claim %s (%s): conf=%.3f evid=%.3f vig=%s uniprobe=%s cf=%s [%.1fs]",
            claim.claim_id,
            claim.claim_type,
            feats.get("vlm_confidence", float("nan")),
            feats.get("evidence_grounding", float("nan")),
            f"{feats.get('vig', float('nan')):.3f}" if not np.isnan(feats.get("vig", float("nan"))) else "nan",
            f"{feats.get('uniprobe', float('nan')):.3f}" if not np.isnan(feats.get("uniprobe", float("nan"))) else "nan",
            f"{feats.get('counterfactual', float('nan')):.3f}" if not np.isnan(feats.get("counterfactual", float("nan"))) else "nan",
            elapsed,
        )

    return {
        "image": str(image_path),
        "response": generation.text,
        "claims": claim_results,
    }


# =========================================================================
# Run summary
# =========================================================================


def run_summary(stats: Counter, models: dict | None, n_ok: int, n_failed: int) -> dict:
    """Summary of the run — read this before interpreting scores."""
    total = stats.get("claims_total", 0)
    lines = [
        "=" * 74,
        "RUN SUMMARY",
        "=" * 74,
        f"images: {n_ok} processed, {n_failed} failed, "
        f"{stats.get('images_with_no_claims', 0)} yielded no claims",
        f"claims: {total} total",
    ]

    summary: dict = {
        "images_ok": n_ok,
        "images_failed": n_failed,
        "claims_total": total,
        "nan_rates": {},
    }

    signal_keys = [
        "vlm_confidence", "evidence_grounding", "ced", "clip_similarity",
        "uniprobe", "counterfactual", "cf_detect", "vig", "vig_per_token",
        "prior_term", "self_consistency", "cross_detector_disagreement",
    ]

    if total:
        lines.append("per-signal NaN rate:")
        for s in signal_keys:
            n = stats.get(f"nan_{s}", 0)
            rate = n / total
            summary["nan_rates"][s] = rate
            flag = "  <-- BROKEN?" if rate > 0.25 else ""
            lines.append(f"    {s:<32} {n:>5}/{total}  ({rate:5.1%}){flag}")

    # Calibration reports
    if models:
        cf = models.get("counterfactual")
        if cf is not None and hasattr(cf, "calibration_report"):
            lines.append(cf.calibration_report())

    lines.append("=" * 74)
    print("\n".join(lines))
    return summary


# =========================================================================
# Config
# =========================================================================

REQUIRED_KEYS = (
    "vlm", "clip", "signals", "aggregation",
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
    cfg.setdefault("seed", 0)
    return cfg


def seed_everything(seed: int) -> None:
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
    """Replace NaN with null for valid JSON."""
    if isinstance(obj, float):
        return None if obj != obj else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


# =========================================================================
# Main
# =========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description="Adaptive V-TRACE+ pipeline")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--image", help="single image path")
    ap.add_argument("--images", help="directory of images")
    ap.add_argument("--prompt", help="override the prompt in config")
    ap.add_argument("--limit", type=int, help="stop after N images")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="load models but don't process images"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.prompt:
        cfg["prompt"] = args.prompt

    seed_everything(int(cfg["seed"]))

    if args.dry_run:
        logger.info("--- DRY RUN: loading models only ---")
        models = load_models(cfg)
        logger.info("all models loaded successfully")
        print("\nLoaded models:")
        for k, v in models.items():
            status = "None" if v is None else type(v).__name__
            print(f"  {k}: {status}")
        return 0

    if not args.image and not args.images:
        ap.error("need --image, --images, or --dry-run")

    if args.image:
        paths = [Path(args.image)]
    else:
        paths = sorted(
            p for p in Path(args.images).iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        logger.error("no images found")
        return 1

    logger.info("processing %d image(s)", len(paths))
    models = load_models(cfg)

    stats: Counter = Counter()
    results = []
    n_failed = 0
    for path in paths:
        try:
            results.append(process_image(path, cfg, models, stats))
        except Exception:
            n_failed += 1
            logger.exception("failed on %s", path)

    summary = run_summary(stats, models, len(results), n_failed)

    # Write results
    out_path = (
        Path(args.out) if args.out
        else Path(cfg.get("paths", {}).get("results_dir", "results")) / "results.json"
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

    if not results:
        logger.error("every image failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
