# V-TRACE+

## Current Checker workflow

The Checker now uses `studio_service.py` and stores each completed analysis in
session state. Changing the selected fact or overlay does not repeat inference.
Changed inputs/settings are marked as stale until **Run verification** is clicked.

- Local extraction separates supported count/activity and clothing/color
  patterns. Example: `5 cats are eating` becomes presence, count and activity.
  Unsupported grammar and hedges remain explicit full-claim-check candidates;
  local rules do not claim complete natural-language atomic decomposition.
- Optional Gemini decomposition handles more complex text, but its output is
  still a set of model-generated candidate facts, not gold annotations.
- Fact cross-checks prioritize count, activity, attribute, spatial and OCR
  claims. Unchecked specialized facts stay unresolved; an explicit budget limits
  calls. **Verify all facts** is an optional, more expensive operation.
- **Original image** is the default visualization. **Detector candidates**
  displays stored OWL-ViT boxes after class-wise NMS and coordinate clipping.
  **Similarity crop** is opt-in and explicitly not an object bounding box.
- **Propose correction and verify** generates a candidate on a separate click,
  verifies all extracted facts, and checks support and relevance against the
  original question. A failure rejects the candidate; the original is retained.
  Model approval does not replace independent human evaluation.
- Strict JSON exports include facts, extraction status, recorded boxes,
  optional evidence, final decisions, timings and correction checks. Nonfinite
  scores are exported as `null` rather than invalid JSON `NaN`.

There is no new dedicated OCR model or validated semantic action detector.
Those questions use the optional focused VLM check or remain unresolved.
Calibration tooling and independent evaluation protocols are under `../research/`;
natural annotations and publication-level accuracy experiments are still pending.

> **Research audit (2026-09-09):** The historical descriptions below are retained
> as implementation notes, not validated research claims. Read
> [`../research_audit.md`](../research_audit.md),
> [`../claim_audit.md`](../claim_audit.md) and the root README first.
> Score windows are not probability calibration; the standalone GBM is trained
> on synthetic data and remains offline; the LoRA launcher is quarantined.
> Matched-coverage reanalysis finds no v3 advantage over confidence selection.
> The audited research entry points are under `../research/`.

Claim-level hallucination risk scoring for vision-language models. Every model is
frozen and used for inference only. There is no training loop, no labelled dataset
requirement, and no train/val/test split anywhere in this repo. The only thing
built from scratch is the fusion logic, and it is deterministic rules — not a
fitted model.

```
image ──> Qwen2.5-VL ──> response ──> claim decomposition ──> per claim:
                                                              confidence
                                                              CLIP evidence
                                                              CLIP similarity
                                                              UniProbe    (stub)
                                                              counterfactual
                                                                  │
                                                          rule-based fusion
                                                                  │
                                                       risk in [0,1] + explanation
```

Text-only shortcut (no local VLM, CPU friendly):

```
image + your text ──> sentence claims ──> CLIP / SigLIP ──> fusion ──> JSON
```

Real-VLM shortcut (no GPU, needs `GEMINI_API_KEY`):

```
image ──> Gemini describe + decompose ──> SigLIP verify ──> fusion ──> JSON
```

## Status

* CPU paths (`--dry-run`, `--text`, Streamlit app, all tests) run and verified.
* Gemini VLM path (real descriptions + atomic claims + self-consistency samples)
  run and verified end-to-end (~40s per image, all-correct on controls).
* Full local-VLM path (`Qwen2.5-VL-7B` generation + confidence + local
  counterfactual) is written but has never run on a GPU — expect first-session
  setup work there.
* `tests/` is fully green: 40 here (see below). Run suites per repo — both
  repos contain a `fusion/` package, so a single joint `pytest` run collides
  on the import name:
  `python -m pytest vtrace-plus/tests -q` and separately for `adaptive-vtrace`.

## Before you run anything

Two model identities could not be verified and are deliberately left unwired:

**`Counterfactual-Hallucination-Detect`** — no Hugging Face repo by this exact
name could be confirmed. Confirm the real model ID before setting
`counterfactual.mode: model`. You probably do not need to: `mode: local` computes
the same signal with the VLM you already have loaded, and is the shipped default.

**UniProbe** — a very recent NVIDIA Research release whose call signature was not
confirmed. `models/uniprobe_wrapper.py` is a scaffold with a header listing the
five things to check on the repo/paper page before filling it in. Do not guess the
API: a plausible-looking wrong implementation imports cleanly, runs without error,
and returns meaningless numbers, which is strictly worse than a stub that
announces itself.

Item 4 in that list — output orientation — is the one that silently inverts your
results. See the orientation table below.

## Install

```bash
pip install -r requirements.txt
```

Qwen2.5-VL needs a recent `transformers`. On Colab-class memory, keep
`vlm.load_in_4bit: true`. For the UI: `pip install streamlit pandas`
(`streamlit>=1.30` verified with 1.56). For the Gemini path:
`pip install google-genai python-dotenv` (plain-HTTPS implementation, no other
dependency), then put the key in `vtrace-plus/.env`:

```
GEMINI_API_KEY=AIza...
```

`.env` is gitignored — verify before committing. Without it the Gemini backend
fails fast with a clear error instead of a cryptic one.

## Run

```bash
# fast logic tests — no GPU, no downloads, no model weights
python tests/test_fusion.py
python tests/test_negation.py
python tests/test_gemini.py        # mocked HTTP, no API spend
python tests/test_ordering.py      # needs CLIP weights (CPU ok)

# see what the rules do on hand-written signal values
python scripts/run_pipeline.py --dry-run

# score a real image, full local-VLM path (needs GPU)
python scripts/run_pipeline.py --image path/to/img.jpg

# text-only CPU path: you supply the claims, CLIP/SigLIP verifies
python scripts/run_pipeline.py --image img.jpg --text "A red square. A blue elephant." \
    --device cpu --clip-model google/siglip-base-patch16-224 --out results/run1.json

# a directory, with results written to JSON
python scripts/run_pipeline.py --images path/to/dir --out results/run1.json

# override the calibration window (required for SigLIP: --cos-window -0.10 0.12)
python scripts/run_pipeline.py --image img.jpg --text "..." --cos-window 0.15 0.35

# the v1/v2 comparison
python scripts/run_pipeline.py --image img.jpg --fusion-mode v1
python scripts/run_pipeline.py --image img.jpg --fusion-mode v2
```

## Streamlit Studio (recommended on CPU)

```bash
streamlit run app.py   # from vtrace-plus/
```

Two backends in the sidebar: **Verifier only** (type claims, CPU seconds) and
**Gemini VLM + verify** (real description → atomic claims → verified, ~40s–2min
with optional 3x self-consistency sampling). Output: verdict banner with risk
gauge, ranked claim cards with per-signal split bars, grounding-box overlay,
table, JSON/CSV export. Defaults: SigLIP verifier, prompt ensembling on, grid 3.

If the UI ever throws an import/unpack error right after a code change, the
server is running stale code (auto-reload missed it) — restart it, don't debug
the app: kill the `streamlit run app.py` process and start it again.

## Signal orientation

This is the highest-consequence detail in the repo. Signals arrive in different
directions and are all converted to risk orientation (higher = more likely
hallucinated) inside `fusion/vtrace_fusion.py` before any weighting happens:

| signal | native meaning of "high" | conversion |
|---|---|---|
| `confidence` | model is certain | `1 - x` |
| `evidence` | claim matches some image region | `1 - x` |
| `clip_similarity` | claim matches the whole image | `1 - x` |
| `counterfactual` | claim depended on the masked region | `1 - x` |
| `uniprobe` | hallucinated | used as-is |
| `self-consistency uncertainty` | samples disagree | used as-is (display + JSON only) |

If you swap in a model with the opposite orientation, fix it in that model's
wrapper so this table stays true. Do not special-case it in the fusion layer.

**Negation** (`fusion/negation.py`): CLIP matches "no dog" to dogs, so negated
claims are post-corrected — risk becomes the *match* instead of `1 - match`,
flagged as `negation-inverted`. Without this, false denials read as grounded.

## Checkpoints: pick by band, not by size

Raw cosine bands differ per family — a window tuned for one saturates another.
Presets (also the UI defaults, applied automatically per model):

| checkpoint | band / window | note |
|---|---|---|
| `google/siglip-base-patch16-224` | `[-0.10, 0.12]` | default; widest true/false margin (~0.79 measured) |
| `openai/clip-vit-base-patch32` | `[0.15, 0.35]` | fastest; margin ~0.28 |
| `openai/clip-vit-large-patch14` | `[0.10, 0.40]` | best CLIP; ~3x slower on CPU |

Every run prints `SUGGESTED WINDOW` from observed cosines — set it and re-run
rather than trusting absolutes. Prompt ensembling (3 caption templates, text
side only) is on by default and costs ~2 extra text encodes per claim.

## Grounding boxes (ROI): global vs local

`clip.evidence()` returns `(score, box, whole_won)`. When the whole image is the
best match, the box is a fallback crop for the counterfactual masker — display
consumers must NOT draw it: `whole_won=True` means "matched globally, no local
box." The app already implements this; any new consumer must too, or close-up
photos get misleading rectangles. Covered by
`test_whole_won_flag_distinguishes_global_from_local`.

## Fusion modes

`fusion_mode: v1` — fixed equal weights across all active signals. This is the
baseline the v2 rules have to beat.

`fusion_mode: v2` — rule-based adaptive weighting. Two rules, both deterministic
functions of the signal values:

*Rule 1, confident but ungrounded.* When `CED = confidence − evidence` exceeds
`CED_THRESHOLD`, the model is sure about something the image does not support.
Weight shifts onto `evidence` and `counterfactual`, and off raw `confidence`.

*Rule 2, detectors disagree.* When the standard deviation across the risk-oriented
`evidence`, `uniprobe`, and `counterfactual` values exceeds
`DISAGREEMENT_THRESHOLD`, those three are contradicting each other. They get
downweighted and `clip_similarity` gets promoted, since CLIP is independent of
both the VLM and the probes.

Rules are independent and can both fire; multipliers compose and the weight vector
is renormalized. If neither fires, weights stay at `BASE_WEIGHTS`, which is
near-equal. Every threshold and multiplier is a named constant at the top of
`fusion/vtrace_fusion.py`. (In text-only CPU mode the VLM signals are off, so
v2 ≈ v1 — the rules need confidence to fire.)

### Known limitation: v2 is not monotone

Hard thresholds make risk discontinuous in the signal values, so improving a
signal can briefly *raise* the score as a rule switches on or off. The largest
observed jump with the shipped constants is about 0.044, at the Rule 2 boundary
where Rule 1 is also active. The mechanism is that Rule 2 downweights `evidence`
exactly when evidence has become the outlier among the detectors — which is also
what happens when evidence is strongly supportive and the others are neutral.

`tests/test_fusion.py::test_v2_monotonicity_violation_is_bounded` measures this
rather than papering over it. If you want it gone, replace the hard `if x >
threshold` gates with a graded multiplier (for example scaling the adjustment by
`clip((x − threshold) / width, 0, 1)`); the constants are already factored so this
is a local change. It was left as hard gates because that is what makes the v1/v2
comparison legible: you can point at which rule fired on which claim.

## Stubbed signals

A wrapper in `placeholder` mode returns the same constant for every claim. It
carries zero information and only drags every score toward that constant, which
will flatten your score distribution and quietly ruin any ranking metric. With
`drop_stubbed: true` (the default) such signals are excluded and the remaining
weights renormalize. Leave it on until the real models are wired in.

## Aggregation

`aggregation.method` turns per-claim risks into one image-level number.

`max` — one bad claim makes the response risky. The usual default for "should a
human look at this?"
`topk_mean` — mean of the k riskiest claims. Less jumpy than `max` on long
responses, where `max` saturates on a single outlier.
`mean` — dilutes a single severe hallucination in a long response. Report it only
alongside one of the others.

## Layout

```
app.py                            V-TRACE+ Studio (verifier-only + Gemini backends)
configs/default.yaml              all knobs; inference-only, no training keys
fusion/vtrace_fusion.py           the deterministic fusion — the core of the repo
fusion/negation.py                negated-claim correction (match = risk)
models/vlm_wrapper.py             Qwen2.5-VL: per-token logprobs, span alignment,
                                  teacher-forced scoring (needs GPU)
models/gemini_wrapper.py          real-VLM API backend: describe / decompose / sample,
                                  retry + model fallback, key from .env
models/claim_extractor.py         response -> atomic claims with char spans
                                  (falls back to sentence split with no VLM)
models/clip_wrapper.py            evidence (claim vs regions + whole_won flag),
                                  similarity; prompt ensembling, per-checkpoint windows
models/uniprobe_wrapper.py        SCAFFOLD — do not fill in by guessing
models/counterfactual_wrapper.py  SCAFFOLD for the external model; working local mode
scripts/run_pipeline.py           end-to-end driver: --dry-run, --text, --cos-window
tests/test_fusion.py              fusion logic, no GPU (27 tests)
tests/test_ordering.py            true-beats-false + ROI-flag guard (needs CLIP, CPU ok)
tests/test_negation.py            negation detection + inversion (no models)
tests/test_gemini.py              backend parsing/fallback, mocked HTTP (no API spend)
```

## Cost

The expensive parts per claim are the CLIP region encodes and, in `local`
counterfactual mode, two extra VLM forward passes. Region proposals are computed
once per image rather than once per claim, which with a 3×3 grid saves ten CLIP
image encodes per claim. Text-side prompt ensembling adds 2 cheap text encodes
per claim. The VLM prefill is the remaining hot spot: each
`score_continuation` call re-prefills the full visual context (roughly 1,300
visual tokens at the configured `max_pixels`). If throughput becomes the
bottleneck, cache the prefix KV across claims for a given image — that is the
single highest-leverage optimization available and it is not implemented here.
The honest version of the cost story is the cascade in `adaptive-vtrace/`
(see below): free signals decide easy claims, VLM passes fire only on ambiguity.

## How to read results (earned the hard way)

* Read the **ranking**, not the absolutes. The top claim is the one to check.
* Calibrate the window from `SUGGESTED WINDOW` before comparing across images.
* Negated claims ("no X") are corrected, not magic — ambiguous hedges
  ("maybe", "appears") are not negation and are not corrected.
* Counting, OCR, and tiny objects are beyond grid-CLIP. They need the GPU path
  (VLM confidence + zoom re-query), not a bigger window.
* A box drawn on a close-up photo used to be a fallback crop, not the match —
  fixed via `whole_won`; global matches now show the full frame.

## What this repo does not do

No training, no fine-tuning, no learned fusion weights, no dataset loaders, no
splits, no evaluation harness, no metrics against a labelled benchmark. Adding
benchmark evaluation means adding labels, at which point the "no labelled dataset"
constraint no longer holds — that is a deliberate boundary, not an oversight.
(The `adaptive-vtrace/` sister project holds the research track, all
CPU-tested: VIG estimator, self-consistency scorer, two-stage mechanism
classifier, eval metrics incl. AUROC/AUPRC/paired bootstrap, POPE loader,
`eval/pope_bench.py` with real COCO/POPE numbers, repair-routing §7 table, and
the cost-aware cascade with its accuracy-vs-cost frontier.)
