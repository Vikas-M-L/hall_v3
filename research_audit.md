# V-TRACE+ research audit

Audit date: 2026-09-09. Baseline revision: `32ff2af`. This document supersedes
earlier research-readiness ratings. Results are exploratory unless explicitly
identified as an untouched test evaluation. See `results/audited/` for the
reproducible artifact analysis and `research/` for the new evaluation harness.

## A. What currently works

- `vtrace-plus/models/clip_wrapper.py`: CPU CLIP/SigLIP whole-image and grid
  compatibility, prompt ensembles, shared image encodings, pairwise scores.
- `models/gemini_wrapper.py`: image description, JSON claim decomposition,
  sampling, and binary cross-check API methods. API availability is external.
- `models/owl_wrapper.py`: open-vocabulary box proposals and count comparison.
- `fusion/vtrace_fusion.py`: deterministic v1/v2 risk and experimental v3
  selective verdicts; CLI exports; Streamlit visualization.
- `fusion/gbm_diagnoser.py`: a standalone sklearn
  `GradientBoostingClassifier`, trained on generated numeric features; it is
  deliberately not the live diagnosis model.
- Saved POPE-derived compatibility predictions, a small scenario gallery, and
  runnable arithmetic/model smoke tests exist. A saved joblib GBM exists locally.

These are implementation capabilities, not proof of real-world correctness.

## B. What does not work or is not established

| Location | Finding | Consequence |
|---|---|---|
| `app.py`, diagnosis before count/cross-check blocks | Diagnosis reads attributes before they are populated | Detector and Gemini cannot affect that run's diagnosis. Fixed during this audit by moving diagnosis after verification. |
| `app.py` banner and cost line | Risk-only labels coexist with v3 verdicts; timer omits API stages; counts only successful cross-checks | Contradictory labels and understated latency/cost. Research harness uses independent telemetry. |
| `models/claim_extractor.py` | Manual fallback splits sentences, not atomic propositions | A sentence with cats, collars, ball and sofa remains compound. |
| `fusion/negation.py` | Negation inversion can invert a whole compound claim because one clause is negative | Not a validated logical contradiction detector. |
| v3 fusion | Tie fallback uses absolute score; negative claims use absolute counter-match; thresholds are heuristics | Not calibrated abstention or a window-independent method. Several YAML knobs are not consumed. |
| v3 fusion and legacy negation | Adjusted risk no longer equals displayed component contribution sum | Explanations are not complete numerical attributions. |
| `models/owl_wrapper.py` | Counts thresholded proposals without instance-level NMS; plural prompts and missing boxes | Duplicate boxes and missed objects; zero detections does not prove absence. |
| `fusion/diagnose.py` | M0 inferred from a score; M1 default without measuring priors; M4 from region/global gap | Hypotheses about errors, not causal identification. |
| `app.py` self-consistency | Token-overlap of short claim against entire captions | Length-confounded lexical overlap, not semantic entropy or model confidence. |
| `adaptive-vtrace/signals/vig.py` | Mixes mean-token likelihoods before expectation; independently mixes token positions | Not the stated sequence-level PMI estimator; all-negative-infinity case is unstable. |
| `adaptive-vtrace/fusion/two_stage.py` | Detection trained before OOF operation; class means computed on all mechanism labels; probe not refit from raw features | Leakage and train/serve feature mismatch; OOF guarantee is false. |
| `adaptive-vtrace/repair/policy.py` | Best uniform and type policy selected using evaluation outcomes; failed CV silently substitutes abstention | Optimistic comparisons and mislabeled policies. |
| `adaptive-vtrace/repair/cascade.py` | Receives full-stack output but sometimes charges stage-1 cost | Synthetic cost assumption, not measured savings. |
| `finetune/finetune_qwen_lora.py` | Validation defaults to a subset of training; labels clone entire input; visual grid metadata omitted; generic collation/truncation | Not a validated multimodal, label-only SFT implementation. No successful adapter experiment found. |
| Autopsy/gallery | Gallery evidence reconstructed from aggregate risk; fallback crop is not an object localization annotation | Visual proof overstates evidence. |

## C. Existing contribution

Useful engineering integration: image/claim evidence inspection, selective
cross-checks, diagnosis hypotheses and action recommendations on a local CPU
with optional remote VLM inference. Offline numeric GBM experiments are feasible.

## D. Weak or already established novelty

Atomic-fact verification precedes this project (FaithScore); post-hoc
extract/verify/correct precedes it (Woodpecker); uncertainty and selective
prediction are established. Thresholds, adding M4, prompt negation, diagrams,
and assembling models are not independently strong scientific contributions.
The current evidence does **not** establish a novel causal diagnosis method.
The defensible hypothesis is whether predicted failure-specific routing improves
verified answer utility over *the same detector* with a fixed repair and over
a taxonomy-free outcome-trained router, at matched cost and coverage.

## E. Unsupported claims to retire

- "10/10", SOTA, guaranteed accuracy, provably knows what it does not know.
- "100% accurate" without decided count, coverage, image grouping, and uncertainty.
- "Calibrated" for hand-set cosine windows or thresholds tuned on inspected POPE.
- "Independent judges": shared training data and same Gemini generator/judge
  can cause correlated errors; independence was not measured.
- "All models frozen / no training anywhere": the GBM and synthetic XGBoost
  experiments do train models. Frozen *backbones* is the accurate statement.
- "Full pipeline completed / repair implemented": legacy UI recommends repairs;
  no saved executed repair-and-reverification study exists.
- "Free CLIP" and percentage of actual API savings inferred from score thresholds.
- Comparing So400M and CLIP score gaps under separately hand-tuned windows as
  an accuracy gain. This is an affine-scale confound.
- Existing paper's Woodpecker arXiv identifier is wrong; checked source is
  `2310.16045`, not `2310.06554` (an unrelated audio paper).

## F. Missing experiments

No untouched evaluation of generated answers with independently annotated
spans, natural M0–M4 labels, executed repairs, preservation, measured API usage,
cross-model transfer, cross-dataset transfer, or adjudicated failure collection.
No task-appropriate CHAIR/AMBER/HallusionBench/FaithScore runs are saved.
More small samples from already inspected COCO images are not a substitute.

## G. Missing baselines

Need original answers; prompt judge; self-correction; detect-then-fixed repair;
same-budget random/type/outcome router; post-hoc verifier pipeline; an authentic
released method (e.g. Woodpecker/VCD where feasible). Reimplementations must be
labeled adaptations and not claimed to reproduce an original method's results.
Compare selective baselines at matched coverage, not selective v3 against a
forced-answer-only baseline. Use all methods on the same images and claims.

## H. Reproducibility

Unpinned model revisions and dataset mirrors, incomplete dependencies, unlogged
API model fallback, no durable raw API responses/usage, two colliding Python
packages named `fusion`, no complete run manifests. Joblib is an artifact, not
proof of its training set. `paper/gen_figures.py` retrains a synthetic GBM and
compares different cohorts while captioning them as the same. Timings include
dataset/network/load time and variable numbers of methods, not isolated
inference latency. Tests validate code contracts, not publication claims.

## I. Evaluation leakage

Legacy windows and v3 tie rules were revised after examining evaluation cases.
These cases are development data now. POPE repeats images and positive claims
across variants. Row-level resampling understates uncertainty. Bootstrapping
must resample images, retaining every associated question/claim. Legacy
`_bca_auroc` is a percentile bootstrap, not BCa. Do not reuse these items as an
untouched final test or choose repair actions using their ground truth.

## J. Contamination

COCO/Visual Genome and public prompts can overlap backbone pretraining;
unknown proprietary pretraining prevents a zero-contamination claim. Group
originals, near duplicates, crops and generated perturbations before splitting.
Record source IDs, hashes, official splits, generator family and template IDs.
Use newly collected, consented photos and held-out generator/template families
for additional transfer tests. Never use Gemini's judgment as its own gold label.

## K. Overfitting

Synthetic GBM classes have nearly deterministic count patterns and separated
feature means. A random split measures recovery of the generator rules.
Feature importance is not learned causal discovery. Hyperparameters, score
windows and rejection thresholds need development/calibration-only selection.

## L. Compute constraints

CPU inference and an API-based closed loop are feasible. Full POPE on CPU is
slower, not inherently impossible. Qwen/LoRA training needs a compatible GPU
environment; no trained adapter or measured GPU experiment was found. Unknown
API prices, token counts and RAM peaks must remain N/A for legacy runs.

## M. Publication requirements and scope

The new `research/` harness establishes typed inputs separated from gold,
record/replay of real responses, image-group split checks, deterministic
baselines and actual repair verification gates. Its unit-test fixtures are
explicitly synthetic, never research evidence. A publication still requires:
independent annotation, a sealed test, authentic baseline runs, full telemetry,
at least three seeds where stochastic, image-cluster paired confidence
intervals, matched-coverage controls, and preservation-aware net repair utility.
See `FINAL_RESEARCH_ROADMAP.md` for executable gates and remaining blockers.
