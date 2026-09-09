# Research question and preregistration draft

**Status: DEVELOPMENT. Freeze a dated hash before accessing the new test.**

Can predicted error-type routing improve hallucination correction and answer
preservation over uniform self-correction and detection-only approaches at
matched verification cost and coverage?

## What can be claimed

Engineering contribution now: typed, replayable image/question/answer pipeline
with controlled repair selection, candidate verification and reject/abstain
transitions. Scientific hypothesis, not result: type-specific actions improve
preservation-aware utility at a fixed budget. No causal-identification claim.
VISOR, Woodpecker, FaithScore and selective prediction are mandatory prior art.

## Primary endpoint

On the **same cases**, report corrected incorrect answers minus originally
correct answers corrupted or removed, divided by case count. Also report both
hallucination reduction and unconditional answer correctness. Abstention can
remove a hallucination, but is not a corrected answer. Penalize removal of a
correct answer. Report completeness on required answer elements separately.

Primary comparison: full route/repair/verify minus detect-then-fixed repair
with the **same detector and final verifier**. Secondary: taxonomy-free outcome
router; original output; a published post-remedy method; uniform correction;
random/surface-type routing; no verifier; no diagnosis; no repair.
Use the same number of candidate attempts, token caps, and API model family.
Candidate repair policy is selected on training/development data only.

## Evaluation schedule

1. Data rights, source hashes, group/near-duplicate review and annotation audit.
2. Pilot on development images: determine feasible strata/sample sizes and
   power from paired discordance. Do not assert a powered sample size now.
3. Train an optional error-type classifier; fit probability calibration on a
   separate image-disjoint calibration set. Compare prompt-only and GBM first;
   LoRA/SFT/DPO are optional until their added value is plausible.
4. Freeze configurations, model revisions, prompts, routing and stopping rules.
5. Seal test inputs and gold independently. An evaluation custodian runs all
   arms once; downstream analysis cannot select thresholds or repairs from gold.
6. Repeat stochastic experiments with preregistered seeds (42, 43, 44), reporting
   mean/SD across seeds and paired image-cluster bootstrap intervals within runs.
   Deterministic duplicate reruns are not independent evidence.
7. Test on held-out generator families and dataset domains with no retuning.

## Annotation and failure audit

Fifty independently reviewed failures minimum (target, not completed work):
detector FP/FN, diagnosis error, routing error, failed repair, overcorrection,
verifier error and ambiguous evidence. Use random sampling within prespecified
strata, not cherry-picked screenshots. Save original/edited text, suspicious
spans, boxes, votes, adjudication and final decisions. The current queue is
unreviewed. Knowledge claims without external evidence are unresolved.

## Metrics and tests

Detection: AUROC, average precision (label as AP/AUPRC), F1/precision/recall,
accuracy, ECE, Brier, span overlap and type macro-F1. Reliability bins fixed
before evaluation; raw similarity is not a probability. For 90%/95% claimed
confidence, report eligible count, correctness and interval, not only a rate.
Selective prediction: risk–coverage/AURC and matched-coverage confidence-only
baseline; zero decided cases is undefined accuracy, not 100%.

Repair: success excluding abstention, final correctness, hallucination removal,
overcorrection, preserved required facts, abstention, rejected-candidate count.
Official CHAIR/POPE/AMBER/HallusionBench/FaithScore metrics apply only to their
proper tasks and official evaluators, never renamed proxy measures.

Cost: calls attempted (including failures/retries), tokens by stage, wall-clock
latency including loading separately from warm inference, peak process RAM,
GPU memory when used, provider price/date, actual cost. Record shared-cache
replays separately from per-policy logical work. Unknown price => cost N/A.

Statistics: paired image-cluster bootstrap, image-level randomization test,
effect size and denominators. Preregister primary test, correct multiple
secondary hypotheses (e.g. Holm), report negative findings. Same evidence on
which rules changed is development evidence; label it explicitly.

## Required ablations and feasibility

`research.ablations` emits configurations for no repair/router/diagnosis/verifier,
random routing, surface-type routing, uniform correction and full loop.
They are API prompt adaptations, not implementations of named external papers.
No OWL/SigLIP ablations can be claimed for the new API-only smoke backend;
add a genuine frozen-evidence adapter and record aligned features first.
Learned router, ranks, training-size sweeps and cross-VLM datasets are registered
as missing, not populated with synthetic paper numbers.

## Stronger formulation worth testing

Treat each proposed action as a prediction of **repair outcome**, not an
explanation of latent cause. Estimate expected corrected facts, expected damage
to supported facts, and measured cost. Compare that direct policy to taxonomy
routing on identical inputs. If the direct policy ties/wins, conclude the
taxonomy improves interpretability only, not intervention selection.
