# V-TRACE+ diagnosis-and-repair evaluation protocol

Status: **protocol and schema, not a completed new benchmark**. No claim of
balanced human annotations, annotation agreement or fifty reviewed failures
is made. `results/audited/failure_review_queue.jsonl` is a review queue, not
an annotated dataset.

## Existing data

Legacy POPE-compatible files under `adaptive-vtrace/results/` contain
question-derived object phrases, absence labels and model scores. They do
not contain original VLM answers, repaired answers or mechanism gold labels.
Pooled cohort: generated inventory and label counts are in
`results/audited/report.json`. Treat every previously inspected image as
development-only, even if its original benchmark split is called test.
The gallery mixes hand-authored controls and COCO images; its annotations were
not independently adjudicated and detector counts are not ground truth.
The standalone GBM trains on numeric synthetic clusters, not real images.

## Sources and licensing

| Source | Intended use | License/redistribution status |
|---|---|---|
| Official POPE + COCO | Object QA and original-model response collection | Record POPE release license; COCO annotations and individual image rights are distinct. Per-image attribution/license must be retained. Current mirror license not audited. |
| AMBER | Attributes/relations + external validation | Verify official release terms before collecting/distributing |
| HallusionBench | Spatial/reasoning stress tests | Verify official image/source rights; preserve control groups |
| M-HalDetect | Natural response spans | Verify release terms, source images and annotation license |
| MMHal-Bench | Free-form repair evaluation | Use release instructions and independent answer adjudication |
| MME/TextVQA-style OCR tasks | Capability and OCR preservation | Verify task-specific rights and official scoring rules |
| Newly collected photos | Contamination-resistant external test | Obtain consent/rights; exclude private identifiers; record provenance |

Do not relicense third-party images with repository code. Store IDs and acquisition
instructions rather than assuming every cached image can be republished.

## Unit and taxonomy

Unit: image + original question + one actual VLM response. Annotation includes
all factual claims, exact suspicious spans and a corrected answer. For the first
repair headline, use single-claim QA so response rewriting does not confound
claim alignment. Multi-claim captions are a separate diagnostic/repair track.

Use **error content** labels: object, attribute, counting, spatial, relation,
OCR, reasoning, knowledge, unsupported inference, grounding, unknown. Keep
mechanism hypotheses (prior dependence/perception/uncertainty/binding) separate:
human inspection cannot establish a latent causal mechanism from text alone.
M0–M4 are legacy coarse categories, not a validated causal ontology.

## Files and schema

`inputs.jsonl` (model-visible; accepted by `research.core.Case`):
`id, group_id, image, image_sha256, question, response, split`.

`gold.jsonl` (evaluator only):
`id, correct_answer, original_hallucinated, original_correct, hallucination_type,
hallucinated_spans, visual_evidence, severity, corrected_answer, source_dataset,
source_id, source_license, generator_model, generator_revision, prompt_id,
annotation_status, annotator_ids, agreement, adjudicator`.

`visual_evidence`: boxes with coordinate convention, source annotation ID and
whether human-confirmed. Missing evidence is null, not a made-up box.
`severity`: prespecified ordinal scale (minor detail / material answer error /
unanswerable); score usefulness separately from factuality.
`outcomes.jsonl`: case ID, strategy, actual candidate/final text, independent
original/final correctness and hallucination labels, abstention, label source.
`research.score_repairs` refuses model-only gold and unknown final labels.

## Collection and annotation

1. Freeze source versions/checksums and input manifests before generation.
2. Generate answers from at least two model families with exact prompts and
   temperature/seed metadata; record API model version, raw answer and usage.
3. Collect natural failures first. Add controlled negative edits only in a
   separately marked diagnostic set; balance length, wording and templates so
   the class cannot be inferred from linguistic artifacts.
4. Two annotators independently judge factuality, answer relevance, error type,
   spans and image evidence. A third adjudicates disagreements. Annotators are
   blind to method and selected repair. Store original votes.
5. Report agreement (e.g. Cohen kappa for class labels; span overlap separately),
   confusion and adjudication rate. **Current agreement: N/A.**
6. Select at least fifty failure cases by a frozen stratified sampling rule;
   review originals, repairs and rejected repairs. Report all categories,
   including false positives and verifier failures. **Completed manual review: N/A.**

## Isolation and balance

Preserve official partitions where provided. Group all questions, captions,
crops, perturbations, near-duplicates and derived negatives from one source
image/scene. Hash image bytes and perform perceptual-duplicate review. Split
new groups by a frozen salted hash (60/20/20 is a protocol setting, not achieved
counts), with generator/template families reserved for transfer. Official
test data must never train a calibrator or router. Previously inspected legacy
groups are excluded from the new test via a development exclusion manifest.

Prespecify minimum per-type targets and balanced *evaluation strata* after a
power calculation. Do not selectively remove hard examples to balance classes;
report prevalence-weighted and macro scores. Rare knowledge/OCR classes may
require separate sourcing. Current real M0–M4 labeled sample size: N/A.

## Limits and contamination

Unknown backbone pretraining, COCO reuse, correlated API judges, synthetic
negative artifacts, ontology-limited gold, occlusion/ambiguity and annotator
disagreement remain. Report unknowns; abstain where the image cannot answer.
The protocol does not imply a completed benchmark or a clean new test until
the manifests, image rights, independent labels and hashes are finalized.
