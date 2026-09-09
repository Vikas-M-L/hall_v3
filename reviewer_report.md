# Three reviewer perspectives

These are three **simulated review perspectives by the same assistant**, not
independent external peer reviews. Scores are qualitative assessments of the
audited research evidence, not acceptance predictions.

## Reviewer 1 — ML research

**Strengths:** clear falsifiable intervention-selection question; paired
evaluation design; negative matched-coverage result retained; explicit
answer-preservation objective.

**Weaknesses:** current gains are encoder compatibility results, not closed-loop
repair results. Broad novelty is preceded by VISOR/Woodpecker. Small development
cohorts and deterministic synthetic labels cannot establish diagnosis learning.

| Novelty | Technical quality | Experiments | Reproducibility | Clarity | Overall |
|---:|---:|---:|---:|---:|---:|
| 3/10 | 4/10 | 2/10 | 6/10 | 7/10 | 3/10 |

**Reject reasons:** central utility/cost claim untested; no clean superiority
against same-detector fixed/outcome policies; prior test reuse.
**Required:** sealed same-case comparison with independently adjudicated answer
preservation; effect sizes/CIs; held-out generator family; direct outcome-router
control; authentic post-remedy baseline. Publish a negative result if routing ties.

## Reviewer 2 — computer vision / VLM

**Strengths:** claim-level visual evidence inspection; CPU-compatible encoders;
recognition that counting, relations and external knowledge differ.

**Weaknesses:** grid maxima are not localization truth; OWL proposals are not
instance-count gold; atomic extraction and negation are brittle; no reference
annotations establish M1 causal prior override or M4 binding causes. Real API
smoke checks are not manually verified correctness results.

| Novelty | Technical quality | Experiments | Reproducibility | Clarity | Overall |
|---:|---:|---:|---:|---:|---:|
| 3/10 | 4/10 | 3/10 | 6/10 | 7/10 | 3/10 |

**Reject reasons:** no multi-type real benchmark and no repair preservation
study; unsupported terminology of "proof"/causes; weak external baselines.
**Required:** natural object/attribute/count/relation/OCR failures with human
spans/boxes, official AMBER/HallusionBench/CHAIR evaluation, NMS/recall analysis,
cross-model judge error audit and authentic VCD/Woodpecker comparison.

## Reviewer 3 — skeptical reproducibility

**Strengths:** new input/gold boundary, replay keys and source/data hashes;
failure-on-missing records; missing metrics remain null; legacy trainer blocked.

**Weaknesses:** legacy dependency/model revisions were not pinned; historical
API claims lack recordings; reused development data; known legacy OOF/VIG
defects; no externally curated test set or fifty completed failure annotations.
New harness tests and smoke traces do not repair old scientific claims.

| Novelty | Technical quality | Experiments | Reproducibility | Clarity | Overall |
|---:|---:|---:|---:|---:|---:|
| 3/10 | 5/10 | 2/10 | 6/10 | 8/10 | 3/10 |

**Reject reasons:** unable to independently reproduce a complete claimed
improvement over strong baselines; old paper overclaims calibration and costs.
**Required:** fresh environment CI, pinned hashes/versions, dataset rights,
same-cohort official metrics, complete API usage and failure accounting,
annotator logs, immutable configuration/test manifests and single-command runs.
