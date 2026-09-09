# Final research roadmap

The current project is **not yet publication-validated**. The broad novelty
claim overlaps established work; see the VISOR entry in `literature_review.md`.
The audited negative result (v3 ties confidence-only selection at matched
coverage) must remain visible. No acceptance probability is scientifically
estimable from this prototype.

## P0 — necessary before submission

1. **Freeze the research question and endpoints.** Use `research/protocol.md`;
   register answer preservation alongside hallucination reduction.
2. **Natural data + independent annotation.** Collect real image/question/VLM
   answers with recorded model revisions, label spans/types independently,
   adjudicate repairs. Fifty failure reviews are pending, not complete.
3. **Isolated test.** Exclude all previously inspected legacy image groups;
   group near-duplicates, preserve official splits, commit manifest hashes.
4. **Authentic baselines.** Same inputs and cost budget: original VLM, uniform
   self-correction, detect-then-fixed, confidence-only abstention, random/type
   routing, taxonomy-free outcome policy and at least one released mitigation.
5. **Run the complete loop.** `research.collect` → `research.run` → independent
   labels → `research.score_repairs`. Smoke recordings verify integration only.
6. **Calibration.** Fit using calibration-only data; evaluate on untouched test.
   Do not rescale score windows per evaluated image or tune after seeing outcomes.
7. **Fix remaining legacy research code before using it.** OOF probe, sequence
   VIG, policy selection leakage and unmasked LoRA trainer remain excluded from
   scientific evidence. LoRA launcher is quarantined to prevent invalid runs.
8. **Statistics + reproducibility.** Same-case comparisons, image-cluster CIs,
   paired randomization, fixed seeds, exact scripts/prompts and usage logs.

Exit criterion: all primary arms executed on sealed data; preserved-answer
utility, costs and confidence intervals reported, including null/negative results.

### Preservation implementation update

The ledger/alignment/candidate-fact acceptance gate is implemented in
`research/preservation.py` and connected to the Studio correction flow.
Fixed-route, predicted-route and audit-compute-matched gate-disabled strategies
exist. A mixed controlled API smoke trace protects four model-supported facts;
that is a functional observation, not independent accuracy evidence. The next
blocking item remains **natural independent annotation and sealed matched-arm
evaluation**, including a direct outcome-router and authentic prior-method baseline.

## P1 — strongly recommended

- Multi-generator and cross-dataset transfer; hold out prompt/template families.
- Blinded error-type and span annotation agreement; human verifier-error audit.
- Local detector NMS, recall audit and robust compound/negated-claim handling.
- Faithfulness of explanations: ablate each evidence channel and measure changes
  against human evidence masks and intervention outcomes (not rule self-tests).
- Larger representative sample after a pilot power analysis; full benchmark
  execution may run on CPU slowly or on GPU, neither guarantees clean labels.
- Container/CI with pinned model revisions, reproducible dependency lock and a
  recorded dataset license/attribution manifest.
- Run official CHAIR/AMBER/HallusionBench/FaithScore evaluators where applicable.

## P2 — optional until P0 results justify them

- LoRA/SFT/DPO: repair multimodal trainer first; compare parameter/time budgets.
- Outcome-trained routing, budget-sensitive action policies and a second judge
  family. Do not keep a learned component that does not improve held-out utility.
- Broader counterfactual interventions and certified selective risk bounds only
  after their assumptions and calibration sample sizes are satisfied.
- Public dataset release, hosted service and polished UI after rights/privacy,
  rate limiting and external validation are complete.

## Deliverables produced in this revision

Research/claim audits; primary-source literature map; dataset protocol;
typed label-free closed-loop runner with live and replay modes; ablation config
generator; calibration and CPU training commands; independent repair scorer;
paired image-cluster analysis; matched-cohort tables/figures; review queue;
actual API smoke recordings; synthetic generalization stress test; revised
paper and reproducibility guide. Missing experiments remain explicitly N/A.
