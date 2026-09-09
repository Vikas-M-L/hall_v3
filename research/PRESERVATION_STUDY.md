# Preservation-aware repair study

## Implemented method, not a proven novelty claim

The method now keeps an original-fact ledger (source spans, predicted evidence
state and stable IDs), verifies a candidate's facts, and independently asks for
text entailment alignment. Only original facts judged supported are protected.
All supported facts must be retained; unknowns stay unknown. Refuted facts that
remain in the candidate cause rejection. New candidate facts require positive
visual verification, and extraction completeness is checked. Question relevance
must pass separately. Missing/duplicate IDs, invented quotes, incomplete audits,
or ambiguous alignments cannot result in acceptance.

These checks are fallible model judgments, not gold. Protecting a false-positive
original fact can block a correct repair. Natural evaluation must measure this
failure mode and verifier correlations. The existing literature overlap remains.

## Controlled comparisons

`configs/preservation.json` runs:

- `original`: untouched answer.
- `detect_then_fixed`: original detection/fixed repair/whole-answer verifier.
- `preservation_fixed`: fixed visual recheck plus fact/preservation gate.
- `preservation_full`: predicted error-type repair plus the same gate.
- `preservation_gate_disabled`: identical predicted router and extra audit calls,
  but preservation alignment is not used for acceptance (compute-matched gate
  ablation). It still requires visual candidate support and question relevance.

The primary routing comparison is **preservation_full vs preservation_fixed**.
The primary gate comparison is **preservation_full vs preservation_gate_disabled**.
Report both quality and costs, not the gate's own approval rate as accuracy.
An outcome-trained router and authentic prior-method baselines remain required.

## Reproduction commands (repository root)

```
python -m pytest research/tests/test_preservation.py research/tests/test_preservation_evaluation.py -q
python -m research.run --config configs/preservation.json --inputs experiments/controlled_edit_inputs.jsonl --recording experiments/preservation_smoke.jsonl --out results/audited/preservation_smoke.json --live
python -m research.run --config configs/preservation.json --inputs experiments/controlled_edit_inputs.jsonl --recording experiments/preservation_smoke.jsonl --out results/audited/preservation_replay.json
python -m research.evaluate_preservation --adjudicated experiments/preservation_gold.jsonl --baseline preservation_fixed --proposed preservation_full --out results/preservation_evaluation.json
```

Live mode requires an environment `GEMINI_API_KEY`. The controlled edit is an
author-edited development example, not a natural hallucination benchmark.
`preservation_gold.jsonl` does **not** exist yet. Do not generate fake labels
to run the last command. Each adjudicated row requires the repair scorer fields
plus `group_id`, `supported_facts_before`, `supported_facts_retained`, and
`new_unsupported_facts`. Human labels must be blind to method and independent
of the models that generated/verifed candidates. The scorer enforces matched
case IDs and original labels and uses paired image-cluster intervals.

## Publication gates

1. Independently annotate real multi-fact responses and original-supported facts.
2. Isolate image families and generator/template families; freeze development
   choices and calibration thresholds before the test.
3. Run every baseline on identical cases with bounded candidates, preserving
   failures/rejections in the denominator and usage in cost totals.
4. Label candidate/final correctness, omissions, new hallucinations and required
   answer content. Compare utility, preservation and cost with intervals.
5. Keep negative results; match coverage and costs; check authentic prior work.

Current correctness/quality gain on a held-out natural dataset: **N/A**.
Publication submission is not complete until these gates are met.
