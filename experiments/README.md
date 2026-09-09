# Experiment provenance

- `live_inputs.jsonl`: actual answer obtained with `research.collect` from
  Gemini on an existing development image, not a new independent benchmark.
- `controlled_edit_inputs.jsonl`: explicitly **author-edited** derivative of
  that answer, replacing cat mentions by dog mentions. It is a functional
  repair smoke test, not a naturally sampled model hallucination.
- `live_smoke.jsonl` / `controlled_edit_smoke.jsonl`: real API request/response
  recordings generated during this audit. No reference answer is sent.
- `fixture/`: synthetic canned outputs for replay testing, never model results.
- `ablation_configs/`: generated protocol arms; no empirical ablation result
  is implied by the existence of a config.

Independent human correctness/repair annotations remain required. API judge
approval is not gold. Recorded IDs, hashes, prompts and returned versions are
retained for replay. Do not promote these development images into sealed test.

## Preservation smoke records

- `preservation_smoke.jsonl`: new audit stage on the prior author-edited dog
  example. The model marked no original facts as protected; this case cannot
  demonstrate protection of originally correct content.
- `preservation_mixed_inputs.jsonl`: author-created mix of cat/location claims
  and an invented elephant, on the same exposed development image.
- `preservation_mixed_smoke.jsonl`: actual recorded responses. The first run
  (`results/audited/preservation_mixed_smoke.json`) includes an HTTP repair
  failure. Recovery and offline replay are separate outputs, preserving failure
  history. Both fixed and predicted-routing arms are present; no superiority
  claim follows from this one case.

Independent gold: absent. The four protected facts in the mixed run are
model-observed support, not new manually verified labels.
