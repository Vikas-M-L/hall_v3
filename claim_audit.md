# Claim audit (supersedes earlier README/paper and chat claims)

Evidence levels: **artifact** = recomputed from saved outputs; **executed smoke**
= request/response path exercised, not independently scored; **synthetic** =
constructed assumptions; **proposal** = missing experiment.

| Claim | Evidence / experiment | Dataset / metric | Result | Confidence / allowed wording |
|---|---|---|---|---|
| Local CPU compatibility scoring works | `models/clip_wrapper.py`, archived predictions | POPE-derived object phrases | Scores exist | Engineering capability; not factuality guarantee |
| 67 evaluated items | `research.analyze`, four legacy cohorts, file hashes in report | Unique image/claim pairs | 67 claims, 34 images | Artifact; do not call this 67 independent images |
| Three baseline methods compared | `results/audited/main_table.md` | Same 32 claims, 18 images | Whole SigLIP accuracy .90625/F1 .93023/AUROC .93182; CLIP-grid .9375/.95455/.95455; SigLIP-grid .65625/.66667/.94091 | Exploratory image-compatibility comparison only |
| CLIP-grid is significantly better | `report.json:comparisons` | Paired image-cluster accuracy | Delta .03125, percentile CI [0,.09677], cluster randomization p=1.0 | **Not supported**; no significant superiority claim |
| v3 is better than a strong selective baseline | `report.json:selective_runs` | Deduplicated seeds 3,4; matched coverage | Each 10/17 decided, 10/10 correct; confidence-only baseline also 10/10 at 10/17 | **Not supported**; retain tie |
| v3 is 100% accurate | Same artifacts | Conditional vs unconditional metrics | 10 correct decisions and 7 abstentions per run; only 10 images/run | Say conditional accuracy on these development cases, not a guarantee |
| Risk is calibrated probability | Raw reliability/ECE/Brier analysis | Common 32-claim cohort | ECE .2585/.2966/.3492 for whole/CLIP-grid/SigLIP-grid | **Not supported**; raw heuristics need held-out calibration |
| 98% M0–M4 accuracy generalizes | `synthetic_gbm.json` | Synthetic 800/200 split | .98 holdout; count-pattern shift .308/.319/.301 | Natural diagnosis **unmeasured**; shifted generator shows shortcut reliance |
| Qwen was fine-tuned | Legacy script + artifact inventory | LoRA/SFT | No validated adapter run; trainer has correctness defects and is quarantined | **False** as an achieved result |
| Atomic decomposition is new | FaithScore primary paper | Related work | Atomic fact extraction/verification predates this project | **False** as novelty |
| Diagnosis-aware routing is unprecedented | VISOR primary paper (2026) | Related work | Null-image diagnosis and routed remediation already described | **Unsupported first/novel claim** |
| Repair routing improves answer utility | New controlled runner and independent scorer | Natural repaired answers | N/A | Hypothesis only; requires preserved correctness and cost comparison |
| Complete loop executes | `live_smoke.json`, `controlled_edit_smoke.json`, recording JSONLs | One old development photo, actual answer plus author-edited derivative | Successful stage traces, raw responses and usage retained | Executed smoke only; no independent accuracy annotation |
| Gemini judge achieves 16/17 | Old conversational claim | No durable historical per-item judge trace found | N/A for paper | Exclude until recoverable/repeated under frozen protocol |
| So400M improves accuracy by larger score margin | Old red-square runs with different score windows | Affine-scaled toy scores | Margin not comparable across windows | Exclude as accuracy evidence |
| 4–13% of full compute with no quality loss | `repair/cascade.py` constructed stage costs and perfect synthetic full risks | Synthetic simulation | Not measured physical cost; stage1 uses full output | **Unsupported efficiency result** |
| Latency below one minute | Old UI timer excludes API stages; benchmark timer includes download/load | No matched per-stage historical telemetry | N/A as universal promise | New telemetry reports recorded seconds, not guarantees |
| Gallery proves counting fixes | Saved gallery; no independent instance annotation/NMS audit | Hand-authored examples | Detector proposals may duplicate objects | Not evidence of reliable counting |
| Fifty human failure reviews complete | `failure_review_queue.jsonl` | Legacy cases | Queue pending, annotators null | **Not complete** |
| Test suite proves research validity | Contract tests | Software tests | Passing contracts possible despite leakage | Tests support implementation only |

## Words requiring specific evidence

Preservation update: the ledger, exact-quote alignment checks and reject gate
now execute in the Studio and recorded research runner. The mixed live smoke
reported four protected facts retained and an invented elephant removed, but
this is a model-audited author-created example. Both fixed and routed arms run;
no human gold or measured advantage is available. One failed HTTP request is
retained in the first smoke output, followed by distinct recovery/replay logs.
Do not call these outcomes a natural benchmark, causal proof or publication result.

- **novel/first:** full related-work boundary; direct VISOR/Woodpecker/FaithScore
  overlap precludes the current broad claims.
- **significant:** preregistered primary comparison, paired test, interval and
  multiple-comparison policy. No selective cherry-picking of tests.
- **robust/generalizable:** held-out models/datasets/types; currently absent.
- **efficient:** measured end-to-end cost at matched correctness/preservation,
  counting retries, remote generation and local encoding.
- **cause/proof:** intervention evidence and annotated grounding; rule labels
  and crop matches alone are hypotheses, not identified mechanisms.
- **publication-ready:** independent gold and primary closed-loop experiments
  are still required. The new paper is an audit/protocol draft.
