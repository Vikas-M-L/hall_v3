# Baseline and experiment matrix

N/A means no completed independently scored experiment, not zero performance.

| Arm | Implementation status | Natural detection F1/AUROC | CHAIR/AMBER/official POPE | Hallucination reduction / answer accuracy / repair success | Latency / cost |
|---|---|---|---|---|---|
| Original VLM | Collector + stored actual answer | N/A | N/A | N/A | Generation trace present for one development image |
| Prompt detector | Research API backend | N/A | N/A | N/A | Smoke stages recorded |
| Uniform reinspection/correction | Prompt adaptation, with verifier | N/A | N/A | N/A | Smoke stages recorded |
| Authentic existing detector | Required (e.g. VisER/SpanCalib, check release) | N/A | N/A | N/A | N/A |
| Authentic existing mitigation | Required (Woodpecker/VCD) | N/A | N/A | N/A | N/A |
| Fine-tuned detector | Standalone synthetic GBM only; LoRA blocked | N/A | N/A | N/A | Synthetic training telemetry only |
| Legacy SigLIP/CLIP | Archived compatibility scores | Matched-cohort table, exploratory | Not official generated-answer POPE | N/A | Historic per-method costs unavailable |
| Detector + fixed repair | Runnable protocol arm | N/A | N/A | N/A | Must record all cases |
| Detector + predicted type router | Runnable protocol arm | N/A | N/A | N/A | Must record all cases |
| Router + repair without verifier | Runnable ablation | N/A | N/A | N/A | Must record all cases |
| Full router + repair + verifier | Executed development smoke | N/A | N/A | Independent adjudication required | Actual smoke records, not efficiency proof |

Do not copy paper benchmark numbers into this table as local reproductions.
The root closed-loop runner is a controlled API protocol, not the frozen local
multisignal Studio backend; integrating recorded frozen evidence is a separate
adapter task before claiming their combination was evaluated.
