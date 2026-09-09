# Reproduction guide

All commands below run from the repository root using Python 3.11. The new
research commands do not import the two legacy packages named `fusion` into
one process. No secret belongs in a command line or recorded artifact.

## Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

This root environment supports the audited analysis/runner. Optional Studio
models need `vtrace-plus/requirements.txt`, Streamlit and compatible PyTorch.
Current audit versions are saved in `results/audited/environment.json`;
`environment.yml` describes the root CPU environment. The old GPU requirements
are not validated by this audit. Pin Hugging Face revisions before new model runs.

## One command per audited experiment

```powershell
python -m research.analyze --config configs/reanalysis.json --out results/audited
python -m research.synthetic_diagnosis --out results/audited/synthetic_gbm.json
python -m research.report_assets
python -m research.inventory
python -m pytest research/tests -q
```

Analysis uses saved legacy outputs, no API/download calls. Source-file hashes,
deduplication keys, per-method cohorts, bootstrap settings and environment are
recorded. Matched-cohort comparisons use the common intersection. Old plots
are not silently overwritten; audited figures live in `results/audited/figures`.

## Label-free closed-loop replay (no API spend)

```powershell
python -m research.make_fixture
python -m research.run --config configs/closed_loop.json --inputs experiments/fixture/inputs.jsonl --recording experiments/fixture/recording.jsonl --out results/audited/fixture_replay.json
python -m research.run --config configs/smoke_live.json --inputs experiments/live_inputs.jsonl --recording experiments/live_smoke.jsonl --out results/audited/live_replay.json
```

The first dataset is synthetic plumbing only; the second replays actual API
responses on a development image. Replay cannot generate missing calls or
pretend that missing data was supported. Edit prompts/models => new recordings.

## Collect actual VLM responses and execute repairs

Set `GEMINI_API_KEY` in your environment locally. `research` does not implicitly
load the Studio's `.env`. Images and question/response are transmitted to the
configured provider in live mode. Model aliases may change; returned version
is logged. Defaults are existing project models, not a claim of best accuracy.

```powershell
python -m research.collect --image vtrace-plus/assets/coco_cats.jpg --question "Describe the animals and their visible positions." --id dev-001 --group-id COCO_val2014_000000075591 --config configs/smoke_live.json --recording experiments/my_calls.jsonl --out experiments/my_inputs.jsonl
python -m research.run --config configs/smoke_live.json --inputs experiments/my_inputs.jsonl --recording experiments/my_calls.jsonl --out results/my_live_run.json --live
```

Maximum calls and timeout are explicit. No silent retry/fallback to a different
model. Failures are recorded and cause nonzero exit. Shared request cache avoids
repeated API spend; logical policy calls and actual network calls are distinct.
Original answer generation is a separate recorded stage and must be added to
total system costs when comparing complete pipelines.
Client RSS is sampled when psutil is available; it excludes remote server memory
and can miss short allocation peaks. API currency cost stays N/A without a
dated provider price table. A replay's local runtime is not model latency.

## Ground truth and repair scoring

Supply image-group-isolated inputs using `dataset_card.md`. Gold does not go
to the model. Independently adjudicate original/final hallucination and answer
correctness. Required outcome fields: `id`, `strategy`, `label_source`,
`original_hallucinated`, `original_correct`, `abstained`, `final_hallucinated`,
`final_correct`. For abstentions final labels may be null. Accepted label sources
are `human_adjudicated` and `exact_official_answer`; API approval is not gold.

```powershell
python -m research.score_repairs --adjudicated experiments/adjudicated_outcomes.jsonl --out results/repair_metrics.json
```

No adjudicated research outcomes are provided yet. Do not run a metric command
on fabricated labels to fill a paper table.

Prepare independently adjudicated data and commit a protocol before test use:

```powershell
python -m research.prepare_dataset --annotations experiments/annotations.jsonl --salt release-v1 --development-exclusion results/audited/development_exclusion.json --out experiments/release_v1
python -m research.seal --config configs/frozen_test.json --inputs experiments/release_v1/test_inputs.jsonl --out experiments/release_v1/seal.json
```

The second command requires a finalized configuration with `status: frozen`;
then pass `--seal` to `research.run`. Inputs/gold are exported separately;
test gold belongs with an independent evaluation custodian. No clean test
configuration or annotated release is fabricated in this repository.

## Calibration and diagnosis training

```powershell
python -m research.calibration --calibration experiments/calibration_scores.jsonl --evaluation experiments/test_scores.jsonl --out results/calibration.json
python -m research.train_diagnosis --data experiments/natural_diagnosis_train_calibration.csv --config configs/diagnosis_gbm.json --out checkpoints/diagnosis_gbm
python -m research.ablations --config configs/closed_loop.json --out experiments/ablation_configs
```

Calibration rows require `id,group_id,split,label,score`. Training CSV requires
the six numeric features plus `id,group_id,split,diagnosis,label_source`; it
rejects test rows and group overlap. No natural training/calibration set exists
yet. Missing model confidence must not be invented. GBM artifacts stay offline.
The legacy LoRA launcher is quarantined; `configs/lora_protocol.json` lists
conditions needed before a valid GPU run.

## Studio and legacy tests

```powershell
python -m streamlit run vtrace-plus/app.py
python -m pytest vtrace-plus/tests -q
python -m pytest adaptive-vtrace/tests -q
```

Legacy model tests may download weights. Never combine the legacy suites in one
pytest process due to package-name collisions. Their passing does not establish
natural-data research claims.

## Paper build

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Run in `paper/`, after generating audited results/assets. IEEEtran is required.
Upload `paper/main.tex` and the repository's `results/audited/` hierarchy to
Overleaf if TeX is not installed. The paper marks all missing studies N/A and
does not claim the protocol is a completed benchmark.
