# V-TRACE+: audited research prototype

**Research question:** Can predicted error-type routing improve hallucination
correction and answer preservation over uniform self-correction at matched
cost and coverage?

Current status: a CPU-friendly Studio prototype, standalone synthetic diagnosis
models, archived exploratory scores, and a new recorded closed-loop research
protocol. **Natural-data repair superiority, causal attribution, calibrated
probabilities and compute savings have not been established.**

## Read the audit first

- [Research audit](research_audit.md): code defects, leakage, unsupported claims.
- [Literature review](literature_review.md): verified primary sources, including
  direct overlap with VISOR, Woodpecker and FaithScore.
- [Dataset card](dataset_card.md): collection/annotation protocol; no completed
  new human-annotated benchmark is claimed.
- [Claim audit](claim_audit.md), [reviewer perspectives](reviewer_report.md),
  [roadmap](FINAL_RESEARCH_ROADMAP.md), [reproduction](REPRODUCIBILITY.md).

## Actual findings

The audited legacy pool contains **67 claims from 34 images**, not 67 independent
images. A matched 32-claim/18-image comparison is in
[`results/audited/main_table.md`](results/audited/main_table.md).
Each deduplicated v3 run decides 10/17 correctly; a confidence-only baseline
matches it at the same coverage. Thus **no v3 advantage is established**.
The offline GBM's 98% synthetic holdout accuracy drops to 30.1–31.9% under a
controlled count-pattern shift. These negative results remain visible.
An extended fifth POPE run grows the deduplicated pool to **88 unique claims**:
whole-image SigLIP AUROC 0.926/F1 0.906 (n=88), CLIP-B/32 grid 0.942/0.932
(n=71) — stability confirmation, not a new best-result claim.

## Human annotation (the blocking step)

`research/annotation_kit/` holds the annotator guidelines, the CSV template,
and `research/annotation_csv.py` (CSV → adjudicated JSONL converter). Two
annotators label independently; disagreements go to adjudicator C. Nothing is
annotated yet — this kit is the prerequisite for every publication gate.

## Layout

| Path | Purpose |
|---|---|
| `research/` | Audited inputs/gold isolation, API record/replay, bounded repair and verification, metrics/statistics/calibration |
| `configs/` | Reanalysis, development loop, GBM and blocked LoRA protocol |
| `results/audited/` | Recomputed tables, figures, source hashes, review queue, recorded smoke traces |
| `experiments/` | Actual API smoke recordings and clearly labeled synthetic replay fixture |
| `vtrace-plus/` | Legacy Studio, frozen encoders, optional OWL/Gemini, rule diagnosis and offline GBM |
| `adaptive-vtrace/` | Legacy experiments; audit required before scientific use |
| `paper/main.tex` | Audited IEEE-style paper; old draft archived as unvalidated |

## Quick start

```powershell
python -m pip install -r requirements.txt
python -m research.analyze
python -m research.synthetic_diagnosis
python -m research.report_assets
python -m pytest research/tests -q
```

For the Studio, install its optional dependencies and run:

```powershell
python -m pip install -r vtrace-plus/requirements.txt
python -m pip install streamlit pandas python-dotenv
python -m streamlit run vtrace-plus/app.py
```

Open `http://localhost:8501`. Risk scores and M0–M4 labels are prototype
heuristics. Counterclaim comparisons and detector proposals are not proof.
Gemini calls use a locally configured key; never commit credentials.

## Closed-loop experiments

### Preservation-aware repair (implemented, empirical benefit unmeasured)

The Studio correction button now re-verifies original facts, builds an
evidence ledger, checks every candidate fact, and audits semantic preservation
with exact candidate quotes and complete fact-ID coverage. Dropped/changed
supported facts, retained refuted facts, unresolved candidate evidence or
malformed audits reject the candidate. The original answer remains available.
These are fallible model checks, not human correctness labels.

`configs/preservation.json` compares predicted routing, fixed repair, and an
audit-compute-matched gate-disabled ablation. See
[`research/PRESERVATION_STUDY.md`](research/PRESERVATION_STUDY.md) for commands,
schema and publication gates. Recorded live smoke tests and a replay are
available in `experiments/` and `results/audited/preservation*`; they are
controlled development examples, not a held-out benchmark.

```powershell
python -m research.make_fixture
python -m research.run --config configs/closed_loop.json --inputs experiments/fixture/inputs.jsonl --recording experiments/fixture/recording.jsonl --out results/audited/fixture_replay.json
python -m research.ablations
```

The fixture is synthetic plumbing, not an accuracy experiment. Live collection,
CPU training, probability calibration, annotation requirements and independent
repair scoring are documented in `REPRODUCIBILITY.md`. The GBM remains offline;
Qwen training is quarantined until its identified defects are repaired.

No benchmark/acceptance guarantee is made. Third-party dataset/model rights must
be checked independently before redistribution or public deployment.
