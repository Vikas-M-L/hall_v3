# hall_v3 — V-TRACE+ Studio

Inference-time hallucination diagnosis and repair-routing for vision-language
models. Split VLM output into atomic claims, verify each against the image with
frozen models, fuse into risk scores, diagnose the failure mechanism (M0–M4),
and route the cheapest repair. No training anywhere.

* `vtrace-plus/` — product: Studio website (`streamlit run app.py`), inference
  pipeline, deterministic fusion, Gemini/ SigLIP / OWL-ViT backends, 52 tests.
  See `vtrace-plus/README.md`.
* `adaptive-vtrace/` — research track: VIG estimator, two-stage mechanism
  classifier, POPE benchmark runner, repair-routing experiment table,
  cost-aware cascade, 35 tests.
* `paper/` — IEEE-format draft (`main.tex`).
* `v-trace-reframed-plan.md` — the 8-week research plan this implements.

Quick start: `pip install -r vtrace-plus/requirements.txt` →
`streamlit run app.py` in `vtrace-plus/` → http://localhost:8501.
