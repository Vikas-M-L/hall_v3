# Checkpoints

The legacy standalone numeric GBM is stored locally under
`vtrace-plus/models/vtrace_m0_m4_diagnosis.joblib`; its SHA256 is recorded by
the audit. It was trained on synthetic numeric clusters, not natural images.

No validated Qwen LoRA/SFT checkpoint exists. Future research artifacts must
include model/data/config hashes, group split manifests, provenance and held-out
metrics. `research.train_diagnosis` saves a GBM bundle and report here when
independently annotated train/calibration data is supplied. No app integration.
