# Reproducible figures

Canonical audited figures are generated under `results/audited/figures/` by:

```
python -m research.analyze
python -m research.synthetic_diagnosis
python -m research.report_assets
```

PDF (vector) and PNG exports are provided. Every numerical figure consumes
saved predictions or a separately recorded synthetic experiment. No synthetic
result is labeled natural-data performance. Missing repair-reduction and full
ablation figures are listed in `results/audited/missing_figures.json`.
