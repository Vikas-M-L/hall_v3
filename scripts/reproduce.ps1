$ErrorActionPreference = "Stop"
python -m research.analyze --config configs/reanalysis.json --out results/audited
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m research.synthetic_diagnosis --out results/audited/synthetic_gbm.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m research.report_assets
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m pytest research/tests -q
exit $LASTEXITCODE
