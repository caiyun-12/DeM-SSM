$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path "$PSScriptRoot/..")
python -m dem_ssm.cli toy-data --out data/toy --image-size 32 --frames 3
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m dem_ssm.cli audit --manifest data/toy/manifest.csv --split data/toy/split.json --check-videos --out outputs/toy_audit.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m dem_ssm.cli train --config configs/smoke.yaml --manifest data/toy/manifest.csv --split data/toy/split.json --out outputs/toy_run
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m dem_ssm.plots --predictions outputs/toy_run/predictions_test.csv --out outputs/toy_plots
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m pytest -q
exit $LASTEXITCODE
