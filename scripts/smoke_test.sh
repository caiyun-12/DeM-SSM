#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m dem_ssm.cli toy-data --out data/toy --image-size 32 --frames 3
python -m dem_ssm.cli audit --manifest data/toy/manifest.csv --split data/toy/split.json --check-videos --out outputs/toy_audit.json
python -m dem_ssm.cli train --config configs/smoke.yaml --manifest data/toy/manifest.csv --split data/toy/split.json --out outputs/toy_run
python -m dem_ssm.plots --predictions outputs/toy_run/predictions_test.csv --out outputs/toy_plots
python -m pytest -q
