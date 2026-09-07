#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config/east_river_config.yaml}

# Preflight first so path/time/unit problems fail before the long run.
east-river validate --config "$CONFIG"
# run-all performs spatial preparation, preprocessing, ASO QC, H1-H4 metrics,
# Tables 3-7, 26 baseline figures, 15 scientific figures, and the manifest.
east-river run-all --config "$CONFIG"
