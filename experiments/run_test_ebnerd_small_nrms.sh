#!/bin/bash
set -euo pipefail

EXPERIMENT_ID="exp_nrms_ebnerd_demo_temporal_36h"
DESCRIPTION="NRMS temporal stability with 36h train, 6h valid, 6h test, 6h stride across 168h EBNeRD dataset"
MODEL="NRMS_ebnerd_s"
DATASET="ebnerd_demo_impressions"
WINDOW_SIZE=48
WINDOW_RATIO="36:6:6"
TOTAL_UNITS=168
GRANULARITY="hour"
WINDOW_STRIDE=6
RUNS=1
PARAMS=(
  "embedding_source=fasttext"
  "embedding_path=~/fasttext/cc.da.300.bin"
)

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/experiment_runner.sh"