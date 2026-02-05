#!/bin/bash
set -euo pipefail

EXPERIMENT_ID="exp_nrms_temporal_2_36h"
DESCRIPTION="NRMS temporal stability with 36h train, 6h valid, 6h test, 6h stride across 144h MIND dataset"
MODEL="NRMS"
DATASET="mind_small_impressions"
WINDOW_SIZE=48
WINDOW_RATIO="36:6:6"
TOTAL_UNITS=144
GRANULARITY="hour"
WINDOW_STRIDE=6
RUNS=3

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/experiment_runner.sh"