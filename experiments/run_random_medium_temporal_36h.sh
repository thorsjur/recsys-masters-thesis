#!/bin/bash
set -euo pipefail

EXPERIMENT_ID="exp_random_medium_2_temporal_36h"
DESCRIPTION="Random baseline temporal stability with 36h train, 6h test, 6h stride across 168h MIND medium dataset"
MODEL="Random"
DATASET="mind_medium_impressions"
WINDOW_SIZE=48
WINDOW_RATIO="42:6"
TOTAL_UNITS=168
GRANULARITY="hour"
WINDOW_STRIDE=6
RUNS=1

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/experiment_runner.sh"