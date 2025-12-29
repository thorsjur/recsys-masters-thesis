#!/bin/bash
set -euo pipefail

EXPERIMENT_ID="exp_tfidf_temporal_36h_fixed"
DESCRIPTION="TFIDF temporal stability with 36h train, 12h test, 12h stride across 144h MIND dataset - FIXED"
MODEL="TFIDF"
DATASET="mind_small"
WINDOW_SIZE=48
WINDOW_RATIO="36:12"
TOTAL_UNITS=144
GRANULARITY="hour"
WINDOW_STRIDE=12
RUNS=3

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/experiment_runner.sh"