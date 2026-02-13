#!/bin/bash
set -euo pipefail

EXPERIMENT_ID="exp_pop_mini_X_temporal_36h"
DESCRIPTION="POP baseline temporal stability with 36h train, 12h test, 12h stride across 144h MIND dataset"
MODEL="POP"
DATASET="mind_mini_impressions"
WINDOW_SIZE=48
WINDOW_RATIO="36:12"
TOTAL_UNITS=144
GRANULARITY="hour"
WINDOW_STRIDE=12
RUNS=3

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/experiment_runner.sh"

#!/bin/bash
set -euo pipefail

EXPERIMENT_ID="exp_pop_small_X_temporal_36h"
DESCRIPTION="POP baseline temporal stability with 36h train, 12h test, 12h stride across 144h MIND dataset"
MODEL="POP"
DATASET="mind_small_impressions"
WINDOW_SIZE=48
WINDOW_RATIO="36:12"
TOTAL_UNITS=144
GRANULARITY="hour"
WINDOW_STRIDE=12
RUNS=3

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/experiment_runner.sh"

#!/bin/bash
set -euo pipefail

EXPERIMENT_ID="exp_pop_medium_X_temporal_36h"
DESCRIPTION="POP baseline temporal stability with 36h train, 12h test, 12h stride across 144h MIND dataset"
MODEL="POP"
DATASET="mind_medium_impressions"
WINDOW_SIZE=48
WINDOW_RATIO="36:12"
TOTAL_UNITS=144
GRANULARITY="hour"
WINDOW_STRIDE=12
RUNS=3

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/experiment_runner.sh"