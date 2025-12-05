#!/bin/bash

ENV_NAME="recsys_stability"

# 1. Activate Conda Environment
# Using single brackets [ ] for compatibility with stricter shells
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "Activating environment: $ENV_NAME..."

    # Try activating. If it fails (returns non-zero), we source the conda setup.
    conda activate "$ENV_NAME" 2>/dev/null

    if [ $? -ne 0 ]; then
        # Find where conda is installed
        CONDA_BASE=$(conda info --base)
        # Source the conda.sh script to enable the 'activate' command
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            conda activate "$ENV_NAME"
        else
            echo "Could not find conda.sh at $CONDA_BASE"
        fi
    fi
else
    echo "Environment '$ENV_NAME' is already active."
fi

# 2. Set PYTHONPATH
# This fixes the "ModuleNotFoundError"
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $(pwd)"

# 3. System Memory Check
echo ""
echo "System Memory Status:"
# Simple formatting that works on all shells
free -h | grep "Mem:" | awk '{print "   Total RAM: " $2 " | Available: " $7}'
echo "   Swap Space: $(free -h | grep "Swap:" | awk '{print $2}')"

echo ""
echo "Ready."
