#!/bin/bash
########################################################
# IDUN HPC ETL Job Script
# Config: mind_small_no_preprocessing
# Generated automatically - do not edit manually
########################################################

#SBATCH --job-name=etl_mind_small_no_preprocessing
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=output/slurm_logs/etl/etl_mind_small_no_preprocessing_%j.out
#SBATCH --error=output/slurm_logs/etl/etl_mind_small_no_preprocessing_%j.err

set -e  # Exit on first error

echo "=========================================="
echo "IDUN ETL Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Config: mind_small_no_preprocessing"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_JOB_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="

# Working directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Load modules
module purge
module load Miniconda3/24.7.1-0
module list

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /cluster/home/thorsj/recsys_stability
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Run ETL pipeline
echo "Starting ETL pipeline..."
python run_etl.py \
    --config mind_small_no_preprocessing \
    --temporal-hours 168

EXIT_CODE=$?

echo "=========================================="
echo "ETL completed"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
