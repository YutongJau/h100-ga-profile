#!/bin/bash
#SBATCH --job-name=h100tc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Portable SLURM template. Edit the partition/account/module lines for the
# target cluster before submitting.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p logs

# Example cluster-specific setup. Keep only what the Oxford environment needs.
# module purge
# module load cuda
# module load python
# source /path/to/venv/bin/activate

export RUN_TAG="${RUN_TAG:-slurm_${SLURM_JOB_ID:-manual}}"
export OUT_DIR="${OUT_DIR:-${PWD}/h100_profile_outputs_${SLURM_JOB_ID:-manual}}"
export STEPS="${STEPS:-100}"
export WARMUP="${WARMUP:-20}"
export MBS="${MBS:-1}"
export SEQ_LEN="${SEQ_LEN:-1024}"
export SEED="${SEED:-1234}"
export GA_LIST="${GA_LIST:-1 2 4}"
export PYTHON="${PYTHON:-python3}"

bash "${SCRIPT_DIR}/run_with_telemetry.sh"
