#!/bin/bash
#SBATCH --job-name=vsd_ridge_eval
#SBATCH --output=logs/ridge_evaluation_%j.out
#SBATCH --error=logs/ridge_evaluation_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu

# Re-run stages 03 + 04 (+ optional 05) with current ridge/evaluation config.
#
# Submit:
#   sbatch slurm/run_ridge_evaluation.slurm
#
# Overrides (same as scripts/run_ridge_evaluation.sh):
#   MONKEY=gandalf
#   CONFIG=configs/default.yaml
#   WINDOW_CONFIG=configs/windows/evoked_32_42.yaml
#   RIDGE_CONFIG=configs/ridge/default.yaml
#   MODELS="configs/models/resnet18.yaml configs/models/gabor_serre.yaml"
#   SPLITS="test val"
#   SKIP_RIDGE=1  SKIP_EVAL=1  SKIP_COMPARE=1  RUN_COMPARE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs
bash scripts/run_ridge_evaluation.sh
