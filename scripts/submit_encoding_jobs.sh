#!/usr/bin/env bash
# Submit SLURM jobs for compute-heavy encoding stages (02b CNN + 03 RidgeCV).
#
# Usage (from repo root, after run_prepare_encoding.sh):
#   bash scripts/submit_encoding_jobs.sh
#
# Environment overrides (forwarded to sbatch scripts):
#   MONKEY=gandalf
#   CONFIG=configs/default.yaml
#   WINDOW_CONFIG=configs/windows/evoked_32_42.yaml
#   MODEL_CONFIG=configs/models/resnet18.yaml
#   RIDGE_CONFIG=configs/ridge/default.yaml
#   DEVICE=auto|cpu|cuda
#   OVERWRITE=1
#   SKIP_FEATURES=1
#   SKIP_RIDGE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

export CONFIG="${CONFIG:-configs/default.yaml}"
export WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_32_42.yaml}"
export MODEL_CONFIG="${MODEL_CONFIG:-configs/models/resnet18.yaml}"
export RIDGE_CONFIG="${RIDGE_CONFIG:-configs/ridge/default.yaml}"
export DEVICE="${DEVICE:-auto}"
export MONKEY="${MONKEY:-}"
export OVERWRITE="${OVERWRITE:-}"

JOB_FEATURES=""
if [[ "${SKIP_FEATURES:-}" != "1" ]]; then
  echo "Submitting stage 02b (stimulus CNN features)..."
  JOB_FEATURES="$(sbatch --parsable slurm/extract_stimulus_features.slurm)"
  echo "  Job ID: ${JOB_FEATURES}"
fi

if [[ "${SKIP_RIDGE:-}" != "1" ]]; then
  echo "Submitting stage 03 (RidgeCV encoder)..."
  if [[ -n "${JOB_FEATURES}" ]]; then
    JOB_RIDGE="$(sbatch --parsable --dependency=afterok:"${JOB_FEATURES}" slurm/train_ridge_encoder.slurm)"
  else
    JOB_RIDGE="$(sbatch --parsable slurm/train_ridge_encoder.slurm)"
  fi
  echo "  Job ID: ${JOB_RIDGE}"
fi

echo ""
echo "Jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: ${REPO_ROOT}/logs/"
