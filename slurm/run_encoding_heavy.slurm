#!/bin/bash
#SBATCH --job-name=vsd_enc_heavy
#SBATCH --output=logs/encoding_heavy_%j.out
#SBATCH --error=logs/encoding_heavy_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu

# Run stages 02b + 03 in one SLURM job (alternative to submit_encoding_jobs.sh).
#
# Submit:
#   sbatch slurm/run_encoding_heavy.slurm
#
# Overrides:
#   MONKEY=gandalf  CONFIG=...  WINDOW_CONFIG=...  MODEL_CONFIG=...
#   RIDGE_CONFIG=...  DEVICE=auto|cpu|cuda  OVERWRITE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_32_42.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/resnet18.yaml}"
RIDGE_CONFIG="${RIDGE_CONFIG:-configs/ridge/default.yaml}"
DEVICE="${DEVICE:-auto}"
MONKEY="${MONKEY:-}"
OVERWRITE="${OVERWRITE:-}"

mkdir -p logs
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: ${PYTHON} not found. Run: bash scripts/cluster_setup.sh" >&2
  exit 1
fi

echo "=== Stage 02b: stimulus CNN features ==="
FEAT_CMD=("${PYTHON}" scripts/02b_extract_stimulus_features.py
  --config "${CONFIG}"
  --model "${MODEL_CONFIG}"
  --device "${DEVICE}"
)
if [[ -n "${MONKEY}" ]]; then
  FEAT_CMD+=(--monkey "${MONKEY}")
fi
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" ]]; then
  FEAT_CMD+=(--overwrite)
fi
echo "Running: ${FEAT_CMD[*]}"
"${FEAT_CMD[@]}"

echo ""
echo "=== Stage 03: RidgeCV encoder ==="
RIDGE_CMD=("${PYTHON}" scripts/03_train_ridge_encoder.py
  --config "${CONFIG}"
  --window "${WINDOW_CONFIG}"
  --ridge-config "${RIDGE_CONFIG}"
  --model "${MODEL_CONFIG}"
)
if [[ -n "${MONKEY}" ]]; then
  RIDGE_CMD+=(--monkey "${MONKEY}")
fi
echo "Running: ${RIDGE_CMD[*]}"
"${RIDGE_CMD[@]}"

echo "Encoding heavy stages complete."
