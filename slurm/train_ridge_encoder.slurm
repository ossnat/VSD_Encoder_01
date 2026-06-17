#!/bin/bash
#SBATCH --job-name=vsd_ridge
#SBATCH --output=logs/train_ridge_encoder_%j.out
#SBATCH --error=logs/train_ridge_encoder_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_32_42.yaml}"
RIDGE_CONFIG="${RIDGE_CONFIG:-configs/ridge/default.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/resnet18.yaml}"
MONKEY="${MONKEY:-}"

mkdir -p logs
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: ${PYTHON} not found. Run: bash scripts/cluster_setup.sh" >&2
  exit 1
fi

CMD=("${PYTHON}" scripts/03_train_ridge_encoder.py
  --config "${CONFIG}"
  --window "${WINDOW_CONFIG}"
  --ridge-config "${RIDGE_CONFIG}"
  --model "${MODEL_CONFIG}"
)

if [[ -n "${MONKEY}" ]]; then
  CMD+=(--monkey "${MONKEY}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
