#!/bin/bash
#SBATCH --job-name=vsd_sweep_resnet18
#SBATCH --output=logs/sweep_resnet18_layers_%j.out
#SBATCH --error=logs/sweep_resnet18_layers_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_35_42.yaml}"
RIDGE_CONFIG="${RIDGE_CONFIG:-configs/ridge/default.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/resnet18.yaml}"
MONKEY="${MONKEY:-}"
DEVICE="${DEVICE:-auto}"
SPLIT="${SPLIT:-val}"
LAYERS="${LAYERS:-layer2 layer3 layer4 avgpool}"

mkdir -p logs
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: ${PYTHON} not found. Run: bash scripts/cluster_setup.sh" >&2
  exit 1
fi

CMD=("${PYTHON}" scripts/09_sweep_feature_layers.py
  --config "${CONFIG}"
  --window "${WINDOW_CONFIG}"
  --ridge-config "${RIDGE_CONFIG}"
  --model "${MODEL_CONFIG}"
  --split "${SPLIT}"
  --device "${DEVICE}"
  --layers ${LAYERS}
)

if [[ -n "${MONKEY}" ]]; then
  CMD+=(--monkey "${MONKEY}")
fi

if [[ "${COMPARE_ONLY:-}" == "1" ]]; then
  CMD+=(--compare-only)
fi
if [[ "${SKIP_EXTRACT:-}" == "1" ]]; then
  CMD+=(--skip-extract)
fi
if [[ "${SKIP_TRAIN:-}" == "1" ]]; then
  CMD+=(--skip-train)
fi
if [[ "${SKIP_EVAL:-}" == "1" ]]; then
  CMD+=(--skip-eval)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
