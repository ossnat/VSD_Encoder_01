#!/bin/bash
#SBATCH --job-name=vsd_stim_feat
#SBATCH --output=logs/extract_stimulus_features_%j.out
#SBATCH --error=logs/extract_stimulus_features_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/resnet18.yaml}"
MONKEY="${MONKEY:-}"
FEATURE_LAYER="${FEATURE_LAYER:-}"
OVERWRITE="${OVERWRITE:-}"
DEVICE="${DEVICE:-auto}"

mkdir -p logs
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: ${PYTHON} not found. Run: bash scripts/cluster_setup.sh" >&2
  exit 1
fi

CMD=("${PYTHON}" scripts/02b_extract_stimulus_features.py
  --config "${CONFIG}"
  --model "${MODEL_CONFIG}"
  --device "${DEVICE}"
)

if [[ -n "${MONKEY}" ]]; then
  CMD+=(--monkey "${MONKEY}")
fi
if [[ -n "${FEATURE_LAYER}" ]]; then
  CMD+=(--feature-layer "${FEATURE_LAYER}")
fi
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" ]]; then
  CMD+=(--overwrite)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
