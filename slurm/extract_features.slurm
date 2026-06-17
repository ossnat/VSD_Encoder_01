#!/bin/bash
#SBATCH --job-name=vsd_extract_feat
#SBATCH --output=logs/extract_features_%j.out
#SBATCH --error=logs/extract_features_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_30_50.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/resnet18.yaml}"
MONKEY="${MONKEY:-}"
FEATURE_LAYER="${FEATURE_LAYER:-}"
MAX_TRIALS="${MAX_TRIALS:-}"
OVERWRITE="${OVERWRITE:-}"
DEVICE="${DEVICE:-auto}"

mkdir -p logs
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: ${PYTHON} not found. Run: bash scripts/cluster_setup.sh" >&2
  exit 1
fi

CMD=("${PYTHON}" scripts/02_extract_features.py
  --config "${CONFIG}"
  --window "${WINDOW_CONFIG}"
  --model "${MODEL_CONFIG}"
  --device "${DEVICE}"
)

if [[ -n "${MONKEY}" ]]; then
  CMD+=(--monkey "${MONKEY}")
fi
if [[ -n "${FEATURE_LAYER}" ]]; then
  CMD+=(--feature-layer "${FEATURE_LAYER}")
fi
if [[ -n "${MAX_TRIALS}" ]]; then
  CMD+=(--max-trials "${MAX_TRIALS}")
fi
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" ]]; then
  CMD+=(--overwrite)
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "WORKSPACE_ROOT=$(dirname "${REPO_ROOT}")"
echo "PYTHON=$("${PYTHON}" --version)"
echo "Running: ${CMD[*]}"
"${CMD[@]}"
