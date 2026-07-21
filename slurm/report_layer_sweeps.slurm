#!/bin/bash
#SBATCH --job-name=vsd_layer_report
#SBATCH --output=logs/report_layer_sweeps_%j.out
#SBATCH --error=logs/report_layer_sweeps_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_35_42.yaml}"
RIDGE_CONFIG="${RIDGE_CONFIG:-configs/ridge/default.yaml}"
RESNET_MODEL="${RESNET_MODEL:-configs/models/resnet18.yaml}"
VGG_MODEL="${VGG_MODEL:-configs/models/vgg16.yaml}"
MONKEY="${MONKEY:-}"
SELECTION_SPLIT="${SELECTION_SPLIT:-val}"
TEST_SPLIT="${TEST_SPLIT:-test}"

mkdir -p logs
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: ${PYTHON} not found. Run: bash scripts/cluster_setup.sh" >&2
  exit 1
fi

CMD=("${PYTHON}" scripts/10_report_layer_sweeps.py
  --config "${CONFIG}"
  --window "${WINDOW_CONFIG}"
  --ridge-config "${RIDGE_CONFIG}"
  --resnet-model "${RESNET_MODEL}"
  --vgg-model "${VGG_MODEL}"
  --selection-split "${SELECTION_SPLIT}"
  --test-split "${TEST_SPLIT}"
)

if [[ -n "${MONKEY}" ]]; then
  CMD+=(--monkey "${MONKEY}")
fi

if [[ "${COMPARE_ONLY:-}" == "1" ]]; then
  CMD+=(--compare-only)
fi
if [[ "${SKIP_TEST_EVAL:-}" == "1" ]]; then
  CMD+=(--skip-test-eval)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
