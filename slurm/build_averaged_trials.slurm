#!/bin/bash
#SBATCH --job-name=vsd_avg_trials
#SBATCH --output=logs/build_averaged_trials_%j.out
#SBATCH --error=logs/build_averaged_trials_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

# Build frame-averaged trial NetCDF files for VSD_Encoder_01.
#
# Expected workspace layout (siblings under same parent):
#   <workspace>/
#   ├── Data/
#   └── VSD_Encoder_01/
#
# One-time setup on login node:
#   bash scripts/cluster_setup.sh
#
# Submit from repo root (or anywhere):
#   sbatch slurm/build_averaged_trials.slurm
#
# Optional environment overrides:
#   WINDOW_CONFIG=configs/windows/baseline_05_25.yaml
#   MONKEY=legolas
#   MAX_TRIALS=100
#   OVERWRITE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_32_42.yaml}"
MONKEY="${MONKEY:-}"
MAX_TRIALS="${MAX_TRIALS:-}"
OVERWRITE="${OVERWRITE:-}"

mkdir -p logs

# Headless plotting on compute nodes
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: ${PYTHON} not found." >&2
  echo "Run once on the login node: bash scripts/cluster_setup.sh" >&2
  exit 1
fi

CMD=("${PYTHON}" scripts/01_build_averaged_trials.py
  --config "${CONFIG}"
  --window "${WINDOW_CONFIG}"
)

if [[ -n "${MONKEY}" ]]; then
  CMD+=(--monkey "${MONKEY}")
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
