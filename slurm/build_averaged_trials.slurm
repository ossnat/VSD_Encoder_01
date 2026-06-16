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
# Submit from anywhere:
#   sbatch slurm/build_averaged_trials.slurm
#
# Override window / monkey via environment variables:
#   WINDOW_CONFIG=configs/windows/baseline_05_25.yaml MONKEY=legolas sbatch ...

set -euo pipefail

# Repo root = directory containing this slurm/ folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_30_50.yaml}"
MAX_TRIALS="${MAX_TRIALS:-}"
OVERWRITE="${OVERWRITE:-}"

mkdir -p logs

# Optional: activate your environment
# source "${HOME}/miniconda3/etc/profile.d/conda.sh"
# conda activate vsd-encoder

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(python scripts/01_build_averaged_trials.py
  --config "${CONFIG}"
  --window "${WINDOW_CONFIG}"
)

if [[ -n "${MAX_TRIALS}" ]]; then
  CMD+=(--max-trials "${MAX_TRIALS}")
fi

if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" ]]; then
  CMD+=(--overwrite)
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "WORKSPACE_ROOT=$(dirname "${REPO_ROOT}")"
echo "Running: ${CMD[*]}"
"${CMD[@]}"
