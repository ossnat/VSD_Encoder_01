#!/bin/bash
#SBATCH --job-name=vsd_stimuli
#SBATCH --output=logs/build_stimulus_images_%j.out
#SBATCH --error=logs/build_stimulus_images_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
STIMULI_CONFIG="${STIMULI_CONFIG:-configs/stimuli/default.yaml}"
MONKEY="${MONKEY:-}"
OVERWRITE="${OVERWRITE:-}"

mkdir -p logs
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: ${PYTHON} not found. Run: bash scripts/cluster_setup.sh" >&2
  exit 1
fi

CMD=("${PYTHON}" scripts/01b_build_stimulus_images.py
  --config "${CONFIG}"
  --stimuli-config "${STIMULI_CONFIG}"
)

if [[ -n "${MONKEY}" ]]; then
  CMD+=(--monkey "${MONKEY}")
fi
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" ]]; then
  CMD+=(--overwrite)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
