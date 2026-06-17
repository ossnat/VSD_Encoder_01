#!/bin/bash
#SBATCH --job-name=vsd_enc_pairs
#SBATCH --output=logs/build_encoding_pairs_%j.out
#SBATCH --error=logs/build_encoding_pairs_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_32_42.yaml}"
MONKEY="${MONKEY:-}"
REQUIRE_NC="${REQUIRE_NC:-}"

mkdir -p logs
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: ${PYTHON} not found. Run: bash scripts/cluster_setup.sh" >&2
  exit 1
fi

CMD=("${PYTHON}" scripts/01c_build_encoding_pairs.py
  --config "${CONFIG}"
  --window "${WINDOW_CONFIG}"
)

if [[ -n "${MONKEY}" ]]; then
  CMD+=(--monkey "${MONKEY}")
fi
if [[ "${REQUIRE_NC}" == "1" || "${REQUIRE_NC}" == "true" ]]; then
  CMD+=(--require-nc)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
