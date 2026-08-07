#!/bin/bash
#SBATCH --job-name=looA_maps_raw
#SBATCH --output=experiments/loo_encoding/slurm/logs/02b_maps_raw_%j.out
#SBATCH --error=experiments/loo_encoding/slurm/logs/02b_maps_raw_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
##SBATCH --partition=generic
##SBATCH --account=CHANGE_ME
# Required env: RUN_ROOT
# Computes pooled encoding maps for raw leaves only (raw/clean + raw/all).
# Safe to run in parallel with a zscore-only (or full) assemble job, as long as
# both jobs do not write the same leaf.

set -euo pipefail

# SLURM copies the script to /var/spool/slurmd/… — do not resolve paths via BASH_SOURCE.
: "${REPO_ROOT:?REPO_ROOT must be set (export before sbatch)}"
export PIPELINE_CONFIG="${PIPELINE_CONFIG:-experiments/loo_encoding/slurm/protocol_A_full.yaml}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/loo_encoding/slurm/common.sh"

: "${RUN_ROOT:?Set RUN_ROOT}"
RUN_ROOT_PATH="${RUN_ROOT}"
if [[ "${RUN_ROOT_PATH}" != /* ]]; then
  RUN_ROOT_PATH="${REPO_ROOT}/${RUN_ROOT_PATH}"
fi
MANIFEST="$(load_manifest "${RUN_ROOT_PATH}")"
echo "Manifest: ${MANIFEST}"

echo "Assembling pooled encoding pixel-r maps (raw leaves only)…"
"${PYTHON}" experiments/loo_encoding/assemble_protocol_A_pooled_maps.py \
  --run-root "${RUN_ROOT_PATH}" \
  --only-raw \
  --skip-existing

echo "Raw pooled maps done."
