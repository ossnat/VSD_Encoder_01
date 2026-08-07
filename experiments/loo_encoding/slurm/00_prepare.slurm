#!/bin/bash
#SBATCH --job-name=looA_prep
#SBATCH --output=experiments/loo_encoding/slurm/logs/00_prepare_%j.out
#SBATCH --error=experiments/loo_encoding/slurm/logs/00_prepare_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# Partition / account: set via sbatch CLI or edit here.
##SBATCH --partition=generic
##SBATCH --account=CHANGE_ME

set -euo pipefail

# SLURM copies the script to /var/spool/slurmd/… — do not resolve paths via BASH_SOURCE.
: "${REPO_ROOT:?REPO_ROOT must be set (export before sbatch)}"
export PIPELINE_CONFIG="${PIPELINE_CONFIG:-experiments/loo_encoding/slurm/protocol_A_full.yaml}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/loo_encoding/slurm/common.sh"

EXTRA=()
if [[ -n "${RUN_DATE:-}" ]]; then
  EXTRA+=(--run-date "${RUN_DATE}")
fi

echo "Preparing Protocol A pipeline leaves (dry-run folds + folds.txt)…"
"${PYTHON}" experiments/loo_encoding/prepare_protocol_A_pipeline.py \
  --config "${PIPELINE_CONFIG}" \
  "${EXTRA[@]}"

echo "Done prepare."
