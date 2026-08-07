#!/bin/bash
#SBATCH --job-name=looA_pdf
#SBATCH --output=experiments/loo_encoding/slurm/logs/04_report_%j.out
#SBATCH --error=experiments/loo_encoding/slurm/logs/04_report_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
##SBATCH --partition=generic
##SBATCH --account=CHANGE_ME
# Required env: RUN_ROOT

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

"${PYTHON}" experiments/loo_encoding/build_protocol_A_report_pdf.py \
  --run-root "${RUN_ROOT_PATH}" \
  --max-overview-pages-per-leaf "${MAX_OVERVIEW_PAGES:-3}"

echo "Report written under ${RUN_ROOT_PATH}/report.pdf"
