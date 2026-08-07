#!/bin/bash
#SBATCH --job-name=looA_noise
#SBATCH --output=experiments/loo_encoding/slurm/logs/03_noise_%j.out
#SBATCH --error=experiments/loo_encoding/slurm/logs/03_noise_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
##SBATCH --partition=generic
##SBATCH --account=CHANGE_ME
# Required env: RUN_ROOT
# Uses all-data Protocol A folds (no cleanliness filter). Can run in parallel
# with encoding once prepare has written folds_index.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export REPO_ROOT
export PIPELINE_CONFIG="${PIPELINE_CONFIG:-experiments/loo_encoding/slurm/protocol_A_full.yaml}"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

: "${RUN_ROOT:?Set RUN_ROOT}"
RUN_ROOT_PATH="${RUN_ROOT}"
if [[ "${RUN_ROOT_PATH}" != /* ]]; then
  RUN_ROOT_PATH="${REPO_ROOT}/${RUN_ROOT_PATH}"
fi
MANIFEST="$(load_manifest "${RUN_ROOT_PATH}")"

read -r FOLDS_INDEX OUT_DIR WIN_Z WIN_R < <(
  "${PYTHON}" - <<PY
import yaml
from pathlib import Path
man = yaml.safe_load(Path("${MANIFEST}").read_text())
cfg = yaml.safe_load(Path("${REPO_ROOT}/${PIPELINE_CONFIG}").read_text())
print(
    man["noise_corr_folds_index"],
    man["noise_corr_dir"],
    cfg["windows"]["zscore"],
    cfg["windows"]["raw"],
)
PY
)

echo "Noise corr folds_index=${FOLDS_INDEX}"
"${PYTHON}" experiments/loo_encoding/compute_fold_noise_corr_odd_even.py \
  --folds-index "${FOLDS_INDEX}" \
  --output-dir "${OUT_DIR}" \
  --window-zscore "${WIN_Z}" \
  --window-raw "${WIN_R}"

echo "Noise corr done -> ${OUT_DIR}"
