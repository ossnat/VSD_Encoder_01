#!/bin/bash
#SBATCH --job-name=looA_enc
#SBATCH --output=experiments/loo_encoding/slurm/logs/01_encode_%A_%a.out
#SBATCH --error=experiments/loo_encoding/slurm/logs/01_encode_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
##SBATCH --partition=generic
##SBATCH --account=CHANGE_ME
# Array bounds set at submit time: --array=0-$((N-1))%THROTTLE
# Required env: LEAF_KEY (e.g. zscore_clean) and RUN_ROOT (absolute or repo-rel)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export REPO_ROOT
export PIPELINE_CONFIG="${PIPELINE_CONFIG:-experiments/loo_encoding/slurm/protocol_A_full.yaml}"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

: "${LEAF_KEY:?Set LEAF_KEY (zscore_clean|zscore_all|raw_clean|raw_all)}"
: "${RUN_ROOT:?Set RUN_ROOT to the flat run root}"

RUN_ROOT_PATH="${RUN_ROOT}"
if [[ "${RUN_ROOT_PATH}" != /* ]]; then
  RUN_ROOT_PATH="${REPO_ROOT}/${RUN_ROOT_PATH}"
fi
MANIFEST="$(load_manifest "${RUN_ROOT_PATH}")"

read -r LEAF_DIR WINDOW FOLDS_TXT CLEAN_CSV CLEAN_KEEP < <(
  "${PYTHON}" - <<PY
import yaml
from pathlib import Path
man = yaml.safe_load(Path("${MANIFEST}").read_text())
leaf = next(L for L in man["leaves"] if L["key"] == "${LEAF_KEY}")
keep = " ".join(leaf.get("trial_cleanliness_keep") or [])
csv = leaf.get("trial_cleanliness_csv") or ""
print(leaf["leaf_dir"], leaf["window"], leaf["folds_txt"], csv, keep)
PY
)

FOLDS_FILE="${REPO_ROOT}/${FOLDS_TXT}"
if [[ ! -f "${FOLDS_FILE}" ]]; then
  echo "ERROR: missing folds file ${FOLDS_FILE}" >&2
  exit 1
fi

N_FOLDS="$(grep -cve '^[[:space:]]*$' "${FOLDS_FILE}" || true)"
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
if (( TASK_ID < 0 || TASK_ID >= N_FOLDS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range n_folds=${N_FOLDS}" >&2
  exit 1
fi

FOLD_ID="$(sed -n "$((TASK_ID + 1))p" "${FOLDS_FILE}")"
echo "LEAF_KEY=${LEAF_KEY} TASK=${TASK_ID}/${N_FOLDS} FOLD_ID=${FOLD_ID}"

MODEL="$("${PYTHON}" - <<PY
import yaml
from pathlib import Path
print(yaml.safe_load(Path("${MANIFEST}").read_text())["model"])
PY
)"
FEATURE_LAYER="$("${PYTHON}" - <<PY
import yaml
from pathlib import Path
print(yaml.safe_load(Path("${MANIFEST}").read_text())["feature_layer"])
PY
)"
LOSS_ROI="$("${PYTHON}" - <<PY
import yaml
from pathlib import Path
print(yaml.safe_load(Path("${MANIFEST}").read_text())["loss_roi"])
PY
)"
RUN_ROOT_NAME="$("${PYTHON}" - <<PY
import yaml
from pathlib import Path
print(yaml.safe_load(Path("${MANIFEST}").read_text())["run_root_name"])
PY
)"
RUN_DATE="$("${PYTHON}" - <<PY
import yaml
from pathlib import Path
print(yaml.safe_load(Path("${MANIFEST}").read_text())["run_date"])
PY
)"
HELDOUT="$("${PYTHON}" - <<PY
import yaml
from pathlib import Path
print(yaml.safe_load(Path("${MANIFEST}").read_text())["heldout_list"])
PY
)"

CMD=(
  "${PYTHON}" experiments/loo_encoding/run_loo_encoding.py
  --window "${WINDOW}"
  --model "${MODEL}"
  --feature-layer "${FEATURE_LAYER}"
  --protocol A
  --layout flat
  --run-root "${RUN_ROOT_NAME}"
  --run-date "${RUN_DATE}"
  --loss-roi "${LOSS_ROI}"
  --heldout "${HELDOUT}"
  --fold-id "${FOLD_ID}"
  --array-worker
)

if [[ -n "${CLEAN_CSV}" ]]; then
  # shellcheck disable=SC2206
  KEEP_ARR=(${CLEAN_KEEP})
  CMD+=(--trial-cleanliness-csv "${CLEAN_CSV}" --trial-cleanliness-keep "${KEEP_ARR[@]}")
fi

if [[ "${FORCE:-}" == "1" ]]; then
  CMD+=(--force)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Finished fold ${FOLD_ID} -> ${LEAF_DIR}"
