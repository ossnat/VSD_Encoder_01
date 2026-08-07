#!/bin/bash
#SBATCH --job-name=looA_fin
#SBATCH --output=experiments/loo_encoding/slurm/logs/02_finalize_%j.out
#SBATCH --error=experiments/loo_encoding/slurm/logs/02_finalize_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
##SBATCH --partition=generic
##SBATCH --account=CHANGE_ME
# Required env: RUN_ROOT
# Finalizes all four leaves (summary + overviews), then pooled encoding maps.
# For raw-only maps in parallel, use 02b_pooled_maps_raw.slurm.
# Resume after TIMEOUT: assemble uses --skip-existing.

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

mapfile -t LEAF_DIRS < <(
  "${PYTHON}" - <<PY
import yaml
from pathlib import Path
man = yaml.safe_load(Path("${MANIFEST}").read_text())
for leaf in man["leaves"]:
    print(leaf["leaf_dir"])
PY
)

FAIL=0
for leaf_dir in "${LEAF_DIRS[@]}"; do
  echo "Finalizing ${leaf_dir} …"
  if ! "${PYTHON}" experiments/loo_encoding/finalize_loo_leaf.py \
      --protocol-dir "${leaf_dir}" \
      --make-overview \
      --overview-per-page "${OVERVIEW_PER_PAGE:-8}"; then
    echo "WARNING: finalize returned non-zero for ${leaf_dir}"
    FAIL=1
  fi
done

echo "Assembling pooled encoding pixel-r maps…"
"${PYTHON}" experiments/loo_encoding/assemble_protocol_A_pooled_maps.py \
  --run-root "${RUN_ROOT_PATH}" \
  --skip-existing

if (( FAIL )); then
  echo "WARNING: one or more leaves incomplete; maps may be partial."
  exit 2
fi
echo "Finalize + pooled maps done."
