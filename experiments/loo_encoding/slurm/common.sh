#!/usr/bin/env bash
# Shared env for Protocol A SLURM jobs.
# Expects REPO_ROOT and optionally PIPELINE_CONFIG.
#
# Safe to source from interactive shells: missing REPO_ROOT (and other errors)
# return 1 instead of exiting the parent shell.

_is_sourced=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  _is_sourced=1
fi

fail() {
  echo "ERROR: $*" >&2
  if [[ $_is_sourced -eq 1 ]]; then
    return 1
  else
    exit 1
  fi
}

# Interactive+sourced: never enable errexit (and clear it) so a failed cmd cannot kill SSH.
if [[ $_is_sourced -eq 1 && $- == *i* ]]; then
  set +e
  set -uo pipefail
else
  set -euo pipefail
fi

if [[ -z "${REPO_ROOT:-}" ]]; then
  fail "REPO_ROOT must be set" || return 1
fi
cd "${REPO_ROOT}" || { fail "cannot cd to REPO_ROOT=${REPO_ROOT}" || return 1; }

export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  fail "${PYTHON} not found. Run: bash scripts/cluster_setup.sh" || return 1
fi

PIPELINE_CONFIG="${PIPELINE_CONFIG:-experiments/loo_encoding/slurm/protocol_A_full.yaml}"
if [[ ! -f "${PIPELINE_CONFIG}" ]]; then
  fail "missing PIPELINE_CONFIG=${PIPELINE_CONFIG}" || return 1
fi

mkdir -p logs experiments/loo_encoding/slurm/logs

# Resolve run root + leaf metadata from pipeline_manifest (after prepare)
# or via --paths-only plan for early stages.
resolve_run_root() {
  "${PYTHON}" - <<'PY'
import os
from pathlib import Path
import yaml
from experiments.loo_encoding.prepare_protocol_A_pipeline import plan_pipeline
from src.paths import project_root

repo = project_root()
cfg_path = Path(os.environ["PIPELINE_CONFIG"])
if not cfg_path.is_absolute():
    cfg_path = repo / cfg_path
cfg = yaml.safe_load(cfg_path.read_text())
run_date = os.environ.get("RUN_DATE") or None
manifest = repo / "experiments/loo_encoding/runs"
# Prefer existing manifest if RUN_ROOT pinned
pinned = os.environ.get("RUN_ROOT")
if pinned:
    root = Path(pinned)
    if not root.is_absolute():
        root = repo / root
    print(root)
else:
    plan = plan_pipeline(cfg, repo=repo, run_date=run_date)
    print(repo / plan["run_root"])
PY
}

load_manifest() {
  local run_root="$1"
  local man="${run_root}/pipeline_manifest.yaml"
  if [[ ! -f "${man}" ]]; then
    fail "missing ${man}. Run prepare stage first." || return 1
  fi
  echo "${man}"
}
