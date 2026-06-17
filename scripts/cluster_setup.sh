#!/usr/bin/env bash
# One-time environment setup for local machine or cluster login node.
#
# Usage (from repo root or anywhere):
#   bash scripts/cluster_setup.sh
#
# Creates .venv/ in the repo, installs requirements.txt, and installs this
# package in editable mode. SLURM jobs use .venv/bin/python automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "WORKSPACE_ROOT=$(dirname "${REPO_ROOT}")"
echo "Using interpreter: $("${PYTHON}" --version)"

if [[ ! -d ".venv" ]]; then
  "${PYTHON}" -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

echo ""
echo "Setup complete."
echo "  Prepare:   bash scripts/run_prepare_encoding.sh"
echo "  SLURM:     bash scripts/submit_encoding_jobs.sh"
echo "  Docs:      docs/cluster_pipeline.md"
