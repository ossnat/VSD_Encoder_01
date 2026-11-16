#!/bin/bash
cd "$(dirname "$0")"
for s in v2rho v2none; do $HOME/venv/bin/python loo_lean.py --config configs/schira_encoding_100718_$s.yaml > loo_$s.log 2>&1; done
echo ALLDONE > loo_done.flag
