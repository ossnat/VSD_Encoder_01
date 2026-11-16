#!/bin/bash
cd "$(dirname "$0")/../.." && exec $HOME/venv/bin/python schira_dashboard/analyses/fit_triangle.py > schira_dashboard/analyses/fit_triangle.log 2>&1
