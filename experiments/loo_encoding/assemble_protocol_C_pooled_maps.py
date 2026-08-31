#!/usr/bin/env python3
"""Pooled fold-level encoding pixel-r maps for Protocol C run roots.

One fold per held-out ``stimulus_id`` (~18–20). Fold means use
**stimulus-level mean evaluation** on Protocol C weights:
  - original = mean VSD over **all** trials with that ``stimulus_id`` (all sessions)
  - predicted = **single** ŷ from the fold's held-out ``(date, condition)``
    feature ``X`` (not averaged across sessions)
Train/W remain the Protocol C fold (stimulus out of train). Stacks those
means and computes pooled per-pixel r / R² (n≈18).

Mirrors ``assemble_protocol_A_pooled_maps.py`` but targets ``protocol_C_*``
leaves (typically raw/all, NChull).

Usage::

  scripts/py experiments/loo_encoding/assemble_protocol_C_pooled_maps.py \\
    --run-root experiments/loo_encoding/runs/2026-08-14_35-46_resnet18_l3

  # Partial encode (only completed fold dirs):
  scripts/py experiments/loo_encoding/assemble_protocol_C_pooled_maps.py \\
    --run-root … --allow-missing-folds
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.loo_encoding.assemble_protocol_A_pooled_maps import (
    _parse_leaf_keys,
    _selection_from_args,
    assemble,
)
from src.paths import project_root

PROTOCOL_C_LEAF_ORDER = (
    ("raw", "all", "protocol_C_raw_NChull_all"),
    ("raw", "clean", "protocol_C_raw_NChull_clean"),
    ("zscore", "all", "protocol_C_zscore_NChull_all"),
    ("zscore", "clean", "protocol_C_zscore_NChull_clean"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Flat run root (…/YYYY-MM-DD_35-46_resnet18_l3)",
    )
    p.add_argument(
        "--ridge-config",
        type=Path,
        default=Path("configs/ridge/default.yaml"),
    )
    p.add_argument("--spatial-size", type=int, nargs=2, default=(100, 100))
    p.add_argument("--avg-method", type=str, default="mean")
    p.add_argument(
        "--leaves",
        type=str,
        default=None,
        help="Comma-separated leaf keys: raw_all, raw_clean, …",
    )
    p.add_argument(
        "--only-raw",
        action="store_true",
        default=True,
        help="Raw only (default for Protocol C)",
    )
    p.add_argument(
        "--only-zscore",
        action="store_true",
        help="Zscore only (not the default Protocol C path)",
    )
    p.add_argument(
        "--window-kind",
        choices=("raw", "zscore"),
        action="append",
        default=None,
    )
    p.add_argument(
        "--cleanliness",
        choices=("clean", "all"),
        action="append",
        default=None,
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--allow-missing-folds",
        action="store_true",
        help="Stack only completed fold dirs (partial runs)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    run_root = args.run_root if args.run_root.is_absolute() else repo / args.run_root

    # Protocol C default: raw × all only (no zscore).
    if args.leaves:
        leaf_keys = _parse_leaf_keys(args.leaves)
        window_kinds = None
        cleanlinesses = None
    elif args.only_zscore or (args.window_kind and "zscore" in args.window_kind):
        leaf_keys = None
        window_kinds = set(args.window_kind or ["zscore"])
        cleanlinesses = set(args.cleanliness or ["all"])
    else:
        leaf_keys = None
        window_kinds = set(args.window_kind or ["raw"])
        cleanlinesses = set(args.cleanliness or ["all"])

    assemble(
        run_root,
        repo=repo,
        ridge_config=args.ridge_config,
        spatial_size=tuple(int(x) for x in args.spatial_size),
        avg_method=args.avg_method,
        leaf_keys=leaf_keys,
        window_kinds=window_kinds,
        cleanlinesses=cleanlinesses,
        skip_existing=bool(args.skip_existing),
        protocol_label="C",
        leaf_order=PROTOCOL_C_LEAF_ORDER,
        allow_missing_folds=bool(args.allow_missing_folds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
