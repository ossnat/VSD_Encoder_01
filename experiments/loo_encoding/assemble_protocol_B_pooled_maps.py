#!/usr/bin/env python3
"""Pooled fold-level encoding pixel-r maps for Protocol B run roots.

Protocol B: one fold per held-out ``stimulus_id``. Fold means are the
**test-split mean** (all test trials of that stimulus) vs mean
reconstruction — cached as ``fold_mean_{orig,recon}.npy`` during encode.
This stage stacks those means and writes the final per-pixel r / R² maps
(no ridge refit).

Mirrors ``assemble_protocol_A_pooled_maps.py`` but targets ``protocol_B_*``
leaves.

Usage::

  scripts/py experiments/loo_encoding/assemble_protocol_B_pooled_maps.py \\
    --run-root experiments/loo_encoding/runs/2026-08-20_35-46_cornet_s_V2

  # Partial encode (only completed fold dirs):
  scripts/py experiments/loo_encoding/assemble_protocol_B_pooled_maps.py \\
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

PROTOCOL_B_LEAF_ORDER = (
    ("zscore", "clean", "protocol_B_zscore_NChull_clean"),
    ("zscore", "all", "protocol_B_zscore_NChull_all"),
    ("raw", "clean", "protocol_B_raw_NChull_clean"),
    ("raw", "all", "protocol_B_raw_NChull_all"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Flat run root (…/YYYY-MM-DD_35-46_cornet_s_V2)",
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
        help="Comma-separated leaf keys: zscore_all, raw_all, …",
    )
    p.add_argument(
        "--only-raw",
        action="store_true",
        help="Shorthand for --window-kind raw",
    )
    p.add_argument(
        "--only-zscore",
        action="store_true",
        help="Shorthand for --window-kind zscore",
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
    leaf_keys, window_kinds, cleanlinesses = _selection_from_args(args)
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
        protocol_label="B",
        leaf_order=PROTOCOL_B_LEAF_ORDER,
        allow_missing_folds=bool(args.allow_missing_folds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
