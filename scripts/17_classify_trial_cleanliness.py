#!/usr/bin/env python3
"""Classify trial cleanliness via LOO full-FOV metrics.

Writes a standalone QC CSV under ``Data/VSD_Encoder_01/qc/`` keyed by
``trial_global_id``. See ``src/qc/trial_cleanliness.py`` for why we do not
mutate ``all_trials_index_gandalf.csv`` or encoding-pairs parquet directly.

Example::

    python scripts/17_classify_trial_cleanliness.py
    python scripts/17_classify_trial_cleanliness.py --output-csv Data/VSD_Encoder_01/qc/custom.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.paths import project_root, resolve_data_path
from src.qc.trial_cleanliness import (
    add_within_group_scores,
    attach_stimulus_metadata,
    build_load_kwargs,
    classify_trial_cleanliness,
    compute_loo_metrics_for_groups,
    default_cleanliness_csv_path,
    load_trials_for_cleanliness,
)

DEFAULT_CONFIG = project_root() / "configs/default.yaml"
DEFAULT_WINDOW = project_root() / "configs/windows/evoked_35_46_zscore.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(config_path: Path, window_path: Path) -> dict:
    cfg = _load_yaml(config_path)
    cfg.update(_load_yaml(window_path))
    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--window", type=Path, default=DEFAULT_WINDOW)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Override output CSV path (default: Data/VSD_Encoder_01/qc/...).",
    )
    parser.add_argument(
        "--pattern-pct-threshold",
        type=float,
        default=0.15,
        help="Bottom fraction of corr_loo within date×condition → pattern_outlier.",
    )
    parser.add_argument(
        "--pattern-rz-threshold",
        type=float,
        default=2.0,
        help="Robust-z threshold on inverted corr_loo (positive = low corr).",
    )
    parser.add_argument(
        "--pattern-corr-max",
        type=float,
        default=0.40,
        help=(
            "When using the bottom-pct rule, require corr_loo below this absolute "
            "ceiling (rz extremes still flag regardless)."
        ),
    )
    parser.add_argument(
        "--amp-rz-threshold",
        type=float,
        default=2.0,
        help="Robust-z threshold for amplitude metrics (rms, p99|z|, etc.).",
    )
    parser.add_argument(
        "--amp-frac-out-clim-min",
        type=float,
        default=0.05,
        help="Minimum frac_out_sib_clim for amp_edge_outlier (unless strong p99 rz).",
    )
    parser.add_argument(
        "--metrics-cache",
        type=Path,
        default=None,
        help=(
            "Optional path to cache/reuse raw LOO metrics parquet "
            "(skips H5 reloads on re-runs)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    cfg = _merge_config(args.config, args.window)
    monkey = str(cfg["monkey"])
    window_id = str(cfg.get("window_id", "win_0035_0046_zscore"))

    trials = load_trials_for_cleanliness(repo=repo, cfg=cfg)
    print(
        f"Classifying {len(trials)} trials | monkey={monkey} | window={window_id} | "
        f"sessions={trials['date'].nunique()}"
    )

    cache_path = args.metrics_cache
    if cache_path is None:
        cache_path = resolve_data_path(
            f"Data/VSD_Encoder_01/qc/trial_cleanliness_metrics_{monkey}__{window_id}.parquet",
            repo,
        )
    elif not cache_path.is_absolute():
        cache_path = resolve_data_path(str(cache_path), repo)

    if cache_path.is_file():
        print(f"Loading cached LOO metrics: {cache_path}")
        metrics = pd.read_parquet(cache_path)
    else:
        metrics = compute_loo_metrics_for_groups(
            trials,
            load_kw=build_load_kwargs(cfg, repo=repo),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(cache_path, index=False)
        print(f"Cached LOO metrics: {cache_path}")

    metrics = add_within_group_scores(metrics)
    metrics = classify_trial_cleanliness(
        metrics,
        pattern_pct_threshold=float(args.pattern_pct_threshold),
        pattern_rz_threshold=float(args.pattern_rz_threshold),
        pattern_corr_max=float(args.pattern_corr_max),
        amp_rz_threshold=float(args.amp_rz_threshold),
        amp_frac_out_clim_min=float(args.amp_frac_out_clim_min),
    )
    metrics = attach_stimulus_metadata(metrics, repo=repo, cfg=cfg, monkey=monkey)

    output_cols = [
        "trial_global_id",
        "date",
        "condition",
        "stimulus_id",
        "stimulus_text",
        "corr_loo",
        "rms_resid",
        "p99_abs_z",
        "frac_out_sib_clim",
        "frac_abs_gt5",
        "rz_corr_loo",
        "rz_rms_resid",
        "rz_p99_abs_z",
        "rz_frac_out_sib_clim",
        "pct_corr_low",
        "trial_cleanliness",
        "flag_reason",
    ]
    for col in output_cols:
        if col not in metrics.columns:
            metrics[col] = None
    out_df = metrics[output_cols].sort_values(
        ["date", "condition", "trial_global_id"]
    )

    out_path = args.output_csv or default_cleanliness_csv_path(
        monkey=monkey, window_id=window_id, repo=repo
    )
    if not out_path.is_absolute():
        out_path = resolve_data_path(str(out_path), repo)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out_df)} rows)")

    counts = out_df["trial_cleanliness"].value_counts()
    print("\nCounts:")
    for label in ["good", "pattern_outlier", "amp_edge_outlier"]:
        print(f"  {label}: {int(counts.get(label, 0))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
