#!/usr/bin/env python3
"""Plot VSD (H5 mean) vs rendered stimulus image, one panel per condition."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.data.trial_frames import load_h5_mean_frame
from src.encoding.ridge_plotting import select_one_trial_per_condition
from src.encoding.schema import encoding_pairs_manifest_path
from src.encoding.vsd_stimulus_plotting import (
    plot_vsd_vs_stimulus_grid,
    plot_vsd_vs_stimulus_pair,
)
from src.paths import project_root, resolve_data_path


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(default_path: Path, window_path: Path) -> dict:
    cfg = _load_yaml(default_path)
    cfg.update(_load_yaml(window_path))
    return cfg


def build_vsd_vs_stimulus_plots(cfg: dict, *, repo: Path | None = None) -> int:
    repo = repo or project_root()
    monkey = cfg["monkey"]
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])
    window_id = cfg.get("window_id") or f"win_{start_frame:04d}_{end_frame:04d}"
    avg_method = cfg.get("avg_method", "mean")

    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        monkey,
        window_id,
    )
    pairs = pd.read_parquet(pairs_path)
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    plot_df = select_one_trial_per_condition(pairs, prefer_split="test")

    plots_root = repo / cfg["paths"].get(
        "vsd_vs_stimulus_plots_root", "plots/vsd_vs_stimulus"
    )
    plot_dir = plots_root / monkey / window_id
    by_cond_dir = plot_dir / "by_condition"
    by_cond_dir.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[dict, np.ndarray, np.ndarray]] = []
    for _, row in plot_df.iterrows():
        vsd = load_h5_mean_frame(
            target_file=row["target_file"],
            trial_global_id=int(row["trial_global_id"]),
            repo=repo,
            spatial_size=spatial_size,
            start_frame=start_frame,
            end_frame=end_frame,
            avg_method=avg_method,
        )
        stim = np.asarray(
            Image.open(resolve_data_path(row["image_path"], repo)).convert("RGB")
        )
        meta = row.to_dict()
        cond_key = f"{row['date']}__{row['condition']}"
        plot_vsd_vs_stimulus_pair(meta, vsd, stim, by_cond_dir / f"{cond_key}.png")
        samples.append((meta, vsd, stim))

    plot_vsd_vs_stimulus_grid(
        samples,
        plot_dir / "all_conditions_grid.png",
        title=f"VSD vs stimulus | {window_id}",
    )

    print(f"Conditions plotted: {len(samples)}")
    print(f"Output: {plot_dir.relative_to(repo)}")
    return len(samples)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/default.yaml",
    )
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--monkey", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _merge_config(args.config, args.window)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey
    build_vsd_vs_stimulus_plots(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
