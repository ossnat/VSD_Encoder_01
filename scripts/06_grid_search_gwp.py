#!/usr/bin/env python3
"""Grid-search Gabor GWP hyperparameters on the validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.DL_features.gwp_grid_search import run_grid_search, save_grid_search_results
from src.paths import project_root, resolve_data_path


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(default_path: Path, window_path: Path, ridge_path: Path) -> dict:
    cfg = _load_yaml(default_path)
    cfg.update(_load_yaml(window_path))
    cfg["ridge"] = _load_yaml(ridge_path)
    return cfg


def _plot_top_results(df: pd.DataFrame, objective: str, out_path: Path) -> None:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty or objective not in ok.columns:
        return
    top = ok.head(min(12, len(ok)))
    labels = [
        str(row.get("variant", idx))[:28]
        for idx, row in top.iterrows()
    ]
    vals = top[objective].to_numpy()
    fig, ax = plt.subplots(figsize=(max(8, len(top) * 0.7), 4))
    ax.bar(range(len(top)), vals, color="seagreen")
    ax.set_xticks(range(len(top)), labels, rotation=35, ha="right")
    ax.set_ylabel(objective)
    ax.set_title(f"GWP grid search (top {len(top)} by {objective})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/default.yaml",
    )
    parser.add_argument(
        "--window",
        type=Path,
        default=project_root() / "configs/windows/evoked_32_42.yaml",
    )
    parser.add_argument(
        "--ridge-config",
        type=Path,
        default=project_root() / "configs/ridge/default.yaml",
    )
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=project_root() / "configs/grid_search/gwp.yaml",
    )
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    cfg = _merge_config(args.config, args.window, args.ridge_config)
    grid_cfg = _load_yaml(args.grid_config)

    window_id = cfg.get("window_id") or (
        f"win_{int(cfg['start_frame']):04d}_{int(cfg['end_frame']):04d}"
    )
    base_model_cfg = _load_yaml(repo / grid_cfg["base_model"])
    param_grid = grid_cfg["param_grid"]
    objective = grid_cfg.get("objective", "r_mean_val")

    df = run_grid_search(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        base_model_cfg=base_model_cfg,
        param_grid=param_grid,
        ridge_cfg=cfg["ridge"],
        objective=objective,
        overwrite_features=args.overwrite_features,
    )

    out_root = resolve_data_path(
        grid_cfg.get("output_root", "Data/VSD_Encoder_01/grid_search/gwp"),
        repo,
    )
    out_dir = args.output_dir or (out_root / cfg["monkey"] / window_id)
    csv_path = save_grid_search_results(
        df,
        out_dir=out_dir,
        base_model_cfg=base_model_cfg,
        param_grid=param_grid,
        objective=objective,
    )
    _plot_top_results(df, objective, out_dir / "grid_search_top.png")

    print(f"Grid search complete: {len(df)} combos")
    print(f"Objective: {objective}")
    ok = df[df["status"] == "ok"]
    if not ok.empty:
        best = ok.iloc[0]
        print(
            f"Best: {best.get('variant')} | {objective}={best.get(objective):.4f} | "
            f"r_mean_train={best.get('r_mean_train', float('nan')):.4f}"
        )
    print(f"Results: {csv_path}")
    print(f"Best config: {out_dir / 'best_model.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
