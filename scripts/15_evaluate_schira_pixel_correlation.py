#!/usr/bin/env python3
"""Pixel-wise correlation eval for Schira local ridge (flatten-ridge style)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.DL_features.schema import model_slug
from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.mask import apply_mask_nan, mask_from_eval_cfg, masked_map_summary
from src.evaluation.pixel_correlation import (
    pixel_correlation_across_trials,
    pixel_r2_across_trials,
)
from src.evaluation.plotting import (
    plot_pixel_correlation_heatmap,
    plot_pixel_mean_maps,
    plot_pixel_r2_heatmap,
)
from src.paths import project_root, resolve_data_path
from src.schira_encoding.io import build_schira_xy, resolve_output_root
from src.schira_encoding.model import LocalRidgeResult, predict_local_ridge
from src.schira_encoding.schema import local_ridge_output_dir


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    repo = project_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=repo / "configs/schira_encoding/default.yaml",
    )
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument(
        "--window-config",
        type=Path,
        default=None,
        help="Default: window_config from --config (raw evoked_35_46).",
    )
    p.add_argument("--feature-layer", type=str, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--anchor-only", action="store_true")
    args = p.parse_args()

    cfg = _load_yaml(args.config)
    win_path = args.window_config
    if win_path is None:
        win_path = repo / str(
            cfg.get("window_config", "configs/windows/evoked_35_46.yaml")
        )
    win = _load_yaml(win_path)
    model_cfg = _load_yaml(args.model_config)
    merged = {**cfg, **{k: v for k, v in win.items() if k != "paths"}}
    merged["paths"] = {**(win.get("paths") or {}), **(cfg.get("paths") or {})}

    monkey = str(merged["monkey"])
    schira_set = str(merged["schira_set"])
    anchor = str(merged["anchor_session"])
    spatial = tuple(int(x) for x in merged["spatial_size"])
    layer = str(args.feature_layer or model_cfg.get("feature_layer", "block1"))
    slug = model_slug(model_cfg)
    split = str(args.split)

    out_root = resolve_output_root(merged["paths"]["schira_encoding_root"], repo)
    start = int(merged["start_frame"])
    end = int(merged["end_frame"])
    window_id = merged.get("window_id") or f"win_{start:04d}_{end:04d}"

    model_dir = local_ridge_output_dir(
        out_root,
        monkey,
        window_id,
        slug,
        layer,
        schira_set=schira_set,
        anchor_session=anchor,
    )
    result: LocalRidgeResult = joblib.load(model_dir / "local_ridge.joblib")

    pairs_root = resolve_data_path(merged["paths"]["encoding_pairs_root"], repo)
    pairs = pd.read_parquet(
        encoding_pairs_manifest_path(pairs_root, monkey, window_id)
    )
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    if args.anchor_only:
        pairs = pairs[pairs["date"].astype(str) == anchor].copy()
    eval_df = pairs[pairs["split"].astype(str) == split].copy()
    if eval_df.empty:
        raise RuntimeError(f"No trials with split={split!r}")

    register_root = None
    if not args.anchor_only and "session_register_root" in merged.get("paths", {}):
        register_root = resolve_output_root(
            merged["paths"]["session_register_root"], repo
        )

    x_eval, y_true = build_schira_xy(
        eval_df,
        repo=repo,
        out_root=out_root,
        monkey=monkey,
        model_slug=slug,
        feature_layer=layer,
        schira_set=schira_set,
        anchor_session=anchor,
        spatial_size=spatial,
        register_root=register_root,
    )
    y_hat = predict_local_ridge(x_eval, result)

    corr_map = pixel_correlation_across_trials(y_true, y_hat)
    r2_map = pixel_r2_across_trials(y_true, y_hat)
    mean_original = np.nanmean(y_true, axis=0).astype(np.float32)
    mean_reconstruction = np.nanmean(y_hat, axis=0).astype(np.float32)
    mean_diff = (mean_reconstruction - mean_original).astype(np.float32)

    eval_cfg = merged.get("evaluation") or {}
    disk = mask_from_eval_cfg(eval_cfg, spatial)
    # Model valid already encodes LUT.valid ∩ loss_roi from training.
    mask = result.valid.copy()
    if disk is not None:
        mask &= disk

    metrics: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "n_test_trials": int(len(eval_df)),
        "n_test_conditions": int(eval_df.groupby(["date", "condition"]).ngroups),
        "mean_r": float(np.nanmean(corr_map)),
        "median_r": float(np.nanmedian(corr_map)),
        "mean_r2": float(np.nanmean(r2_map)),
        "median_r2": float(np.nanmedian(r2_map)),
        "rmse_mean_maps": float(np.sqrt(np.nanmean(mean_diff**2))),
        "model_slug": slug,
        "feature_layer": layer,
        "schira_set": schira_set,
        "anchor_session": anchor,
        "window_id": window_id,
    }
    corr_plot = apply_mask_nan(corr_map, mask)
    r2_plot = apply_mask_nan(r2_map, mask)
    r_sum = masked_map_summary(corr_map, mask)
    r2_sum = masked_map_summary(r2_map, mask)
    metrics["mean_r_masked"] = r_sum["mean"]
    metrics["median_r_masked"] = r_sum["median"]
    metrics["mean_r2_masked"] = r2_sum["mean"]
    metrics["median_r2_masked"] = r2_sum["median"]
    metrics["n_masked_pixels"] = int(mask.sum())
    if disk is not None:
        metrics["mask_radius"] = int(eval_cfg.get("mask_radius", 0))

    plot_dir = model_dir / "plots" / f"pixel_eval_{split}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_pixel_correlation_heatmap(
        corr_plot,
        plot_dir / f"pixel_correlation_{split}.png",
        title=(
            f"Schira pixel r ({split}) | mean={metrics['mean_r_masked']:.3f} "
            f"| T={metrics['n_test_trials']} | {slug}/{layer}"
        ),
    )
    plot_pixel_r2_heatmap(
        r2_plot,
        plot_dir / f"pixel_r2_{split}.png",
        title=(
            f"Schira pixel R² ({split}) | mean={metrics['mean_r2_masked']:.3f} "
            f"| T={metrics['n_test_trials']} | {slug}/{layer}"
        ),
    )
    plot_pixel_mean_maps(
        apply_mask_nan(mean_original, mask),
        apply_mask_nan(mean_reconstruction, mask),
        apply_mask_nan(mean_diff, mask),
        plot_dir / f"pixel_mean_maps_{split}.png",
        title=(
            f"Trial-mean maps ({split}) | RMSE={metrics['rmse_mean_maps']:.4f} "
            f"| T={metrics['n_test_trials']}"
        ),
    )

    np.save(plot_dir / f"pixel_correlation_{split}.npy", corr_map)
    np.save(plot_dir / f"pixel_r2_{split}.npy", r2_map)
    (plot_dir / f"pixel_evaluation_{split}.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {plot_dir}")


if __name__ == "__main__":
    main()
