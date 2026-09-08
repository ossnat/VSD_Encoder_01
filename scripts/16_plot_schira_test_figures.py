#!/usr/bin/env python3
"""Flatten-ridge-style test figures for Schira local ridge.

Writes under the model ``plots/`` directory:

  - reconstructions_by_condition_pageXX.png  (orig | recon, test, one/cond)
  - by_condition/{date}__{condition}.png
  - pixel_correlation_test.png / pixel_r2_test.png / pixel_mean_maps_test.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.DL_features.schema import model_slug
from src.encoding.ridge_plotting import (
    plot_reconstruction_grid_pages,
    plot_reconstruction_pair,
    select_one_trial_per_condition,
)
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
from src.schira_encoding.io import (
    build_schira_xy,
    resolve_output_root,
    stimulus_label,
)
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
    p.add_argument("--rows-per-page", type=int, default=12)
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
    eval_df = pairs[pairs["split"].astype(str) == split].copy()
    if eval_df.empty:
        raise RuntimeError(f"No trials with split={split!r}")

    register_root = resolve_output_root(
        merged["paths"]["session_register_root"], repo
    )

    # --- Original vs reconstructed (one trial per condition, prefer this split) ---
    plot_df = select_one_trial_per_condition(eval_df, prefer_split=split)
    x_plot, y_plot = build_schira_xy(
        plot_df,
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
    y_hat_plot = predict_local_ridge(x_plot, result)

    plot_dir = model_dir / "plots"
    by_cond = plot_dir / "by_condition"
    by_cond.mkdir(parents=True, exist_ok=True)

    grid_samples: list[tuple[dict, np.ndarray, np.ndarray]] = []
    for i, row in enumerate(plot_df.itertuples(index=False)):
        meta = {
            "date": str(row.date),
            "condition": str(row.condition),
            "shape_type": str(getattr(row, "shape_type", "")),
            "stimulus_label": stimulus_label(row),
            "trial_global_id": int(row.trial_global_id),
            "split": str(row.split),
            "trial_dataset": str(getattr(row, "trial_dataset", "")),
        }
        original = apply_mask_nan(y_plot[i], result.valid)
        recon = apply_mask_nan(y_hat_plot[i], result.valid)
        cond_key = f"{row.date}__{row.condition}"
        plot_reconstruction_pair(
            meta, original, recon, by_cond / f"{cond_key}.png"
        )
        grid_samples.append((meta, original, recon))

    written = plot_reconstruction_grid_pages(
        grid_samples,
        plot_dir,
        title=(
            f"Schira local ridge reconstructions ({split}) | {slug}/{layer}\n"
            f"set={schira_set} anchor={anchor} · {window_id}"
        ),
        rows_per_page=int(args.rows_per_page),
    )
    print(f"Wrote {len(written)} recon grid page(s) + {len(grid_samples)} pair PNGs")

    # --- Per-pixel r / R² on full split (all test trials) ---
    x_all, y_all = build_schira_xy(
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
    y_hat_all = predict_local_ridge(x_all, result)

    corr_map = pixel_correlation_across_trials(y_all, y_hat_all)
    r2_map = pixel_r2_across_trials(y_all, y_hat_all)
    mean_o = np.nanmean(y_all, axis=0).astype(np.float32)
    mean_r = np.nanmean(y_hat_all, axis=0).astype(np.float32)
    mean_d = (mean_r - mean_o).astype(np.float32)

    eval_cfg = merged.get("evaluation") or {}
    disk = mask_from_eval_cfg(eval_cfg, spatial)
    mask = result.valid.copy()
    if disk is not None:
        mask &= disk

    corr_m = apply_mask_nan(corr_map, mask)
    r2_m = apply_mask_nan(r2_map, mask)
    r_sum = masked_map_summary(corr_map, mask)
    r2_sum = masked_map_summary(r2_map, mask)

    pix_dir = plot_dir / f"pixel_eval_{split}"
    pix_dir.mkdir(parents=True, exist_ok=True)
    underlay = apply_mask_nan(mean_o, mask)

    plot_pixel_correlation_heatmap(
        corr_m,
        pix_dir / f"pixel_correlation_{split}.png",
        title=(
            f"Pixel correlation ({split}) | mean r = {r_sum['mean']:.3f} "
            f"(masked) | T = {len(eval_df)} | {slug}/{layer}"
        ),
        underlay=underlay,
    )
    # Also a clean copy at plots/ root for easy find
    plot_pixel_correlation_heatmap(
        corr_m,
        plot_dir / f"pixel_correlation_{split}.png",
        title=(
            f"Pixel correlation ({split}) | mean r = {r_sum['mean']:.3f} "
            f"(masked) | T = {len(eval_df)} | {slug}/{layer}"
        ),
        underlay=underlay,
    )
    plot_pixel_r2_heatmap(
        r2_m,
        pix_dir / f"pixel_r2_{split}.png",
        title=(
            f"Pixel R² ({split}) | mean = {r2_sum['mean']:.3f} "
            f"(masked) | T = {len(eval_df)} | {slug}/{layer}"
        ),
    )
    plot_pixel_mean_maps(
        apply_mask_nan(mean_o, mask),
        apply_mask_nan(mean_r, mask),
        apply_mask_nan(mean_d, mask),
        pix_dir / f"pixel_mean_maps_{split}.png",
        title=(
            f"Trial-mean maps ({split}) | T = {len(eval_df)} | {slug}/{layer}"
        ),
    )

    np.save(pix_dir / f"pixel_correlation_{split}.npy", corr_map)
    np.save(pix_dir / f"pixel_r2_{split}.npy", r2_map)
    metrics = {
        "split": split,
        "n_trials": int(len(eval_df)),
        "n_conditions_plotted": int(len(grid_samples)),
        "mean_r_masked": r_sum["mean"],
        "median_r_masked": r_sum["median"],
        "mean_r2_masked": r2_sum["mean"],
        "median_r2_masked": r2_sum["median"],
        "n_masked_pixels": int(mask.sum()),
        "model_slug": slug,
        "feature_layer": layer,
    }
    (pix_dir / f"pixel_evaluation_{split}.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Plots: {plot_dir}")


if __name__ == "__main__":
    main()
