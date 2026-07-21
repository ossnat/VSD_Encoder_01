#!/usr/bin/env python3
"""Regenerate bias / alpha / weight-norm QC heatmaps from a saved Ridge model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.DL_features.schema import model_slug
from src.encoding.ridge import (
    alpha_map,
    attach_feature_paths,
    bias_map,
    build_xy,
    weight_norm_map,
)
from src.encoding.ridge_plotting import (
    plot_alpha_map,
    plot_bias_map,
    plot_weight_norm_map,
)
from src.encoding.schema import encoding_pairs_manifest_path, ridge_output_dir
from src.evaluation.mask import apply_mask_nan, mask_from_eval_cfg
from src.paths import project_root, resolve_data_path


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=project_root() / "configs/default.yaml")
    p.add_argument("--window", type=Path, default=project_root() / "configs/windows/evoked_32_42.yaml")
    p.add_argument("--ridge-config", type=Path, default=project_root() / "configs/ridge/default.yaml")
    p.add_argument("--model", type=Path, default=project_root() / "configs/models/resnet18.yaml")
    p.add_argument("--feature-layer", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = project_root()
    cfg = _load_yaml(args.config)
    cfg.update(_load_yaml(args.window))
    ridge_cfg = _load_yaml(args.ridge_config)
    model_cfg = _load_yaml(args.model)

    monkey = cfg["monkey"]
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])
    window_id = cfg.get("window_id") or f"win_{start_frame:04d}_{end_frame:04d}"
    feature_layer = args.feature_layer or model_cfg.get("feature_layer", "layer3")
    slug = model_slug(model_cfg)
    eval_mask = mask_from_eval_cfg(ridge_cfg.get("evaluation", {}), spatial_size)

    ridge_dir = ridge_output_dir(
        resolve_data_path(cfg["paths"]["ridge_encode_root"], repo),
        monkey,
        window_id,
        slug,
        feature_layer,
    )
    model_path = ridge_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    pairs = pd.read_parquet(
        encoding_pairs_manifest_path(
            resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
            monkey,
            window_id,
        )
    )
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    pairs = attach_feature_paths(
        pairs,
        features_root=resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo),
        monkey=monkey,
        model_slug=slug,
        feature_layer=feature_layer,
        repo=repo,
    )
    train_df = pairs[pairs["split"] == "train"]
    _, y_train = build_xy(train_df, repo=repo, spatial_size=spatial_size)
    underlay = y_train.reshape(len(train_df), *spatial_size).mean(axis=0)

    result = joblib.load(model_path)["result"]
    result.spatial_size = spatial_size
    if not hasattr(result, "alpha_per_target"):
        result.alpha_per_target = np.ndim(result.alpha) > 0 and np.size(result.alpha) > 1

    plot_dir = (
        repo
        / cfg["paths"].get("ridge_plots_root", "plots/ridge_encode")
        / monkey
        / window_id
        / slug
        / feature_layer
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    bias = bias_map(result, spatial_size)
    if eval_mask is not None:
        bias = apply_mask_nan(bias, eval_mask)
    plot_bias_map(
        bias,
        plot_dir / "bias.png",
        title=f"RidgeCV intercept | {slug} {feature_layer}\ngray = train-mean VSD",
        underlay=underlay,
    )

    weights = weight_norm_map(result, spatial_size)
    np.save(ridge_dir / "weight_norm_per_pixel.npy", weights.astype(np.float32))
    weights_plot = weights.copy()
    if eval_mask is not None:
        weights_plot = apply_mask_nan(weights_plot, eval_mask)
    plot_weight_norm_map(
        weights_plot,
        plot_dir / "weight_norm_per_pixel.png",
        title=(
            f"RidgeCV ||w||₂ per pixel | {slug} {feature_layer}\n"
            f"median={float(np.nanmedian(weights_plot)):.3g} | gray = train-mean VSD"
        ),
        underlay=underlay,
    )

    if getattr(result, "alpha_per_target", False):
        alphas = alpha_map(result, spatial_size)
        if eval_mask is not None:
            alphas = apply_mask_nan(alphas, eval_mask)
        plot_alpha_map(
            alphas,
            plot_dir / "alpha_per_pixel.png",
            title=(
                f"RidgeCV α per pixel | {slug} {feature_layer}\n"
                f"median={float(np.nanmedian(alphas)):.3g} | gray = train-mean VSD"
            ),
            underlay=underlay,
        )

    print(f"Wrote plots under {plot_dir.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
