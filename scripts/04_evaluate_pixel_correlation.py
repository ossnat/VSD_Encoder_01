#!/usr/bin/env python3
"""Evaluate Ridge encoder with pixel-wise test-set correlation heatmaps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.DL_features.schema import model_slug
from src.encoding.ridge import attach_feature_paths
from src.encoding.schema import encoding_pairs_manifest_path, ridge_output_dir
from src.evaluation.pixel_correlation import evaluate_pixel_correlation
from src.evaluation.plotting import (
    plot_condition_mean_originals,
    plot_pixel_correlation_heatmap,
    plot_pixel_mean_maps,
    plot_pixel_r2_heatmap,
)
from src.paths import project_root, resolve_data_path, workspace_root


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(default_path: Path, window_path: Path, ridge_path: Path) -> dict:
    cfg = _load_yaml(default_path)
    cfg.update(_load_yaml(window_path))
    cfg["ridge"] = _load_yaml(ridge_path)
    return cfg


def _portable_path(abs_path: Path, repo: Path) -> str:
    ws = workspace_root(repo)
    resolved = abs_path.resolve()
    try:
        return str(resolved.relative_to(ws))
    except ValueError:
        return str(resolved)


def evaluate_pixel_correlation_run(
    cfg: dict,
    *,
    model_cfg_path: Path,
    repo: Path | None = None,
    split: str = "test",
) -> dict:
    repo = repo or project_root()
    monkey = cfg["monkey"]
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])
    window_id = cfg.get("window_id") or f"win_{start_frame:04d}_{end_frame:04d}"
    avg_method = cfg.get("avg_method", "mean")

    model_cfg = _load_yaml(model_cfg_path)
    backbone_name = model_cfg["name"]
    feature_layer = model_cfg.get("feature_layer", "layer3")
    model_name = model_slug(model_cfg)

    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        monkey,
        window_id,
    )
    if not pairs_path.exists():
        raise FileNotFoundError(f"Encoding pairs manifest not found: {pairs_path}")

    model_path = (
        ridge_output_dir(
            resolve_data_path(cfg["paths"]["ridge_encode_root"], repo),
            monkey,
            window_id,
            model_name,
            feature_layer,
        )
        / "model.joblib"
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Ridge model not found: {model_path}. Run stage 03 first."
        )

    pairs = pd.read_parquet(pairs_path)
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    if pairs.empty:
        raise RuntimeError("No complete encoding pairs with nc + stimulus on disk")

    features_root = resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo)
    pairs = attach_feature_paths(
        pairs,
        features_root=features_root,
        monkey=monkey,
        model_slug=model_name,
        feature_layer=feature_layer,
        repo=repo,
    )

    eval_df = pairs[pairs["split"] == split].copy()
    if eval_df.empty:
        raise RuntimeError(f"No trials with split={split!r} in encoding pairs manifest")

    payload = joblib.load(model_path)
    result = payload["result"]
    result.spatial_size = spatial_size

    (
        corr_map,
        r2_map,
        mean_original,
        mean_reconstruction,
        mean_diff,
        cond_means,
        metrics,
    ) = evaluate_pixel_correlation(
        eval_df,
        result=result,
        repo=repo,
        spatial_size=spatial_size,
        start_frame=start_frame,
        end_frame=end_frame,
        avg_method=avg_method,
    )

    plots_root = repo / cfg["paths"].get("evaluation_plots_root", "plots/evaluation")
    plot_dir = plots_root / monkey / window_id / model_name / feature_layer
    plot_dir.mkdir(parents=True, exist_ok=True)

    corr_plot_path = plot_dir / f"pixel_correlation_{split}.png"
    plot_pixel_correlation_heatmap(
        corr_map,
        corr_plot_path,
        title=(
            f"Pixel correlation ({split}) | "
            f"mean r = {metrics['mean_r']:.3f} | "
            f"T = {metrics['n_test_trials']}"
        ),
    )

    r2_plot_path = plot_dir / f"pixel_r2_{split}.png"
    plot_pixel_r2_heatmap(
        r2_map,
        r2_plot_path,
        title=(
            f"Pixel R² ({split}) | "
            f"mean R² = {metrics['mean_r2']:.3f} | "
            f"T = {metrics['n_test_trials']}"
        ),
    )

    cond_mean_path = plot_dir / f"condition_mean_originals_{split}.png"
    plot_condition_mean_originals(
        cond_means,
        cond_mean_path,
        title=(
            f"Condition-mean originals ({split}) | "
            f"frames [{start_frame}, {end_frame})"
        ),
    )

    mean_maps_path = plot_dir / f"pixel_mean_maps_{split}.png"
    plot_pixel_mean_maps(
        mean_original,
        mean_reconstruction,
        mean_diff,
        mean_maps_path,
        title=(
            f"Trial-mean maps ({split}) | "
            f"RMSE = {metrics['rmse_mean_maps']:.4f} | "
            f"T = {metrics['n_test_trials']}"
        ),
    )

    run_cfg = {
        "monkey": monkey,
        "window_id": window_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "model_slug": model_name,
        "feature_layer": feature_layer,
        "split": split,
        "metrics": metrics,
        "encoding_pairs_manifest": _portable_path(pairs_path, repo),
        "model_path": _portable_path(model_path, repo),
        "plot_paths": {
            "pixel_correlation": str(corr_plot_path.relative_to(repo)),
            "pixel_r2": str(r2_plot_path.relative_to(repo)),
            "pixel_mean_maps": str(mean_maps_path.relative_to(repo)),
            "condition_mean_originals": str(cond_mean_path.relative_to(repo)),
        },
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with (plot_dir / f"pixel_evaluation_{split}.json").open("w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"Monkey: {monkey}")
    print(f"Window: {window_id} [{start_frame}, {end_frame})")
    print(f"Model: {model_name} / {feature_layer}")
    print(f"Split: {split}")
    print(f"Trials: {metrics['n_test_trials']} | conditions: {metrics['n_test_conditions']}")
    print(f"Mean r: {metrics['mean_r']:.4f} | median r: {metrics['median_r']:.4f}")
    print(f"Mean R²: {metrics['mean_r2']:.4f} | median R²: {metrics['median_r2']:.4f}")
    print(
        f"Mean-map RMSE: {metrics['rmse_mean_maps']:.4f} | "
        f"mean |diff|: {metrics['mean_abs_diff']:.4f}"
    )
    print(f"Plots:")
    print(f"  {corr_plot_path.relative_to(repo)}")
    print(f"  {r2_plot_path.relative_to(repo)}")
    print(f"  {mean_maps_path.relative_to(repo)}")
    print(f"  {cond_mean_path.relative_to(repo)}")
    return run_cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/default.yaml",
    )
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument(
        "--ridge-config",
        type=Path,
        default=project_root() / "configs/ridge/default.yaml",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root() / "configs/models/resnet18.yaml",
    )
    parser.add_argument("--monkey", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _merge_config(args.config, args.window, args.ridge_config)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey
    evaluate_pixel_correlation_run(
        cfg,
        model_cfg_path=args.model,
        split=args.split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
