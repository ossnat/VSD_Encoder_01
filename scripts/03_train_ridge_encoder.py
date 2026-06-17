#!/usr/bin/env python3
"""Train RidgeCV encoder: stimulus CNN features → averaged VSD maps."""

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
from src.data.averaging import average_frames
from src.data.h5_io import read_trial
from src.encoding.ridge import (
    attach_feature_paths,
    bias_map,
    build_xy,
    fit_ridge_encoder,
    pearson_r,
    predict_maps,
)
from src.encoding.ridge_plotting import (
    plot_bias_map,
    plot_reconstruction_grid,
    plot_reconstruction_pair,
    select_plot_samples,
)
from src.encoding.schema import encoding_pairs_manifest_path, ridge_output_dir
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


def _load_h5_mean_frame(
    row,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    start_frame: int,
    end_frame: int,
    avg_method: str,
) -> np.ndarray:
    h5_path = resolve_data_path(row.target_file, repo)
    trial = read_trial(h5_path, row.trial_dataset)
    return average_frames(
        trial,
        start_frame,
        end_frame,
        spatial_size=spatial_size,
        method=avg_method,
    )


def train_ridge_encoder(
    cfg: dict,
    *,
    model_cfg_path: Path,
    repo: Path | None = None,
) -> dict:
    repo = repo or project_root()
    monkey = cfg["monkey"]
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])
    window_id = cfg.get("window_id") or f"win_{start_frame:04d}_{end_frame:04d}"
    avg_method = cfg.get("avg_method", "mean")
    ridge_cfg = cfg["ridge"]

    model_cfg = _load_yaml(model_cfg_path)
    backbone_name = model_cfg["name"]
    pretrained = bool(model_cfg.get("pretrained", True))
    feature_layer = model_cfg.get("feature_layer", "layer3")
    model_name = model_slug(backbone_name, pretrained)

    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        monkey,
        window_id,
    )
    if not pairs_path.exists():
        raise FileNotFoundError(f"Encoding pairs manifest not found: {pairs_path}")

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

    train_df = pairs[pairs["split"] == "train"].copy()
    eval_df = pairs[pairs["split"] != "train"].copy()
    if train_df.empty:
        raise RuntimeError("No training trials in encoding pairs manifest")

    x_train, y_train = build_xy(train_df, repo=repo, spatial_size=spatial_size)
    alphas = np.asarray(ridge_cfg["alphas"], dtype=np.float64)
    result = fit_ridge_encoder(
        x_train,
        y_train,
        alphas=alphas,
        cv_folds=int(ridge_cfg.get("cv_folds", 5)),
        standardize_features=bool(ridge_cfg.get("standardize_features", True)),
    )
    result.spatial_size = spatial_size
    result.feature_layer = feature_layer
    result.model_slug = model_name

    out_dir = ridge_output_dir(
        resolve_data_path(cfg["paths"]["ridge_encode_root"], repo),
        monkey,
        window_id,
        model_name,
        feature_layer,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"result": result}, out_dir / "model.joblib")

    metrics: dict[str, object] = {
        "alpha": result.alpha,
        "n_train": int(len(train_df)),
        "n_eval": int(len(eval_df)),
        "feature_layer": feature_layer,
        "model_slug": model_name,
        "window_id": window_id,
    }

    split_scores: dict[str, list[float]] = {}
    for split_name, split_df in pairs.groupby("split"):
        if split_df.empty:
            continue
        x_split, y_split = build_xy(split_df, repo=repo, spatial_size=spatial_size)
        preds = predict_maps(result, x_split, spatial_size)
        y_true = y_split.reshape(len(split_df), *spatial_size)
        rs = [
            pearson_r(y_true[i], preds[i])
            for i in range(len(split_df))
        ]
        split_scores[str(split_name)] = rs
        metrics[f"r_mean_{split_name}"] = float(np.nanmean(rs))
        metrics[f"r_median_{split_name}"] = float(np.nanmedian(rs))

    metrics["split_r"] = split_scores

    plots_root = repo / cfg["paths"].get("ridge_plots_root", "plots/ridge_encode")
    plot_dir = plots_root / monkey / window_id / model_name / feature_layer
    plot_dir.mkdir(parents=True, exist_ok=True)

    bias = bias_map(result, spatial_size)
    plot_bias_map(
        bias,
        plot_dir / "bias.png",
        title=f"RidgeCV intercept | {model_name} {feature_layer}",
    )

    n_plot = int(ridge_cfg.get("n_plot_samples", 4))
    prefer_split = str(ridge_cfg.get("plot_prefer_split", "test"))
    plot_rows = select_plot_samples(
        eval_df.to_dict("records") if not eval_df.empty else pairs.to_dict("records"),
        n_samples=n_plot,
        prefer_split=prefer_split,
    )

    grid_samples: list[tuple[dict, np.ndarray, np.ndarray]] = []
    for meta in plot_rows:
        row = pairs.loc[pairs["trial_global_id"] == meta["trial_global_id"]].iloc[0]
        x_row, _ = build_xy(pd.DataFrame([row]), repo=repo, spatial_size=spatial_size)
        recon = predict_maps(result, x_row, spatial_size)[0]
        original = _load_h5_mean_frame(
            row,
            repo=repo,
            spatial_size=spatial_size,
            start_frame=start_frame,
            end_frame=end_frame,
            avg_method=avg_method,
        )
        meta_dict = row.to_dict()
        plot_reconstruction_pair(
            meta_dict,
            original,
            recon,
            plot_dir / f"reconstruction_{int(row.trial_global_id):06d}.png",
        )
        grid_samples.append((meta_dict, original, recon))

    plot_reconstruction_grid(
        grid_samples,
        plot_dir / "reconstructions_grid.png",
        title=f"RidgeCV reconstructions | {window_id}",
    )

    run_cfg = {
        "monkey": monkey,
        "window_id": window_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "model_slug": model_name,
        "feature_layer": feature_layer,
        "alpha": result.alpha,
        "metrics": metrics,
        "encoding_pairs_manifest": _portable_path(pairs_path, repo),
        "model_path": _portable_path(out_dir / "model.joblib", repo),
        "plot_dir": str(plot_dir.relative_to(repo)),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(run_cfg, f, indent=2)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Monkey: {monkey}")
    print(f"Window: {window_id}")
    print(f"Model: {model_name} / {feature_layer}")
    print(f"Selected alpha: {result.alpha}")
    print(f"Train trials: {len(train_df)} | Eval trials: {len(eval_df)}")
    for split_name, rs in split_scores.items():
        print(f"  r ({split_name}): mean={np.nanmean(rs):.3f} median={np.nanmedian(rs):.3f}")
    print(f"Model: {_portable_path(out_dir / 'model.joblib', repo)}")
    print(f"Plots: {plot_dir.relative_to(repo)}")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _merge_config(args.config, args.window, args.ridge_config)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey
    train_ridge_encoder(cfg, model_cfg_path=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
