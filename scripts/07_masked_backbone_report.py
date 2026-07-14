#!/usr/bin/env python3
"""Masked backbone comparison: table + side-by-side pixel-r heatmaps with VSD underlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.DL_features.schema import model_slug
from src.encoding.ridge import attach_feature_paths
from src.encoding.schema import encoding_pairs_manifest_path, ridge_output_dir
from src.evaluation.compare import collect_backbone_metrics, comparison_output_dir
from src.evaluation.mask import apply_mask_nan, mask_from_eval_cfg
from src.evaluation.pixel_correlation import (
    load_reconstructed_maps,
    load_trial_mean_maps,
    pixel_correlation_across_trials,
)
from src.evaluation.plotting import plot_backbone_correlation_comparison
from src.paths import project_root, resolve_data_path


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(default_path: Path, window_path: Path, ridge_path: Path) -> dict:
    cfg = _load_yaml(default_path)
    cfg.update(_load_yaml(window_path))
    cfg["ridge"] = _load_yaml(ridge_path)
    return cfg


def _comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Select masked metrics for display."""
    display = df.copy()
    display["label"] = display["model_slug"].map(
        lambda s: "ResNet18" if "resnet18" in s else "GWP (best, γ=0.3)"
    )
    cols = {
        "label": "Model",
        "r_mean_test_masked": "Trial r (test, masked)",
        "eval_mean_r_masked": "Mean pixel r (test, masked)",
        "eval_mean_r2_masked": "Mean pixel R² (test, masked)",
    }
    out = display[list(cols.keys())].rename(columns=cols)
    for col in out.columns[1:]:
        out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    return out


def render_masked_comparison(
    cfg: dict,
    *,
    model_cfg_paths: list[Path],
    ridge_cfg_path: Path,
    repo: Path | None = None,
    split: str = "test",
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    repo = repo or project_root()
    window_id = cfg.get("window_id") or (
        f"win_{int(cfg['start_frame']):04d}_{int(cfg['end_frame']):04d}"
    )
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    ridge_cfg = cfg["ridge"]
    eval_mask = mask_from_eval_cfg(ridge_cfg.get("evaluation"), spatial_size)
    mask_radius = int(ridge_cfg["evaluation"]["mask_radius"])

    df = collect_backbone_metrics(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        model_cfg_paths=model_cfg_paths,
        split=split,
    )

    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        cfg["monkey"],
        window_id,
    )
    pairs = pd.read_parquet(pairs_path)
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    split_df = pairs[pairs["split"] == split].copy()

    originals = load_trial_mean_maps(
        split_df,
        repo=repo,
        spatial_size=spatial_size,
        start_frame=int(cfg["start_frame"]),
        end_frame=int(cfg["end_frame"]),
        avg_method=cfg.get("avg_method", "mean"),
    )
    mean_original = np.nanmean(originals, axis=0).astype(np.float32)

    panels: list[tuple[str, np.ndarray]] = []

    for model_path in model_cfg_paths:
        model_cfg = _load_yaml(model_path)
        slug = model_slug(model_cfg)
        feature_layer = str(model_cfg.get("feature_layer", "layer3"))

        features_root = resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo)
        pairs_with_feat = attach_feature_paths(
            pairs,
            features_root=features_root,
            monkey=cfg["monkey"],
            model_slug=slug,
            feature_layer=feature_layer,
            repo=repo,
        )
        model_split_df = pairs_with_feat[pairs_with_feat["split"] == split].copy()

        model_path_disk = (
            ridge_output_dir(
                resolve_data_path(cfg["paths"]["ridge_encode_root"], repo),
                cfg["monkey"],
                window_id,
                slug,
                feature_layer,
            )
            / "model.joblib"
        )
        result = joblib.load(model_path_disk)["result"]
        result.spatial_size = spatial_size

        reconstructions = load_reconstructed_maps(
            model_split_df,
            result=result,
            repo=repo,
            spatial_size=spatial_size,
        )
        corr_map = pixel_correlation_across_trials(originals, reconstructions)
        if eval_mask is not None:
            corr_map = apply_mask_nan(corr_map, eval_mask)

        short = model_cfg.get("name", slug)
        if model_cfg.get("variant"):
            short = f"{short} (γ={model_cfg.get('gwp', {}).get('gamma', '?')})"
        row = df[df["model_slug"] == slug]
        mr = float(row["eval_mean_r_masked"].iloc[0]) if not row.empty else float("nan")
        panels.append((f"{short}\nmean r = {mr:.3f}", corr_map))

    out_dir = output_dir or (
        comparison_output_dir(repo, cfg, window_id) / "masked_comparison"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    table = _comparison_table(df)
    table_path = out_dir / f"masked_comparison_table_{split}.csv"
    table.to_csv(table_path, index=False)

    heatmap_path = out_dir / f"pixel_correlation_side_by_side_{split}.png"
    plot_backbone_correlation_comparison(
        panels,
        mean_original,
        heatmap_path,
        title=(
            f"Pixel correlation ({split}, masked r={mask_radius}) | "
            f"gray = trial-mean VSD"
        ),
    )

    summary = {
        "window_id": window_id,
        "monkey": cfg["monkey"],
        "split": split,
        "mask_radius": mask_radius,
        "models": df.to_dict(orient="records"),
        "table": table.to_dict(orient="records"),
        "heatmap": str(heatmap_path.relative_to(repo)),
    }
    with (out_dir / f"masked_comparison_{split}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return table, table_path, heatmap_path


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
    parser.add_argument("--model", type=Path, action="append", dest="models", required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    cfg = _merge_config(args.config, args.window, args.ridge_config)
    model_paths = [p if p.is_absolute() else repo / p for p in args.models]

    table, table_path, heatmap_path = render_masked_comparison(
        cfg,
        model_cfg_paths=model_paths,
        ridge_cfg_path=args.ridge_config,
        repo=repo,
        split=args.split,
        output_dir=args.output_dir,
    )

    print(table.to_string(index=False))
    print(f"\nTable: {table_path.relative_to(repo)}")
    print(f"Heatmap: {heatmap_path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
