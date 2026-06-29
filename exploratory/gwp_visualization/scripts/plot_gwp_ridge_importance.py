#!/usr/bin/env python3
"""Plot Ridge coefficient importance for Gabor GWP features (interpretability)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.DL_features.schema import model_slug
from src.encoding.schema import ridge_output_dir
from src.evaluation.gwp_importance import compute_gwp_importance, importance_summary_table
from src.evaluation.gwp_importance_plotting import plot_all_gwp_importance
from src.paths import project_root, resolve_data_path


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(default_path: Path, window_path: Path) -> dict:
    cfg = _load_yaml(default_path)
    cfg.update(_load_yaml(window_path))
    return cfg


def _parse_feature_shape(shape_str: str) -> tuple[int, int, int]:
    inner = shape_str.strip().strip("()")
    parts = [int(x.strip()) for x in inner.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected (C,H,W), got {shape_str!r}")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
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
        "--model",
        type=Path,
        default=project_root() / "configs/models/gabor_serre.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: exploratory/gwp_visualization/results/importance/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = project_root()
    cfg = _merge_config(args.config, args.window)
    model_cfg = _load_yaml(args.model)
    monkey = cfg["monkey"]
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    window_id = cfg.get("window_id") or (
        f"win_{int(cfg['start_frame']):04d}_{int(cfg['end_frame']):04d}"
    )

    slug = model_slug(model_cfg)
    feature_layer = str(model_cfg.get("feature_layer", "energy"))
    gwp_cfg = model_cfg.get("gwp", {}) or {}
    n_scales = int(gwp_cfg.get("number_of_scales", 5))
    n_orientations = int(gwp_cfg.get("number_of_directions", 8))

    model_path = (
        ridge_output_dir(
            resolve_data_path(cfg["paths"]["ridge_encode_root"], repo),
            monkey,
            window_id,
            slug,
            feature_layer,
        )
        / "model.joblib"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Ridge model not found: {model_path}. Run stage 03.")

    stim_cfg_path = (
        resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo)
        / monkey
        / slug
        / feature_layer
        / "config.json"
    )
    with stim_cfg_path.open() as f:
        stim_cfg = json.load(f)
    feature_shape = _parse_feature_shape(stim_cfg["feature_shape"])

    payload = joblib.load(model_path)
    result = payload["result"]
    result.spatial_size = spatial_size

    maps = compute_gwp_importance(
        result,
        feature_shape=feature_shape,
        n_scales=n_scales,
        n_orientations=n_orientations,
        min_wavelength=float(gwp_cfg.get("min_wavelength", 3.0)),
        wavelength_factor=float(gwp_cfg.get("wavelength_factor", 2**0.5)),
    )

    out_dir = args.output_dir or (
        repo / "exploratory" / "gwp_visualization" / "results" / "importance" / slug
    )
    plot_paths = plot_all_gwp_importance(maps, out_dir, prefix=window_id)

    table = importance_summary_table(maps)
    csv_path = out_dir / f"{window_id}_importance_by_scale.csv"
    pd.DataFrame(table).to_csv(csv_path, index=False)

    meta = {
        "model_slug": slug,
        "window_id": window_id,
        "feature_shape": list(feature_shape),
        "energy_mode": gwp_cfg.get("energy_mode", "sum_squares"),
        "scale_ranking": table,
        "plots": {k: str(v.relative_to(repo)) for k, v in plot_paths.items()},
    }
    with (out_dir / f"{window_id}_importance_summary.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model: {slug} / {feature_layer}")
    print("Scale ranking (by total |coef| mass):")
    for row in table:
        print(
            f"  #{row['rank']} scale {row['scale']} λ={row['wavelength_px']:.1f}px "
            f"— total={row['total_importance']:.2e}, "
            f"peak ori={row['best_orientation_deg']:.0f}°"
        )
    print(f"Plots under {out_dir.relative_to(repo)}/")
    for name, path in plot_paths.items():
        print(f"  {name}: {path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
