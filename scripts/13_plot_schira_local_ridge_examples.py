#!/usr/bin/env python3
"""Plot a few true vs predicted VSD maps for a saved Schira local-ridge model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from src.DL_features.schema import model_slug
from src.encoding.ridge_plotting import plot_reconstruction_grid
from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.mask import apply_mask_nan
from src.paths import project_root, resolve_data_path
from src.schira_encoding.io import resolve_output_root, stimulus_label
from src.schira_encoding.model import LocalRidgeResult, predict_local_ridge
from src.schira_encoding.schema import local_ridge_output_dir, warped_feature_map_path


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _load_target(nc_path: Path, spatial_size: tuple[int, int]) -> np.ndarray:
    da = xr.open_dataarray(nc_path)
    image = da.values.astype(np.float32)
    da.close()
    if image.shape != spatial_size:
        raise ValueError(f"Expected {spatial_size}, got {image.shape}")
    return image


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
    p.add_argument("--n-examples", type=int, default=2)
    p.add_argument(
        "--prefer-split",
        type=str,
        default="test",
        help="Prefer this split when picking example trials.",
    )
    p.add_argument("--session", type=str, default=None, help="Default: anchor_session")
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
    session = str(args.session or anchor)
    spatial = tuple(int(x) for x in merged["spatial_size"])
    layer = str(args.feature_layer or model_cfg.get("feature_layer", "block1"))
    slug = model_slug(model_cfg)

    start = int(merged["start_frame"])
    end = int(merged["end_frame"])
    window_id = merged.get("window_id") or f"win_{start:04d}_{end:04d}"

    out_root = resolve_output_root(merged["paths"]["schira_encoding_root"], repo)
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
    pairs = pairs[pairs["date"].astype(str) == session].copy()
    if pairs.empty:
        raise RuntimeError(f"No trials for session {session}")

    # One trial per condition; prefer requested split.
    prefer = str(args.prefer_split)
    pairs = pairs.sort_values(
        ["condition", "trial_global_id"]
    )
    if (pairs["split"].astype(str) == prefer).any():
        pairs["_prefer"] = (pairs["split"].astype(str) == prefer).astype(int)
        pairs = pairs.sort_values(["condition", "_prefer"], ascending=[True, False])
    picked = pairs.drop_duplicates("condition", keep="first").head(int(args.n_examples))

    samples: list[tuple[dict, np.ndarray, np.ndarray]] = []
    for row in picked.itertuples(index=False):
        feat_path = warped_feature_map_path(
            out_root,
            monkey,
            model_slug=slug,
            feature_layer=layer,
            schira_set=schira_set,
            anchor_session=anchor,
            h5_session=str(row.date),
            condition=str(row.condition),
        )
        x = np.load(feat_path)[None, ...]
        y = _load_target(resolve_data_path(row.nc_path, repo), spatial)
        y_hat = predict_local_ridge(x, result)[0]
        y_m = apply_mask_nan(y, result.valid)
        yhat_m = apply_mask_nan(y_hat, result.valid)
        meta = {
            "date": str(row.date),
            "condition": str(row.condition),
            "shape_type": str(getattr(row, "shape_type", "")),
            "stimulus_label": stimulus_label(row),
            "trial_global_id": int(row.trial_global_id),
            "split": str(row.split),
            "trial_dataset": str(getattr(row, "trial_dataset", "")),
        }
        samples.append((meta, y_m, yhat_m))

    plot_dir = model_dir / "plots"
    out_path = plot_dir / f"recon_examples__{session}.png"
    plot_reconstruction_grid(
        samples,
        out_path,
        title=(
            f"Schira local ridge · {slug}/{layer} · {session}\n"
            f"set={schira_set} anchor={anchor} · {window_id}"
        ),
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
