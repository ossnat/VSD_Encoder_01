"""Per-pixel correlation across test trials for a Schira LOO directory.

For each pixel in the fold mask (LUT.valid ∩ loss-roi), correlate the
predicted values across held-out trials with the true VSD values at that
pixel. Writes a map per fold and one pooled map over all test trials.

Example::

  scripts/py scripts/24_plot_schira_pixel_r.py \\
    --loo-dir experiments/schira_encoding/gandalf/loo/win_0035_0046/\\
vgg16_imagenet/block1_prepool/set-100718__anchor-100718a/\\
global_channel_ridge/protocol_B_noise_ceiling_hull
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.evaluation.mask import apply_mask_nan, masked_map_summary
from src.evaluation.pixel_correlation import pixel_correlation_across_trials
from src.evaluation.plotting import (
    plot_loo_all_shapes_orig_recon,
    plot_pixel_correlation_heatmap,
)
from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import register_mapgeog
from src.schira_encoding.global_channel_model import predict_global_channel_ridge
from src.schira_encoding.io import build_schira_xy, resolve_output_root
from src.schira_encoding.lut import SchiraLUT


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _fold_dirs(loo_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in loo_dir.iterdir()
        if p.is_dir()
        and (p.name.startswith("B__") or p.name.startswith("C__"))
        and (p / "fold_pairs.parquet").is_file()
        and (p / "global_channel_ridge.joblib").is_file()
    )


def main() -> None:
    repo = project_root()
    register_mapgeog()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loo-dir", type=Path, required=True)
    p.add_argument(
        "--config",
        type=Path,
        default=repo / "configs/schira_encoding/default.yaml",
    )
    args = p.parse_args()

    loo_dir = args.loo_dir if args.loo_dir.is_absolute() else repo / args.loo_dir
    if not loo_dir.is_dir():
        raise FileNotFoundError(loo_dir)
    cfg = _load_yaml(args.config)
    params = _load_yaml(loo_dir / "params.yaml") if (loo_dir / "params.yaml").is_file() else {}
    monkey = str(params.get("monkey") or cfg.get("monkey") or "gandalf")
    schira_set = str(params.get("schira_set") or cfg["schira_set"])
    anchor = str(params.get("anchor_session") or cfg["anchor_session"])
    spatial = tuple(int(x) for x in cfg["spatial_size"])
    slug = str(params.get("model_slug") or "vgg16_imagenet")
    layer = str(params.get("feature_layer") or "block1_prepool")
    out_root = resolve_output_root(cfg["paths"]["schira_encoding_root"], repo)
    register_root = resolve_output_root(cfg["paths"]["session_register_root"], repo)
    features_root = resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo)

    fold_dirs = _fold_dirs(loo_dir)
    if not fold_dirs:
        raise RuntimeError(f"No completed folds under {loo_dir}")

    pooled_o: list[np.ndarray] = []
    pooled_r: list[np.ndarray] = []
    pooled_valid: np.ndarray | None = None
    fold_means: list[tuple[str, float]] = []
    for fold_dir in fold_dirs:
        fold_df = pd.read_parquet(fold_dir / "fold_pairs.parquet")
        test_df = fold_df[fold_df["loo_split"] == "test"].copy()
        if test_df.empty:
            continue
        result = joblib.load(fold_dir / "global_channel_ridge.joblib")
        lut_path = fold_dir / "lut.npz"
        lut = SchiraLUT.load(lut_path) if lut_path.is_file() else None
        x, y = build_schira_xy(
            test_df,
            repo=repo,
            out_root=out_root,
            monkey=monkey,
            model_slug=slug,
            feature_layer=layer,
            schira_set=schira_set,
            anchor_session=anchor,
            spatial_size=spatial,
            register_root=register_root,
            lut=lut,
            features_root=features_root,
        )
        y_hat = predict_global_channel_ridge(x, result)
        mask = np.asarray(result.valid, dtype=bool)
        corr = apply_mask_nan(pixel_correlation_across_trials(y, y_hat), mask)
        mean_r = float(masked_map_summary(corr, mask)["mean"])
        fold_means.append((fold_dir.name, mean_r))
        np.save(fold_dir / "pixel_r_test.npy", corr.astype(np.float32))
        plot_pixel_correlation_heatmap(
            corr,
            fold_dir / "pixel_r_test.png",
            title=(
                f"{fold_dir.name} · per-pixel r across test trials\n"
                f"mean={mean_r:.3f} · LUT.valid ∩ loss-roi"
            ),
        )
        print(f"{fold_dir.name}: pixel-r mean={mean_r:.3f}  n={y.shape[0]}", flush=True)
        pooled_o.append(y)
        pooled_r.append(y_hat)
        pooled_valid = mask if pooled_valid is None else (pooled_valid & mask)

    y_all = np.concatenate(pooled_o, axis=0)
    yhat_all = np.concatenate(pooled_r, axis=0)
    pooled = apply_mask_nan(
        pixel_correlation_across_trials(y_all, yhat_all), pooled_valid
    )
    pooled_mean = float(masked_map_summary(pooled, pooled_valid)["mean"])
    overview = loo_dir / "overview"
    overview.mkdir(parents=True, exist_ok=True)
    np.save(overview / "pixel_r_test_pooled.npy", pooled.astype(np.float32))
    plot_path = overview / "pixel_r_test_pooled.png"
    plot_pixel_correlation_heatmap(
        pooled,
        plot_path,
        title=(
            f"Per-pixel r across held-out trials\n"
            f"mean={pooled_mean:.3f} · n={y_all.shape[0]} · LUT.valid ∩ loss-roi"
        ),
    )
    (overview / "pixel_r_test_pooled.json").write_text(
        json.dumps(
            {
                "mean_pixel_r_pooled": pooled_mean,
                "n_trials": int(y_all.shape[0]),
                "n_folds": len(fold_means),
                "per_fold_mean_pixel_r": {k: v for k, v in fold_means},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Pooled pixel-r mean={pooled_mean:.3f}  {plot_path}")

    mean_dirs = [
        p
        for p in fold_dirs
        if (p / "fold_mean_orig.npy").is_file() and (p / "fold_mean_recon.npy").is_file()
    ]
    if mean_dirs:
        valid = pooled_valid
        if valid is None:
            valid = np.load(mean_dirs[0] / "valid.npy").astype(bool)
        grid_path = overview / "all_shapes_orig_recon.png"
        plot_loo_all_shapes_orig_recon(
            mean_dirs,
            valid=valid,
            out_path=grid_path,
            title="Schira LOO · fold-mean orig|recon",
        )
        print(f"Wrote {grid_path}")


if __name__ == "__main__":
    main()
