#!/usr/bin/env python3
"""Pooled fold-level per-pixel Pearson r maps for LOO protocol A/B runs.

At each pixel, correlates fold-mean original vs fold-mean reconstruction across
fold-level samples (12 for protocol A, 3 for protocol B). Mean r inside the NC
hull ROI should match the corrected pooled metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.encoding.ridge import (
    RidgeEncodeResult,
    build_xy,
    flatten_target_mask,
    predict_maps,
    select_target_pixels,
)
from src.evaluation.loss_roi import NOISE_CEILING_HULL_MASK_RELPATH
from src.evaluation.mask import apply_mask_nan, masked_map_summary
from src.evaluation.pixel_correlation import (
    load_trial_mean_maps,
    pixel_correlation_across_trials,
)
from src.paths import project_root
from src.plotting_colormaps import register_mapgeog

DEFAULT_RUNS = {
    ("A", "zscore"): (
        "experiments/loo_encoding/runs/win_0035_0046_zscore/"
        "resnet18_imagenet/layer3/protocol_A_noise_ceiling_hull__clean_good"
    ),
    ("A", "raw"): (
        "experiments/loo_encoding/runs/win_0035_0046/"
        "resnet18_imagenet/layer3/protocol_A_noise_ceiling_hull__clean_good"
    ),
    ("B", "zscore"): (
        "experiments/loo_encoding/runs/win_0035_0046_zscore/"
        "resnet18_imagenet/layer3/protocol_B_noise_ceiling_hull__clean_good"
    ),
    ("B", "raw"): (
        "experiments/loo_encoding/runs/win_0035_0046/"
        "resnet18_imagenet/layer3/protocol_B_noise_ceiling_hull__clean_good"
    ),
}


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _load_folds_index(protocol_dir: Path) -> dict:
    path = protocol_dir / "folds_index.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing folds_index.yaml under {protocol_dir}")
    return _load_yaml(path)


def _load_hull_mask(repo: Path, spatial_size: tuple[int, int]) -> np.ndarray:
    mask_path = repo / NOISE_CEILING_HULL_MASK_RELPATH
    mask = np.load(mask_path).astype(bool)
    if mask.shape != spatial_size:
        raise ValueError(
            f"Hull mask shape {mask.shape} != spatial_size {spatial_size}"
        )
    return mask


def _fit_ridge_fixed_alphas(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alphas: np.ndarray,
    *,
    standardize_features: bool,
    target_mask: np.ndarray | None,
    spatial_size: tuple[int, int],
) -> RidgeEncodeResult:
    scaler: StandardScaler | None = None
    x_fit = x_train
    if standardize_features:
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_train)

    indices: np.ndarray | None = None
    mask_arr: np.ndarray | None = None
    y_fit = y_train
    if target_mask is not None:
        mask_arr = np.asarray(target_mask, dtype=bool)
        indices = flatten_target_mask(mask_arr, spatial_size)
        y_fit = select_target_pixels(y_train, indices)

    model = Ridge(alpha=alphas, fit_intercept=True)
    model.fit(x_fit, y_fit)
    return RidgeEncodeResult(
        model=model,
        scaler=scaler,
        alpha=alphas,
        spatial_size=spatial_size,
        feature_layer="",
        model_slug="",
        alpha_per_target=True,
        target_mask=mask_arr,
        target_pixel_indices=indices,
    )


def _fold_window_params(fold_dir: Path) -> dict:
    with (fold_dir / "metrics.json").open() as f:
        m = json.load(f)
    return {
        "start_frame": int(m["start_frame"]),
        "end_frame": int(m["end_frame"]),
        "normalization": str(m["normalization"]),
        "baseline_start_frame": int(m["baseline_start_frame"]),
        "baseline_end_frame": int(m["baseline_end_frame"]),
        "baseline_std_eps": float(m["baseline_std_eps"]),
    }


def _fold_mean_maps(
    fold_dir: Path,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    avg_method: str,
    standardize_features: bool,
) -> tuple[np.ndarray, np.ndarray]:
    fold_id = fold_dir.name
    manifest_path = fold_dir / f"{fold_id}__manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    fold_df = pd.read_parquet(manifest_path)
    test_df = fold_df[fold_df["loo_split"] == "test"].reset_index(drop=True)
    train_df = fold_df[fold_df["loo_split"] == "train"].reset_index(drop=True)
    if test_df.empty or train_df.empty:
        raise ValueError(f"{fold_id}: empty train or test split")

    win = _fold_window_params(fold_dir)
    train_mask_path = fold_dir / "train_target_mask.npy"
    train_mask = (
        np.load(train_mask_path).astype(bool)
        if train_mask_path.is_file()
        else None
    )
    alphas = np.load(fold_dir / "alphas_per_target.npy").astype(np.float64)

    x_train, y_train = build_xy(train_df, repo=repo, spatial_size=spatial_size)
    result = _fit_ridge_fixed_alphas(
        x_train,
        y_train,
        alphas,
        standardize_features=standardize_features,
        target_mask=train_mask,
        spatial_size=spatial_size,
    )

    originals = load_trial_mean_maps(
        test_df,
        repo=repo,
        spatial_size=spatial_size,
        avg_method=avg_method,
        **win,
    )
    x_test, _ = build_xy(test_df, repo=repo, spatial_size=spatial_size)
    recons = predict_maps(result, x_test, spatial_size)

    orig_mean = np.nanmean(originals, axis=0).astype(np.float32)
    recon_mean = np.nanmean(recons, axis=0).astype(np.float32)
    return orig_mean, recon_mean


def pooled_fold_pixel_r_map(
    protocol_dir: Path,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    hull_mask: np.ndarray,
    avg_method: str = "mean",
    standardize_features: bool = True,
) -> tuple[np.ndarray, float, int, list[str]]:
    """Return masked corr map, mean r inside hull, n folds, fold ids."""
    folds_index = _load_folds_index(protocol_dir)
    fold_ids = [str(f["fold_id"]) for f in folds_index["folds"]]

    orig_means: list[np.ndarray] = []
    recon_means: list[np.ndarray] = []
    used_ids: list[str] = []
    for fold_id in fold_ids:
        fold_dir = protocol_dir / fold_id
        if not fold_dir.is_dir():
            raise FileNotFoundError(f"Missing fold dir: {fold_dir}")
        orig_mean, recon_mean = _fold_mean_maps(
            fold_dir,
            repo=repo,
            spatial_size=spatial_size,
            avg_method=avg_method,
            standardize_features=standardize_features,
        )
        orig_means.append(orig_mean)
        recon_means.append(recon_mean)
        used_ids.append(fold_id)

    cond_orig = np.stack(orig_means, axis=0)
    cond_recon = np.stack(recon_means, axis=0)
    corr_map = pixel_correlation_across_trials(cond_orig, cond_recon)
    corr_masked = apply_mask_nan(corr_map, hull_mask)
    mean_r = masked_map_summary(corr_map, hull_mask)["mean"]
    return corr_masked, mean_r, len(used_ids), used_ids


def _mean_underlay(
    protocol_dir: Path,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    avg_method: str,
    standardize_features: bool,
) -> np.ndarray:
    """Mean fold-mean original map for underlay."""
    folds_index = _load_folds_index(protocol_dir)
    maps: list[np.ndarray] = []
    for fold in folds_index["folds"]:
        fold_dir = protocol_dir / str(fold["fold_id"])
        orig_mean, _ = _fold_mean_maps(
            fold_dir,
            repo=repo,
            spatial_size=spatial_size,
            avg_method=avg_method,
            standardize_features=standardize_features,
        )
        maps.append(orig_mean)
    return np.nanmean(np.stack(maps, axis=0), axis=0).astype(np.float32)


def plot_corr_panel(
    ax: plt.Axes,
    corr_map: np.ndarray,
    *,
    title: str,
    underlay: np.ndarray | None = None,
    vmin: float = -0.5,
    vmax: float = 0.5,
) -> matplotlib.cm.ScalarMappable:
    register_mapgeog()
    if underlay is not None:
        finite = underlay[np.isfinite(underlay)]
        if finite.size:
            u_lo = float(np.percentile(finite, 1))
            u_hi = float(np.percentile(finite, 99))
        else:
            u_lo, u_hi = 0.0, 1.0
        ax.imshow(underlay, cmap="mapgeog", vmin=u_lo, vmax=u_hi, alpha=0.45)
    im = ax.imshow(
        corr_map,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        alpha=0.88,
    )
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return im


def save_single_map(
    corr_map: np.ndarray,
    *,
    out_path: Path,
    title: str,
    underlay: np.ndarray | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = plot_corr_panel(ax, corr_map, title=title, underlay=underlay)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_grid_2x2(
    panels: list[tuple[str, np.ndarray, float, int, str]],
    *,
    out_path: Path,
    underlays: dict[str, np.ndarray] | None = None,
    suptitle: str = "",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5), layout="constrained")
    im = None
    for ax, (key, corr_map, mean_r, n_folds, window_kind) in zip(
        axes.ravel(), panels
    ):
        underlay = (underlays or {}).get(key)
        title = (
            f"{key} · {window_kind} · n={n_folds} folds\n"
            f"mean r (NC hull) = {mean_r:.3f}"
        )
        im = plot_corr_panel(ax, corr_map, title=title, underlay=underlay)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Pearson r")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/loo_encoding/runs/comparisons"),
    )
    p.add_argument(
        "--ridge-config",
        type=Path,
        default=Path("configs/ridge/default.yaml"),
    )
    p.add_argument(
        "--spatial-size",
        type=int,
        nargs=2,
        default=(100, 100),
    )
    p.add_argument("--avg-method", type=str, default="mean")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = project_root()
    ridge_cfg = _load_yaml(repo / args.ridge_config)
    spatial_size = tuple(int(x) for x in args.spatial_size)
    standardize_features = bool(ridge_cfg.get("standardize_features", True))
    hull_mask = _load_hull_mask(repo, spatial_size)

    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    corr_by_key: dict[str, np.ndarray] = {}
    panel_specs: list[tuple[str, np.ndarray, float, int, str]] = []

    for (protocol, window_kind), rel_path in DEFAULT_RUNS.items():
        protocol_dir = repo / rel_path
        key = f"Protocol {protocol}"
        print(f"Computing {key} · {window_kind} …", flush=True)
        corr_map, mean_r, n_folds, fold_ids = pooled_fold_pixel_r_map(
            protocol_dir,
            repo=repo,
            spatial_size=spatial_size,
            hull_mask=hull_mask,
            avg_method=args.avg_method,
            standardize_features=standardize_features,
        )
        print(f"  n_folds={n_folds}  mean_r_hull={mean_r:.4f}", flush=True)

        stem = f"pooled_fold_pixel_r__protocol_{protocol}__{window_kind}"
        np.save(out_dir / f"{stem}.npy", corr_map.astype(np.float32))

        underlay = _mean_underlay(
            protocol_dir,
            repo=repo,
            spatial_size=spatial_size,
            avg_method=args.avg_method,
            standardize_features=standardize_features,
        )
        title = (
            f"Protocol {protocol} · {window_kind} · n={n_folds} fold samples\n"
            f"Per-pixel r (fold-mean orig vs recon) · NC hull · mean r={mean_r:.3f}"
        )
        png_path = out_dir / f"{stem}.png"
        save_single_map(corr_map, out_path=png_path, title=title, underlay=underlay)

        # Also save under protocol overview/
        overview_dir = protocol_dir / "overview"
        overview_dir.mkdir(parents=True, exist_ok=True)
        overview_path = overview_dir / f"pooled_fold_pixel_r__{window_kind}.png"
        save_single_map(
            corr_map, out_path=overview_path, title=title, underlay=underlay
        )
        np.save(
            overview_dir / f"pooled_fold_pixel_r__{window_kind}.npy",
            corr_map.astype(np.float32),
        )

        corr_by_key[f"{protocol}_{window_kind}"] = corr_map
        panel_specs.append((key, corr_map, mean_r, n_folds, window_kind))
        results.append(
            {
                "protocol": protocol,
                "window_kind": window_kind,
                "n_folds": n_folds,
                "mean_r_hull": mean_r,
                "n_pixels_hull": int(hull_mask.sum()),
                "png": str(png_path.relative_to(repo)),
                "npy": str((out_dir / f"{stem}.npy").relative_to(repo)),
                "fold_ids": fold_ids,
            }
        )

    # 2×2 comparison grid (A-zscore, A-raw, B-zscore, B-raw order)
    grid_order = [
        ("A", "zscore"),
        ("A", "raw"),
        ("B", "zscore"),
        ("B", "raw"),
    ]
    grid_panels: list[tuple[str, np.ndarray, float, int, str]] = []
    grid_underlays: dict[str, np.ndarray] = {}
    for protocol, window_kind in grid_order:
        rel_path = DEFAULT_RUNS[(protocol, window_kind)]
        protocol_dir = repo / rel_path
        rec = next(
            r
            for r in results
            if r["protocol"] == protocol and r["window_kind"] == window_kind
        )
        key = f"Protocol {protocol}"
        grid_panels.append(
            (
                key,
                corr_by_key[f"{protocol}_{window_kind}"],
                float(rec["mean_r_hull"]),
                int(rec["n_folds"]),
                window_kind,
            )
        )
        grid_underlays[key] = _mean_underlay(
            protocol_dir,
            repo=repo,
            spatial_size=spatial_size,
            avg_method=args.avg_method,
            standardize_features=standardize_features,
        )

    grid_path = out_dir / "pooled_fold_pixel_r__2x2_A_B_zscore_raw.png"
    save_grid_2x2(
        grid_panels,
        out_path=grid_path,
        underlays=grid_underlays,
        suptitle=(
            "Pooled fold-level per-pixel r (NC hull)\n"
            "fold-mean orig vs fold-mean recon"
        ),
    )

    summary_path = out_dir / "pooled_fold_pixel_r_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote summary: {summary_path.relative_to(repo)}")
    print(f"Wrote 2×2 grid: {grid_path.relative_to(repo)}")
    for rec in results:
        print(
            f"  Protocol {rec['protocol']} {rec['window_kind']}: "
            f"mean_r={rec['mean_r_hull']:.3f}  → {rec['png']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
