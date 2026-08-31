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

from src.encoding.ridge import (
    attach_feature_paths,
    build_xy,
    fit_ridge_fixed_alphas,
    predict_maps,
)
from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.loss_roi import NOISE_CEILING_HULL_MASK_RELPATH
from src.evaluation.mask import apply_mask_nan, masked_map_summary
from src.evaluation.pixel_correlation import (
    load_trial_mean_maps,
    pixel_correlation_across_trials,
    pixel_r2_across_trials,
)
from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import register_mapgeog
from src.stimuli.identity import attach_stimulus_ids

# Fold-mean evaluation modes (cached under fold_mean_{orig,recon}.npy).
# Protocol C pooled maps use stimulus_level_mean:
#   orig = mean VSD over ALL trials with held-out stimulus_id (all sessions)
#   recon = SINGLE ŷ from the fold's held-out (date, condition) feature X
#           (one map; do not average recons across sessions/trials)
# Train/W still from the Protocol C fold (stimulus out of train).
FOLD_MEAN_EVAL_TEST_SPLIT = "test_split_mean"
FOLD_MEAN_EVAL_STIMULUS = "stimulus_level_mean"
# Cache key distinguishes single-ŷ design from the earlier mean-of-recons variant.
FOLD_MEAN_STIMULUS_RECON_MODE = "single_heldout_x"

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


def _fold_mean_cache_paths(fold_dir: Path) -> tuple[Path, Path, Path]:
    return (
        fold_dir / "fold_mean_orig.npy",
        fold_dir / "fold_mean_recon.npy",
        fold_dir / "fold_mean_meta.yaml",
    )


def _resolve_fold_mean_eval_mode(
    fold_dir: Path,
    *,
    fold_mean_eval: str | None = None,
) -> str:
    """Choose fold-mean aggregation: test-split (A/B) or stimulus-level (C)."""
    if fold_mean_eval in (FOLD_MEAN_EVAL_TEST_SPLIT, FOLD_MEAN_EVAL_STIMULUS):
        return fold_mean_eval
    if fold_dir.name.startswith("C__") or "protocol_C" in fold_dir.parent.name:
        return FOLD_MEAN_EVAL_STIMULUS
    return FOLD_MEAN_EVAL_TEST_SPLIT


def _load_cached_fold_means(
    fold_dir: Path,
    *,
    expected_eval: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    orig_path, recon_path, meta_path = _fold_mean_cache_paths(fold_dir)
    if not (orig_path.is_file() and recon_path.is_file()):
        return None
    if meta_path.is_file():
        meta = _load_yaml(meta_path) or {}
        if str(meta.get("evaluation") or "") != expected_eval:
            return None
        if expected_eval == FOLD_MEAN_EVAL_STIMULUS:
            if str(meta.get("recon_mode") or "") != FOLD_MEAN_STIMULUS_RECON_MODE:
                return None
    elif expected_eval != FOLD_MEAN_EVAL_TEST_SPLIT:
        # Legacy caches without meta are session/test-split means only.
        return None
    return (
        np.load(orig_path).astype(np.float32),
        np.load(recon_path).astype(np.float32),
    )


def _protocol_run_params(fold_dir: Path) -> dict:
    params_path = fold_dir.parent / "params.yaml"
    if params_path.is_file():
        return _load_yaml(params_path) or {}
    return {}


def _load_stimulus_eval_df(
    fold_dir: Path,
    *,
    repo: Path,
) -> tuple[pd.DataFrame, str]:
    """All encoding-pair rows for the fold's held-out stimulus_id (all sessions)."""
    with (fold_dir / "metrics.json").open() as f:
        metrics = json.load(f)
    fold_meta = metrics.get("fold") or {}
    sid = str(fold_meta.get("heldout_stimulus_id") or metrics.get("stimulus_id") or "")
    if not sid:
        raise ValueError(f"{fold_dir.name}: missing heldout_stimulus_id in metrics.json")

    window_id = str(metrics.get("window_id") or "")
    model_slug = str(metrics.get("model_slug") or "")
    feature_layer = str(metrics.get("feature_layer") or "")
    if not window_id or not model_slug or not feature_layer:
        raise ValueError(
            f"{fold_dir.name}: metrics.json missing window_id/model_slug/feature_layer"
        )

    run_params = _protocol_run_params(fold_dir)
    monkey = str(run_params.get("monkey") or "gandalf")
    cfg = _load_yaml(repo / "configs/default.yaml")
    pairs_root = resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo)
    pairs_path = encoding_pairs_manifest_path(pairs_root, monkey, window_id)
    if not pairs_path.is_file():
        raise FileNotFoundError(pairs_path)
    pairs = pd.read_parquet(pairs_path)
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    pairs = attach_stimulus_ids(pairs)
    stim_df = pairs[pairs["stimulus_id"].astype(str) == sid].copy()
    if stim_df.empty:
        raise ValueError(f"{fold_dir.name}: no encoding pairs for stimulus_id={sid}")

    features_root = resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo)
    stim_df = attach_feature_paths(
        stim_df,
        features_root=features_root,
        monkey=monkey,
        model_slug=model_slug,
        feature_layer=feature_layer,
        repo=repo,
    )
    return stim_df.reset_index(drop=True), sid


def _fold_mean_maps(
    fold_dir: Path,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    avg_method: str,
    standardize_features: bool,
    use_cache: bool = True,
    fold_mean_eval: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold-mean orig/recon. Caches tiny .npy unlocks to avoid ridge re-fit.

    Protocol C (``stimulus_level_mean``):
      - orig = mean VSD over **all** trials with the held-out ``stimulus_id``
        (all sessions)
      - recon = **one** ŷ from the fold's held-out ``(date, condition)``
        feature vector ``X`` (not averaged across sessions/trials)
    Weights still come from the Protocol C fold train split (stimulus held
    out) with saved alphas.
    """
    eval_mode = _resolve_fold_mean_eval_mode(fold_dir, fold_mean_eval=fold_mean_eval)
    if use_cache:
        cached = _load_cached_fold_means(fold_dir, expected_eval=eval_mode)
        if cached is not None:
            return cached

    fold_id = fold_dir.name
    manifest_path = fold_dir / f"{fold_id}__manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    fold_df = pd.read_parquet(manifest_path)
    train_df = fold_df[fold_df["loo_split"] == "train"].reset_index(drop=True)
    test_df = fold_df[fold_df["loo_split"] == "test"].reset_index(drop=True)
    if train_df.empty:
        raise ValueError(f"{fold_id}: empty train split")

    win = _fold_window_params(fold_dir)
    train_mask_path = fold_dir / "train_target_mask.npy"
    train_mask = (
        np.load(train_mask_path).astype(bool)
        if train_mask_path.is_file()
        else None
    )
    alphas = np.load(fold_dir / "alphas_per_target.npy").astype(np.float64)

    x_train, y_train = build_xy(train_df, repo=repo, spatial_size=spatial_size)
    result = fit_ridge_fixed_alphas(
        x_train,
        y_train,
        alphas,
        standardize_features=standardize_features,
        target_mask=train_mask,
        spatial_size=spatial_size,
    )

    if eval_mode == FOLD_MEAN_EVAL_STIMULUS:
        stim_df, sid = _load_stimulus_eval_df(fold_dir, repo=repo)
        if test_df.empty:
            raise ValueError(f"{fold_id}: empty test split (need held-out X)")
        # One feature vector: held-out (date, condition) session only.
        pred_df = (
            test_df.drop_duplicates(["date", "condition"])
            .head(1)
            .reset_index(drop=True)
        )
        originals = load_trial_mean_maps(
            stim_df,
            repo=repo,
            spatial_size=spatial_size,
            avg_method=avg_method,
            **win,
        )
        orig_mean = np.nanmean(originals, axis=0).astype(np.float32)
        x_one, _ = build_xy(pred_df, repo=repo, spatial_size=spatial_size)
        recon_mean = predict_maps(result, x_one, spatial_size)[0].astype(np.float32)
        n_orig_trials = int(len(stim_df))
        heldout_date = str(pred_df.iloc[0]["date"])
        heldout_condition = str(pred_df.iloc[0]["condition"])
        eval_note = (
            f"stimulus_id={sid} orig_n={n_orig_trials} all sessions; "
            f"ŷ from {heldout_date}/{heldout_condition} only"
        )
        meta_extra = {
            "recon_mode": FOLD_MEAN_STIMULUS_RECON_MODE,
            "n_orig_trials": n_orig_trials,
            "n_sessions_orig": int(
                stim_df[["date", "condition"]].drop_duplicates().shape[0]
            ),
            "heldout_date": heldout_date,
            "heldout_condition": heldout_condition,
            "note": (
                "Protocol C stimulus-level mean evaluation on fold weights: "
                "orig = mean VSD over all stimulus_id trials (all sessions); "
                "recon = single ŷ from held-out (date, condition) feature X "
                "(not averaged across sessions)."
            ),
        }
    else:
        if test_df.empty:
            raise ValueError(f"{fold_id}: empty test split")
        sid = ""
        originals = load_trial_mean_maps(
            test_df,
            repo=repo,
            spatial_size=spatial_size,
            avg_method=avg_method,
            **win,
        )
        x_eval, _ = build_xy(test_df, repo=repo, spatial_size=spatial_size)
        recons = predict_maps(result, x_eval, spatial_size)
        orig_mean = np.nanmean(originals, axis=0).astype(np.float32)
        recon_mean = np.nanmean(recons, axis=0).astype(np.float32)
        n_orig_trials = int(len(test_df))
        eval_note = f"test-split n_trials={n_orig_trials}"
        meta_extra = {
            "note": "Mean over fold test-split trials only.",
        }

    if use_cache:
        orig_path, recon_path, meta_path = _fold_mean_cache_paths(fold_dir)
        np.save(orig_path, orig_mean)
        np.save(recon_path, recon_mean)
        meta = {
            "evaluation": eval_mode,
            "n_trials": n_orig_trials,
            "heldout_stimulus_id": sid or None,
            **meta_extra,
        }
        with meta_path.open("w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)
        print(
            f"  cached fold means ({eval_mode}) → {fold_id}/ "
            f"({eval_note})",
            flush=True,
        )
    return orig_mean, recon_mean


def _fold_dir_complete(fold_dir: Path) -> bool:
    fold_id = fold_dir.name
    required = (
        fold_dir / f"{fold_id}__manifest.parquet",
        fold_dir / "alphas_per_target.npy",
        fold_dir / "metrics.json",
    )
    return all(p.is_file() for p in required)


def _pooled_fold_mean_stacks(
    protocol_dir: Path,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    avg_method: str,
    standardize_features: bool,
    allow_missing_folds: bool = False,
    fold_mean_eval: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack fold-mean original and reconstruction maps → (n_folds, H, W)."""
    folds_index = _load_folds_index(protocol_dir)
    fold_ids = [str(f["fold_id"]) for f in folds_index["folds"]]

    orig_means: list[np.ndarray] = []
    recon_means: list[np.ndarray] = []
    used_ids: list[str] = []
    for fold_id in fold_ids:
        fold_dir = protocol_dir / fold_id
        if not fold_dir.is_dir():
            if allow_missing_folds:
                print(f"SKIP missing fold dir: {fold_dir}", flush=True)
                continue
            raise FileNotFoundError(f"Missing fold dir: {fold_dir}")
        if allow_missing_folds and not _fold_dir_complete(fold_dir):
            print(f"SKIP incomplete fold: {fold_dir}", flush=True)
            continue
        orig_mean, recon_mean = _fold_mean_maps(
            fold_dir,
            repo=repo,
            spatial_size=spatial_size,
            avg_method=avg_method,
            standardize_features=standardize_features,
            fold_mean_eval=fold_mean_eval,
        )
        orig_means.append(orig_mean)
        recon_means.append(recon_mean)
        used_ids.append(fold_id)

    if not orig_means:
        raise RuntimeError(
            f"No completed folds under {protocol_dir} "
            f"(allow_missing_folds={allow_missing_folds})"
        )
    return np.stack(orig_means, axis=0), np.stack(recon_means, axis=0), used_ids


def pooled_fold_pixel_maps(
    protocol_dir: Path,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    hull_mask: np.ndarray,
    avg_method: str = "mean",
    standardize_features: bool = True,
    allow_missing_folds: bool = False,
    fold_mean_eval: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, int, list[str]]:
    """
    Pooled fold-level per-pixel r and R² maps.

    For Protocol C leaves, fold means default to ``stimulus_level_mean``
    (all sessions of the held-out stimulus_id).

    Returns masked corr map, masked R² map, mean r/R² inside hull, n folds, fold ids.
    """
    cond_orig, cond_recon, used_ids = _pooled_fold_mean_stacks(
        protocol_dir,
        repo=repo,
        spatial_size=spatial_size,
        avg_method=avg_method,
        standardize_features=standardize_features,
        allow_missing_folds=allow_missing_folds,
        fold_mean_eval=fold_mean_eval,
    )
    corr_map = pixel_correlation_across_trials(cond_orig, cond_recon)
    r2_map = pixel_r2_across_trials(cond_orig, cond_recon)
    corr_masked = apply_mask_nan(corr_map, hull_mask)
    r2_masked = apply_mask_nan(r2_map, hull_mask)
    mean_r = masked_map_summary(corr_map, hull_mask)["mean"]
    mean_r2 = masked_map_summary(r2_map, hull_mask)["mean"]
    return corr_masked, r2_masked, mean_r, mean_r2, len(used_ids), used_ids


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
    corr_map, _, mean_r, _, n_folds, fold_ids = pooled_fold_pixel_maps(
        protocol_dir,
        repo=repo,
        spatial_size=spatial_size,
        hull_mask=hull_mask,
        avg_method=avg_method,
        standardize_features=standardize_features,
    )
    return corr_map, mean_r, n_folds, fold_ids


def pooled_fold_pixel_r2_map(
    protocol_dir: Path,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    hull_mask: np.ndarray,
    avg_method: str = "mean",
    standardize_features: bool = True,
) -> tuple[np.ndarray, float, int, list[str]]:
    """Return masked R² map, mean R² inside hull, n folds, fold ids."""
    _, r2_map, _, mean_r2, n_folds, fold_ids = pooled_fold_pixel_maps(
        protocol_dir,
        repo=repo,
        spatial_size=spatial_size,
        hull_mask=hull_mask,
        avg_method=avg_method,
        standardize_features=standardize_features,
    )
    return r2_map, mean_r2, n_folds, fold_ids


def _mean_underlay(
    protocol_dir: Path,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    avg_method: str,
    standardize_features: bool,
    fold_ids: list[str] | None = None,
    allow_missing_folds: bool = False,
    fold_mean_eval: str | None = None,
) -> np.ndarray:
    """Mean fold-mean original map for underlay."""
    if fold_ids is None:
        folds_index = _load_folds_index(protocol_dir)
        fold_ids = [str(f["fold_id"]) for f in folds_index["folds"]]
    maps: list[np.ndarray] = []
    for fold_id in fold_ids:
        fold_dir = protocol_dir / fold_id
        if not fold_dir.is_dir():
            if allow_missing_folds:
                continue
            raise FileNotFoundError(f"Missing fold dir: {fold_dir}")
        orig_mean, _ = _fold_mean_maps(
            fold_dir,
            repo=repo,
            spatial_size=spatial_size,
            avg_method=avg_method,
            standardize_features=standardize_features,
            fold_mean_eval=fold_mean_eval,
        )
        maps.append(orig_mean)
    if not maps:
        raise RuntimeError(f"No underlay maps for {protocol_dir}")
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
    vmin: float = -0.5,
    vmax: float = 0.5,
    cbar_label: str = "Pearson r",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = plot_corr_panel(
        ax, corr_map, title=title, underlay=underlay, vmin=vmin, vmax=vmax
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_single_r2_map(
    r2_map: np.ndarray,
    *,
    out_path: Path,
    title: str,
    underlay: np.ndarray | None = None,
) -> None:
    save_single_map(
        r2_map,
        out_path=out_path,
        title=title,
        underlay=underlay,
        vmin=-1.0,
        vmax=1.0,
        cbar_label="R²",
    )


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
        corr_map, r2_map, mean_r, mean_r2, n_folds, fold_ids = pooled_fold_pixel_maps(
            protocol_dir,
            repo=repo,
            spatial_size=spatial_size,
            hull_mask=hull_mask,
            avg_method=args.avg_method,
            standardize_features=standardize_features,
        )
        print(
            f"  n_folds={n_folds}  mean_r_hull={mean_r:.4f}  mean_r2_hull={mean_r2:.4f}",
            flush=True,
        )

        stem = f"pooled_fold_pixel_r__protocol_{protocol}__{window_kind}"
        r2_stem = f"pooled_fold_pixel_r2__protocol_{protocol}__{window_kind}"
        np.save(out_dir / f"{stem}.npy", corr_map.astype(np.float32))
        np.save(out_dir / f"{r2_stem}.npy", r2_map.astype(np.float32))

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
        r2_title = (
            f"Protocol {protocol} · {window_kind} · n={n_folds} fold samples\n"
            f"Per-pixel R² (fold-mean orig vs recon) · NC hull · mean R²={mean_r2:.3f}"
        )
        save_single_r2_map(
            r2_map,
            out_path=out_dir / f"{r2_stem}.png",
            title=r2_title,
            underlay=underlay,
        )

        # Also save under protocol overview/
        overview_dir = protocol_dir / "overview"
        overview_dir.mkdir(parents=True, exist_ok=True)
        overview_path = overview_dir / f"pooled_fold_pixel_r__{window_kind}.png"
        save_single_map(
            corr_map, out_path=overview_path, title=title, underlay=underlay
        )
        save_single_r2_map(
            r2_map,
            out_path=overview_dir / f"pooled_fold_pixel_r2__{window_kind}.png",
            title=r2_title,
            underlay=underlay,
        )
        np.save(
            overview_dir / f"pooled_fold_pixel_r__{window_kind}.npy",
            corr_map.astype(np.float32),
        )
        np.save(
            overview_dir / f"pooled_fold_pixel_r2__{window_kind}.npy",
            r2_map.astype(np.float32),
        )

        corr_by_key[f"{protocol}_{window_kind}"] = corr_map
        panel_specs.append((key, corr_map, mean_r, n_folds, window_kind))
        results.append(
            {
                "protocol": protocol,
                "window_kind": window_kind,
                "n_folds": n_folds,
                "mean_r_hull": mean_r,
                "mean_r2_hull": mean_r2,
                "n_pixels_hull": int(hull_mask.sum()),
                "png": str(png_path.relative_to(repo)),
                "r2_png": str((out_dir / f"{r2_stem}.png").relative_to(repo)),
                "npy": str((out_dir / f"{stem}.npy").relative_to(repo)),
                "r2_npy": str((out_dir / f"{r2_stem}.npy").relative_to(repo)),
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
