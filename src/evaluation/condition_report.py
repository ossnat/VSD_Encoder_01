"""Condition-specific reconstruction and feature-count helpers for reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.DL_features.backbone import build_feature_extractor
from src.DL_features.schema import model_slug
from src.encoding.ridge import attach_feature_paths, build_xy, predict_maps
from src.encoding.ridge_plotting import _shared_limits
from src.encoding.schema import encoding_pairs_manifest_path, ridge_output_dir
from src.evaluation.mask import apply_mask_nan, masked_pearson_r
from src.paths import resolve_data_path
from src.plotting_colormaps import VSD_CMAP


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def feature_map_n_params(shape: tuple[int, ...]) -> int:
    """Number of Ridge features = product of (C, H, W)."""
    n = 1
    for dim in shape:
        n *= int(dim)
    return int(n)


def mean_trial_spatial_r(
    originals: np.ndarray,
    recons: np.ndarray,
    eval_mask: np.ndarray | None,
) -> float:
    """Mean over trials of spatial Pearson r(original, recon) within the eval mask."""
    rs: list[float] = []
    for i in range(originals.shape[0]):
        if eval_mask is not None:
            rs.append(masked_pearson_r(originals[i], recons[i], eval_mask))
        else:
            a = originals[i].ravel().astype(np.float64)
            b = recons[i].ravel().astype(np.float64)
            if a.std() < 1e-12 or b.std() < 1e-12:
                rs.append(float("nan"))
            else:
                rs.append(float(np.corrcoef(a, b)[0, 1]))
    arr = np.asarray(rs, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def load_feature_shape(
    *,
    repo: Path,
    cfg: dict,
    model_slug_str: str,
    feature_layer: str,
) -> tuple[int, ...]:
    feat_root = resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo)
    maps_dir = feat_root / cfg["monkey"] / model_slug_str / feature_layer / "maps"
    npy_files = sorted(maps_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No feature maps under {maps_dir}")
    arr = np.load(npy_files[0])
    return tuple(int(x) for x in arr.shape)


def build_parameter_table(
    *,
    repo: Path,
    cfg: dict,
    entries: list[tuple[Path, str]],
) -> pd.DataFrame:
    """
    Rows: model/layer with backbone parameter count and Ridge feature count.

    ``entries`` is a list of (model_yaml_path, feature_layer).
    """
    import yaml

    rows: list[dict[str, Any]] = []
    backbone_cache: dict[str, int] = {}
    for model_path, layer in entries:
        with model_path.open() as f:
            model_cfg = yaml.safe_load(f)
        slug = model_slug(model_cfg)
        cache_key = (
            f"{model_cfg.get('type')}:{model_cfg.get('name')}:"
            f"pretrained={model_cfg.get('pretrained', True)}"
        )
        if cache_key not in backbone_cache:
            extractor = build_feature_extractor(
                {**model_cfg, "pretrained": False},
                feature_layer=layer,
            )
            backbone_cache[cache_key] = count_trainable_parameters(extractor)
        try:
            shape = load_feature_shape(
                repo=repo,
                cfg=cfg,
                model_slug_str=slug,
                feature_layer=layer,
            )
            n_feat = feature_map_n_params(shape)
            shape_str = str(shape)
        except FileNotFoundError:
            shape_str = None
            n_feat = None
        rows.append(
            {
                "model_slug": slug,
                "feature_layer": layer,
                "feature_shape": shape_str,
                "n_ridge_features": n_feat,
                "n_backbone_params": backbone_cache[cache_key],
            }
        )
    return pd.DataFrame(rows)


def load_ridge_result(model_joblib: Path):
    payload = joblib.load(model_joblib)
    return payload["result"] if isinstance(payload, dict) and "result" in payload else payload


def predict_condition_split(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    model_slug_str: str,
    feature_layer: str,
    date: str,
    condition: str,
    split: str = "test",
    eval_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Load all split trials for one condition and return originals + reconstructions."""
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        cfg["monkey"],
        window_id,
    )
    pairs = pd.read_parquet(pairs_path)
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    pairs = pairs[
        (pairs["split"] == split)
        & (pairs["date"].astype(str) == date)
        & (pairs["condition"].astype(str) == condition)
    ].copy()
    if pairs.empty:
        raise FileNotFoundError(
            f"No {split} trials for {date}/{condition} in {pairs_path}"
        )

    pairs = attach_feature_paths(
        pairs,
        features_root=resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo),
        monkey=cfg["monkey"],
        model_slug=model_slug_str,
        feature_layer=feature_layer,
        repo=repo,
    )

    model_path = (
        ridge_output_dir(
            resolve_data_path(cfg["paths"]["ridge_encode_root"], repo),
            cfg["monkey"],
            window_id,
            model_slug_str,
            feature_layer,
        )
        / "model.joblib"
    )
    result = load_ridge_result(model_path)
    x, y = build_xy(pairs, repo=repo, spatial_size=spatial_size)
    y_hat = predict_maps(result, x, spatial_size)
    originals = y.reshape(-1, *spatial_size)
    recons = y_hat.reshape(-1, *spatial_size)
    meta0 = pairs.iloc[0].to_dict()
    residual = originals.mean(axis=0) - recons.mean(axis=0)
    return {
        "date": date,
        "condition": condition,
        "split": split,
        "n_trials": int(len(pairs)),
        "stimulus_text": str(meta0.get("stimulus_text", "")),
        "shape_type": str(meta0.get("shape_type", "")),
        "originals": originals,
        "recons": recons,
        "mean_original": originals.mean(axis=0),
        "mean_recon": recons.mean(axis=0),
        "residual": residual,
        "mean_trial_spatial_r": mean_trial_spatial_r(originals, recons, eval_mask),
        "pairs": pairs,
    }


def plot_condition_orig_recon_corr(
    payload: dict[str, Any],
    output_path: Path,
    *,
    title_prefix: str,
    eval_mask: np.ndarray | None = None,
) -> Path:
    """
    Three-panel exemplar figure:

    1. Condition-mean original
    2. Condition-mean reconstruction
    3. Residual (mean orig − mean recon)

    Across-trial per-pixel r within one condition is undefined for stimulus
    encoders; full-split pixel-r heatmaps are plotted separately.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    orig = payload["mean_original"]
    recon = payload["mean_recon"]
    residual = payload["residual"].astype(float)
    if eval_mask is not None:
        residual = apply_mask_nan(residual, eval_mask)
    trial_r = payload.get("mean_trial_spatial_r", float("nan"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    vmin, vmax = _shared_limits([orig, recon])

    im0 = axes[0].imshow(orig, cmap=VSD_CMAP, vmin=vmin, vmax=vmax)
    axes[0].set_title("Condition-mean original")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(recon, cmap=VSD_CMAP, vmin=vmin, vmax=vmax)
    axes[1].set_title(
        f"Condition-mean reconstruction\nmean trial spatial r={trial_r:.3f}"
    )
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    rmax = float(np.nanpercentile(np.abs(residual), 99)) if np.isfinite(residual).any() else 1.0
    rmax = max(rmax, 1e-12)
    im2 = axes[2].imshow(residual, cmap="coolwarm", vmin=-rmax, vmax=rmax)
    axes[2].set_title(f"Residual (orig−recon)\nT={payload['n_trials']}")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{title_prefix} | {payload['date']}/{payload['condition']} "
        f"({payload.get('stimulus_text', '')}) | split={payload['split']}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_split_corr_map(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    model_slug_str: str,
    feature_layer: str,
    split: str = "test",
    eval_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Compute full-split per-pixel r map for one model/layer."""
    from src.evaluation.pixel_correlation import pixel_correlation_across_trials

    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        cfg["monkey"],
        window_id,
    )
    pairs = pd.read_parquet(pairs_path)
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    pairs = pairs[pairs["split"] == split].copy()
    if pairs.empty:
        raise FileNotFoundError(f"No {split} pairs in {pairs_path}")
    pairs = attach_feature_paths(
        pairs,
        features_root=resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo),
        monkey=cfg["monkey"],
        model_slug=model_slug_str,
        feature_layer=feature_layer,
        repo=repo,
    )
    model_path = (
        ridge_output_dir(
            resolve_data_path(cfg["paths"]["ridge_encode_root"], repo),
            cfg["monkey"],
            window_id,
            model_slug_str,
            feature_layer,
        )
        / "model.joblib"
    )
    result = load_ridge_result(model_path)
    x, y = build_xy(pairs, repo=repo, spatial_size=spatial_size)
    y_hat = predict_maps(result, x, spatial_size)
    originals = y.reshape(-1, *spatial_size)
    recons = y_hat.reshape(-1, *spatial_size)
    corr = pixel_correlation_across_trials(originals, recons)
    if eval_mask is not None:
        corr_m = apply_mask_nan(corr.astype(float), eval_mask)
        mean_r = float(np.nanmean(corr_m))
    else:
        mean_r = float(np.nanmean(corr))
    return corr.astype(np.float32), mean_r


def list_split_conditions(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    split: str = "test",
) -> list[tuple[str, str, str, str]]:
    """Return [(date, condition, shape_type, stimulus_text), ...] for the split."""
    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        cfg["monkey"],
        window_id,
    )
    pairs = pd.read_parquet(pairs_path)
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    pairs = pairs[pairs["split"] == split]
    rows: list[tuple[str, str, str, str]] = []
    for (date, condition), sub in pairs.groupby(["date", "condition"], sort=True):
        rows.append(
            (
                str(date),
                str(condition),
                str(sub.iloc[0].get("shape_type", "")),
                str(sub.iloc[0].get("stimulus_text", "")),
            )
        )
    return rows


def plot_parameter_table(df: pd.DataFrame, output_path: Path) -> Path:
    """Render parameter/feature counts as a table figure for the PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.55 * (len(df) + 2))))
    ax.axis("off")
    cols = [
        c
        for c in [
            "model_slug",
            "feature_layer",
            "feature_shape",
            "n_ridge_features",
            "n_backbone_params",
        ]
        if c in df.columns
    ]
    cell = df[cols].astype(str).values.tolist()
    table = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    ax.set_title("Parameter / feature counts", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
