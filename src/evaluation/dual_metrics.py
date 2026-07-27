"""Dual-mask evaluation helpers: circular disk vs stimulus ROI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.encoding.ridge import pearson_r
from src.evaluation.mask import (
    apply_mask_nan,
    masked_map_summary,
    masked_pearson_r,
)
from src.evaluation.pixel_correlation import (
    pixel_correlation_across_trials,
    pixel_r2_across_trials,
)
from src.evaluation.roi_mask import load_roi_mask
from src.stimuli.identity import attach_stimulus_ids


def mean_trial_spatial_r(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    mask: np.ndarray | None,
) -> float:
    """Mean over trials of spatial Pearson r within an optional mask."""
    rs: list[float] = []
    for i in range(originals.shape[0]):
        if mask is not None:
            rs.append(masked_pearson_r(originals[i], reconstructions[i], mask))
        else:
            rs.append(pearson_r(originals[i], reconstructions[i]))
    arr = np.asarray(rs, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def summarize_corr_under_mask(
    corr_map: np.ndarray,
    r2_map: np.ndarray,
    mask: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float | int]:
    """Mean/median pixel-r and R² inside ``mask``."""
    r_sum = masked_map_summary(corr_map, mask)
    r2_sum = masked_map_summary(r2_map, mask)
    return {
        f"mean_r_{prefix}": r_sum["mean"],
        f"median_r_{prefix}": r_sum["median"],
        f"mean_r2_{prefix}": r2_sum["mean"],
        f"median_r2_{prefix}": r2_sum["median"],
        f"n_pixels_{prefix}": int(mask.sum()),
    }


def dual_mask_metrics(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    *,
    disk_mask: np.ndarray | None,
    roi_mask: np.ndarray | None,
    disk_radius: int | None = None,
) -> dict[str, float | int]:
    """
    Pixel-r across trials + mean trial spatial-r under disk and/or ROI masks.

    Keys:
      - mean_r_disk / mean_r_roi (pixel-wise across trials, then mean over mask)
      - mean_trial_spatial_r_disk / mean_trial_spatial_r_roi
    """
    if originals.shape != reconstructions.shape:
        raise ValueError(
            f"Shape mismatch: {originals.shape} vs {reconstructions.shape}"
        )
    corr_map = pixel_correlation_across_trials(originals, reconstructions)
    r2_map = pixel_r2_across_trials(originals, reconstructions)

    metrics: dict[str, float | int] = {
        "n_trials": int(originals.shape[0]),
        "mean_r_full": float(np.nanmean(corr_map)),
        "mean_r2_full": float(np.nanmean(r2_map)),
        "mean_trial_spatial_r_full": mean_trial_spatial_r(
            originals, reconstructions, None
        ),
    }
    if disk_mask is not None:
        metrics.update(
            summarize_corr_under_mask(corr_map, r2_map, disk_mask, prefix="disk")
        )
        metrics["mean_trial_spatial_r_disk"] = mean_trial_spatial_r(
            originals, reconstructions, disk_mask
        )
        if disk_radius is not None:
            metrics["disk_mask_radius"] = int(disk_radius)
    if roi_mask is not None:
        metrics.update(
            summarize_corr_under_mask(corr_map, r2_map, roi_mask, prefix="roi")
        )
        metrics["mean_trial_spatial_r_roi"] = mean_trial_spatial_r(
            originals, reconstructions, roi_mask
        )
    return metrics


def dual_mask_metrics_for_stimulus(
    eval_df: pd.DataFrame,
    originals: np.ndarray,
    reconstructions: np.ndarray,
    *,
    stimulus_id: str,
    disk_mask: np.ndarray | None,
    repo,
    spatial_size: tuple[int, int] = (100, 100),
    disk_radius: int | None = None,
) -> dict[str, Any]:
    """Dual metrics for one held-out stimulus_id subset."""
    df = attach_stimulus_ids(eval_df.reset_index(drop=True))
    idx = np.where(df["stimulus_id"].to_numpy() == stimulus_id)[0]
    if idx.size == 0:
        raise ValueError(f"No trials for stimulus_id={stimulus_id!r}")
    roi = load_roi_mask(stimulus_id, repo=repo, spatial_size=spatial_size)
    metrics = dual_mask_metrics(
        originals[idx],
        reconstructions[idx],
        disk_mask=disk_mask,
        roi_mask=roi,
        disk_radius=disk_radius,
    )
    metrics["stimulus_id"] = stimulus_id
    metrics["n_conditions"] = int(
        df.iloc[idx].groupby(["date", "condition"]).ngroups
    )
    return metrics


def dual_metrics_by_stimulus(
    eval_df: pd.DataFrame,
    originals: np.ndarray,
    reconstructions: np.ndarray,
    *,
    disk_mask: np.ndarray | None,
    repo,
    spatial_size: tuple[int, int] = (100, 100),
    disk_radius: int | None = None,
    stimulus_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Per-stimulus_id dual disk/ROI metrics for rows in ``eval_df``.

    ``originals`` / ``reconstructions`` must align with ``eval_df`` row order.
    """
    df = attach_stimulus_ids(eval_df.reset_index(drop=True))
    if len(df) != originals.shape[0]:
        raise ValueError("eval_df length must match originals trial axis")
    ids = stimulus_ids or sorted(
        sid for sid in df["stimulus_id"].dropna().unique().tolist()
    )
    rows: list[dict[str, Any]] = []
    for sid in ids:
        idx = np.where(df["stimulus_id"].to_numpy() == sid)[0]
        if idx.size == 0:
            continue
        try:
            roi = load_roi_mask(sid, repo=repo, spatial_size=spatial_size)
        except FileNotFoundError:
            continue
        m = dual_mask_metrics(
            originals[idx],
            reconstructions[idx],
            disk_mask=disk_mask,
            roi_mask=roi,
            disk_radius=disk_radius,
        )
        m["stimulus_id"] = sid
        m["n_conditions"] = int(
            df.iloc[idx].groupby(["date", "condition"]).ngroups
        )
        rows.append(m)
    return pd.DataFrame(rows)


def apply_dual_masks_to_corr_map(
    corr_map: np.ndarray,
    *,
    disk_mask: np.ndarray | None,
    roi_mask: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Return NaN-masked correlation maps for plotting."""
    out: dict[str, np.ndarray] = {"full": corr_map.astype(np.float32, copy=True)}
    if disk_mask is not None:
        out["disk"] = apply_mask_nan(corr_map, disk_mask)
    if roi_mask is not None:
        out["roi"] = apply_mask_nan(corr_map, roi_mask)
    return out


def pixel_r_means_from_corr_map(
    corr_map: np.ndarray,
    *,
    disk_mask: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """
    Mean per-pixel Pearson r inside disk / ROI from an existing corr map.

    Use this when ``corr_map`` was computed across a full eval split (or full
    LOO test fold), not within a single condition / identical-feature subset.
    """
    out: dict[str, float | int] = {}
    if disk_mask is not None:
        s = masked_map_summary(corr_map, disk_mask)
        out["mean_pixel_r_disk"] = s["mean"]
        out["median_pixel_r_disk"] = s["median"]
        out["n_pixels_disk"] = int(disk_mask.sum())
    if roi_mask is not None:
        s = masked_map_summary(corr_map, roi_mask)
        out["mean_pixel_r_roi"] = s["mean"]
        out["median_pixel_r_roi"] = s["median"]
        out["n_pixels_roi"] = int(roi_mask.sum())
    return out


def roi_pixel_r_from_global_corr(
    corr_map: np.ndarray,
    *,
    disk_mask: np.ndarray | None,
    roi_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Apply one global (full-split) pixel-r map under the disk and each ROI.

    Returns one row per ROI stimulus_id with disk/ROI means of the same map.
    """
    rows: list[dict[str, Any]] = []
    disk_mean = (
        masked_map_summary(corr_map, disk_mask)["mean"]
        if disk_mask is not None
        else float("nan")
    )
    for sid, roi in roi_masks.items():
        roi_sum = masked_map_summary(corr_map, roi)
        rows.append(
            {
                "stimulus_id": sid,
                "mean_pixel_r_disk": disk_mean,
                "mean_pixel_r_roi": roi_sum["mean"],
                "n_pixels_roi": int(roi.sum()),
                "n_pixels_disk": int(disk_mask.sum()) if disk_mask is not None else 0,
            }
        )
    return pd.DataFrame(rows)
