"""Pixel-wise Pearson correlation across test trials."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.trial_frames import load_h5_mean_frame
from src.encoding.ridge import RidgeEncodeResult, build_xy, predict_maps
from src.evaluation.mask import apply_mask_nan, masked_map_summary


def pixel_correlation_across_trials(
    originals: np.ndarray,
    reconstructions: np.ndarray,
) -> np.ndarray:
    """
    Pearson r at each pixel, correlating across the trial axis.

    Parameters
    ----------
    originals, reconstructions
        Arrays with shape (T, H, W).

    Returns
    -------
    ndarray
        Correlation map with shape (H, W). NaN where variance is zero.
    """
    if originals.shape != reconstructions.shape:
        raise ValueError(
            f"Shape mismatch: originals {originals.shape} vs "
            f"reconstructions {reconstructions.shape}"
        )
    if originals.ndim != 3:
        raise ValueError(f"Expected (T, H, W), got shape {originals.shape}")

    o = originals.astype(np.float64)
    r = reconstructions.astype(np.float64)
    o_c = o - np.nanmean(o, axis=0, keepdims=True)
    r_c = r - np.nanmean(r, axis=0, keepdims=True)
    num = np.nansum(o_c * r_c, axis=0)
    denom = np.sqrt(np.nansum(o_c**2, axis=0) * np.nansum(r_c**2, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / denom
    return corr.astype(np.float32)


def pixel_r2_across_trials(
    originals: np.ndarray,
    reconstructions: np.ndarray,
) -> np.ndarray:
    """
    Coefficient of determination R² at each pixel across the trial axis.

    R² = 1 - SS_res / SS_tot with originals as observed values and
    reconstructions as predictions (fixed per trial / condition).
    Can be negative when predictions are worse than the trial mean.
    """
    if originals.shape != reconstructions.shape:
        raise ValueError(
            f"Shape mismatch: originals {originals.shape} vs "
            f"reconstructions {reconstructions.shape}"
        )
    if originals.ndim != 3:
        raise ValueError(f"Expected (T, H, W), got shape {originals.shape}")

    o = originals.astype(np.float64)
    r = reconstructions.astype(np.float64)
    o_mean = np.nanmean(o, axis=0, keepdims=True)
    ss_tot = np.nansum((o - o_mean) ** 2, axis=0)
    ss_res = np.nansum((o - r) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    return r2.astype(np.float32)


def condition_mean_original_maps(
    test_df: pd.DataFrame,
    originals: np.ndarray,
) -> list[dict[str, object]]:
    """Mean trial-averaged original map per (date, condition)."""
    if len(test_df) != originals.shape[0]:
        raise ValueError("test_df row count must match originals trial axis")

    df = test_df.reset_index(drop=True)
    entries: list[dict[str, object]] = []
    for (date, condition), group in df.groupby(
        ["date", "condition"], sort=True
    ):
        idx = group.index.to_numpy()
        entries.append(
            {
                "date": str(date),
                "condition": str(condition),
                "n_trials": int(len(group)),
                "map": np.nanmean(originals[idx], axis=0).astype(np.float32),
            }
        )
    return entries


def load_trial_mean_maps(
    test_df: pd.DataFrame,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    start_frame: int,
    end_frame: int,
    avg_method: str,
) -> np.ndarray:
    """Stack trial-averaged original maps from H5 with shape (T, H, W)."""
    maps: list[np.ndarray] = []
    for row in test_df.itertuples(index=False):
        maps.append(
            load_h5_mean_frame(
                target_file=str(row.target_file),
                trial_global_id=int(row.trial_global_id),
                repo=repo,
                spatial_size=spatial_size,
                start_frame=start_frame,
                end_frame=end_frame,
                avg_method=avg_method,
            )
        )
    return np.stack(maps, axis=0)


def load_reconstructed_maps(
    test_df: pd.DataFrame,
    *,
    result: RidgeEncodeResult,
    repo: Path,
    spatial_size: tuple[int, int],
) -> np.ndarray:
    """Stack Ridge predictions with shape (T, H, W)."""
    x_test, _ = build_xy(test_df, repo=repo, spatial_size=spatial_size)
    return predict_maps(result, x_test, spatial_size)


def evaluate_pixel_correlation(
    test_df: pd.DataFrame,
    *,
    result: RidgeEncodeResult,
    repo: Path,
    spatial_size: tuple[int, int],
    start_frame: int,
    end_frame: int,
    avg_method: str,
    mask: np.ndarray | None = None,
    mask_radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], dict[str, float | int]]:
    """
    Compute pixel-wise r and R² between trial-mean originals and reconstructions.

    Returns (correlation_map, r2_map, condition_mean_entries, summary_metrics).
    """
    if test_df.empty:
        raise ValueError("No test trials to evaluate")

    eval_df = test_df.reset_index(drop=True)
    originals = load_trial_mean_maps(
        eval_df,
        repo=repo,
        spatial_size=spatial_size,
        start_frame=start_frame,
        end_frame=end_frame,
        avg_method=avg_method,
    )
    reconstructions = load_reconstructed_maps(
        eval_df,
        result=result,
        repo=repo,
        spatial_size=spatial_size,
    )
    corr_map = pixel_correlation_across_trials(originals, reconstructions)
    r2_map = pixel_r2_across_trials(originals, reconstructions)
    mean_original = np.nanmean(originals, axis=0).astype(np.float32)
    mean_reconstruction = np.nanmean(reconstructions, axis=0).astype(np.float32)
    mean_diff = (mean_reconstruction - mean_original).astype(np.float32)
    cond_means = condition_mean_original_maps(eval_df, originals)
    metrics: dict[str, float | int] = {
        "n_test_trials": int(len(eval_df)),
        "n_test_conditions": int(
            eval_df.groupby(["date", "condition"]).ngroups
        ),
        "mean_r": float(np.nanmean(corr_map)),
        "median_r": float(np.nanmedian(corr_map)),
        "mean_r2": float(np.nanmean(r2_map)),
        "median_r2": float(np.nanmedian(r2_map)),
        "frac_finite_pixels": float(np.isfinite(corr_map).mean()),
        "mean_abs_diff": float(np.nanmean(np.abs(mean_diff))),
        "rmse_mean_maps": float(np.sqrt(np.nanmean(mean_diff**2))),
    }
    if mask is not None:
        corr_masked = apply_mask_nan(corr_map, mask)
        r2_masked = apply_mask_nan(r2_map, mask)
        diff_masked = apply_mask_nan(mean_diff, mask)
        r_summary = masked_map_summary(corr_map, mask)
        r2_summary = masked_map_summary(r2_map, mask)
        if mask_radius is not None:
            metrics["mask_radius"] = int(mask_radius)
        metrics["n_masked_pixels"] = int(mask.sum())
        metrics["mean_r_masked"] = r_summary["mean"]
        metrics["median_r_masked"] = r_summary["median"]
        metrics["mean_r2_masked"] = r2_summary["mean"]
        metrics["median_r2_masked"] = r2_summary["median"]
        metrics["mean_abs_diff_masked"] = float(np.nanmean(np.abs(diff_masked)))
        metrics["rmse_mean_maps_masked"] = float(
            np.sqrt(np.nanmean(diff_masked**2))
        )
        corr_map = corr_masked
        r2_map = r2_masked
    return corr_map, r2_map, mean_original, mean_reconstruction, mean_diff, cond_means, metrics
