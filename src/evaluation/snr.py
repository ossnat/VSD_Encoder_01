"""Trial-to-trial SNR summaries for VSD maps.

Catalog / report figures use a transparent amplitude SNR:

  For a stack of trial maps ``(T, H, W)``:
    snr_pix = |mean_T| / std_T   (sample std, ddof=1)
    snr     = mean of snr_pix over a boolean ROI (default: full FOV)

This is **not** a correlation / split-half reliability metric (see
``experiments/noise_ceiling_roi`` for those). It answers: how large is the
mean evoked pattern relative to trial-to-trial variability, in z-scored
(or raw) map units.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def map_snr_across_trials(
    trial_maps: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    eps: float = 1e-8,
    ddof: int = 1,
) -> dict[str, Any]:
    """
    Pixel-wise |mean| / std across trials, summarized inside ``mask``.

    Parameters
    ----------
    trial_maps
        Array shaped ``(T, H, W)`` with T >= 2 (T == 1 yields NaN SNR).
    mask
        Optional boolean ``(H, W)``. When None, all finite pixels are used.
    eps
        Floor for per-pixel std so near-zero-variance pixels stay finite.
    ddof
        Delta degrees of freedom for ``std`` (1 = sample std).

    Returns
    -------
    dict with:
      - ``snr``: mean of pixel-wise SNR inside the mask
      - ``snr_median``: median of pixel-wise SNR inside the mask
      - ``snr_map``: ``(H, W)`` pixel-wise SNR (NaN outside mask / invalid)
      - ``mean_map``: trial mean ``(H, W)``
      - ``std_map``: trial std ``(H, W)``
      - ``n_trials``: T
      - ``n_pixels``: number of pixels entering the mean SNR
    """
    maps = np.asarray(trial_maps, dtype=np.float64)
    if maps.ndim != 3:
        raise ValueError(f"Expected trial_maps with shape (T, H, W), got {maps.shape}")
    n_trials, height, width = maps.shape
    if mask is None:
        mask_bool = np.ones((height, width), dtype=bool)
    else:
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape != (height, width):
            raise ValueError(
                f"mask shape {mask_bool.shape} != map spatial {(height, width)}"
            )

    mean_map = np.nanmean(maps, axis=0)
    if n_trials < 2:
        std_map = np.full((height, width), np.nan, dtype=np.float64)
        snr_map = np.full((height, width), np.nan, dtype=np.float64)
        return {
            "snr": float("nan"),
            "snr_median": float("nan"),
            "snr_map": snr_map.astype(np.float32),
            "mean_map": mean_map.astype(np.float32),
            "std_map": std_map.astype(np.float32),
            "n_trials": int(n_trials),
            "n_pixels": 0,
        }

    std_map = np.nanstd(maps, axis=0, ddof=int(ddof))
    std_safe = np.maximum(std_map, float(eps))
    snr_map = np.abs(mean_map) / std_safe
    snr_map = np.where(np.isfinite(mean_map) & np.isfinite(std_map), snr_map, np.nan)

    inside = mask_bool & np.isfinite(snr_map)
    values = snr_map[inside]
    n_pixels = int(values.size)
    snr_mean = float(np.mean(values)) if n_pixels else float("nan")
    snr_median = float(np.median(values)) if n_pixels else float("nan")

    snr_map_out = snr_map.astype(np.float32)
    snr_map_out[~mask_bool] = np.nan

    return {
        "snr": snr_mean,
        "snr_median": snr_median,
        "snr_map": snr_map_out,
        "mean_map": mean_map.astype(np.float32),
        "std_map": std_map.astype(np.float32),
        "n_trials": int(n_trials),
        "n_pixels": n_pixels,
    }


def scalar_roi_snr_across_trials(
    trial_maps: np.ndarray,
    mask: np.ndarray,
    *,
    eps: float = 1e-8,
    ddof: int = 1,
) -> dict[str, float | int]:
    """
    ROI-mean amplitude SNR: |mean_T(roi_mean)| / std_T(roi_mean).

    For each trial, take the mean of finite pixels inside ``mask``, then
    compute |μ| / σ across those T scalars. Useful as a compact companion
    statistic to :func:`map_snr_across_trials`.
    """
    maps = np.asarray(trial_maps, dtype=np.float64)
    if maps.ndim != 3:
        raise ValueError(f"Expected trial_maps with shape (T, H, W), got {maps.shape}")
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != maps.shape[1:]:
        raise ValueError(
            f"mask shape {mask_bool.shape} != map spatial {maps.shape[1:]}"
        )

    roi_means = np.array(
        [
            np.nanmean(frame[mask_bool]) if np.isfinite(frame[mask_bool]).any() else np.nan
            for frame in maps
        ],
        dtype=np.float64,
    )
    finite = roi_means[np.isfinite(roi_means)]
    n = int(finite.size)
    if n < 2:
        return {
            "snr": float("nan"),
            "mean": float(finite[0]) if n == 1 else float("nan"),
            "std": float("nan"),
            "n_trials": n,
        }
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=int(ddof)))
    snr = abs(mean) / max(std, float(eps))
    return {"snr": float(snr), "mean": mean, "std": std, "n_trials": n}
