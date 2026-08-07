"""Per-pixel frame averaging and optional baseline z-score."""

from __future__ import annotations

import numpy as np

# Supported ``normalization`` values in window YAML.
NORMALIZATION_NONE = "none"
NORMALIZATION_BASELINE_ZSCORE = "baseline_zscore"
# Alias accepted in configs / docs.
NORMALIZATION_ALIASES = {
    "raw": NORMALIZATION_NONE,
    "zscore_baseline": NORMALIZATION_BASELINE_ZSCORE,
}


def resolve_normalization(mode: str | None) -> str:
    """Map config aliases to the canonical normalization mode."""
    if mode is None:
        return NORMALIZATION_NONE
    key = str(mode).strip().lower()
    return NORMALIZATION_ALIASES.get(key, key)


def baseline_zscore_trial(
    trial: np.ndarray,
    baseline_start_frame: int,
    baseline_end_frame: int,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Per-pixel z-score using baseline frames ``[baseline_start_frame, baseline_end_frame)``.

    Frame indexing is 0-based and half-open (Python slice convention). Example:
    ``baseline_start_frame=5``, ``baseline_end_frame=26`` → frames 5..25 inclusive.

    For each pixel: ``(x - mean) / max(std, eps)`` over the full trial time axis.
    Zero / near-zero std pixels use ``eps`` so training stays finite (not NaN).
    """
    n_pixels, n_frames = trial.shape
    if (
        baseline_start_frame < 0
        or baseline_end_frame > n_frames
        or baseline_start_frame >= baseline_end_frame
    ):
        raise ValueError(
            f"Invalid baseline window [{baseline_start_frame}, {baseline_end_frame}) "
            f"for trial with {n_frames} frames"
        )

    baseline = trial[:, baseline_start_frame:baseline_end_frame].astype(np.float64)
    mean = baseline.mean(axis=1, keepdims=True)
    std = baseline.std(axis=1, keepdims=True, ddof=0)
    std = np.maximum(std, float(eps))
    z = (trial.astype(np.float64) - mean) / std
    return z.astype(np.float32)


def average_frames(
    trial: np.ndarray,
    start_frame: int,
    end_frame: int,
    spatial_size: tuple[int, int],
    method: str = "mean",
    *,
    normalization: str = NORMALIZATION_NONE,
    baseline_start_frame: int = 5,
    baseline_end_frame: int = 26,
    baseline_std_eps: float = 1e-8,
) -> np.ndarray:
    """
    Average frames over [start_frame, end_frame) and reshape to (H, W).

    Parameters
    ----------
    trial : ndarray, shape (n_pixels, n_frames)
    normalization : ``none`` / ``raw`` or ``baseline_zscore`` / ``zscore_baseline``
        When baseline z-score is enabled, each pixel is z-scored using the
        baseline window before the analysis window is averaged.
    """
    if method != "mean":
        raise ValueError(f"Unsupported avg_method: {method!r}")

    n_pixels, n_frames = trial.shape
    height, width = spatial_size
    if n_pixels != height * width:
        raise ValueError(
            f"Expected {height * width} pixels, got {n_pixels} "
            f"for spatial_size={spatial_size}"
        )
    if start_frame < 0 or end_frame > n_frames or start_frame >= end_frame:
        raise ValueError(
            f"Invalid window [{start_frame}, {end_frame}) for trial with "
            f"{n_frames} frames"
        )

    mode = resolve_normalization(normalization)
    work = trial
    if mode == NORMALIZATION_BASELINE_ZSCORE:
        work = baseline_zscore_trial(
            trial,
            baseline_start_frame,
            baseline_end_frame,
            eps=baseline_std_eps,
        )
    elif mode != NORMALIZATION_NONE:
        raise ValueError(
            f"Unsupported normalization: {normalization!r} "
            f"(resolved {mode!r}). Use 'none'/'raw' or "
            f"'baseline_zscore'/'zscore_baseline'."
        )

    window = work[:, start_frame:end_frame]
    averaged = window.mean(axis=1, dtype=np.float64).astype(np.float32)
    return averaged.reshape(height, width)
