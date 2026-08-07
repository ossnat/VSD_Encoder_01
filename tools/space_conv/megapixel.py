"""Block-mean megapixel traces from VSD trials (100×100 → 10×10)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.averaging import (
    NORMALIZATION_BASELINE_ZSCORE,
    NORMALIZATION_NONE,
    baseline_zscore_trial,
    resolve_normalization,
)
from src.data.h5_io import read_trial_by_global_id

from tools.space_conv.discovery import TrialRef

DEFAULT_SPATIAL = (100, 100)
DEFAULT_BLOCK = 10
BASELINE_START = 5
BASELINE_END = 26  # half-open [5, 26)


@dataclass
class MegapixelStack:
    """
    mean / std over time for each megapixel.

    Shapes: mean, std → (n_mega_h, n_mega_w, n_frames)
    """

    mean: np.ndarray
    std: np.ndarray
    mode: str  # "single" | "all"
    n_trials: int
    spatial_size: tuple[int, int]
    block_size: int
    normalization: str

    @property
    def n_frames(self) -> int:
        return int(self.mean.shape[-1])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return int(self.mean.shape[0]), int(self.mean.shape[1])


def apply_normalization(
    trial: np.ndarray,
    normalization: str,
    *,
    baseline_start_frame: int = BASELINE_START,
    baseline_end_frame: int = BASELINE_END,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return trial (n_pixels, n_frames), optionally baseline-zscored."""
    mode = resolve_normalization(normalization)
    if mode == NORMALIZATION_NONE:
        return np.asarray(trial, dtype=np.float32)
    if mode == NORMALIZATION_BASELINE_ZSCORE:
        return baseline_zscore_trial(
            trial,
            baseline_start_frame,
            baseline_end_frame,
            eps=eps,
        )
    raise ValueError(f"Unsupported normalization: {normalization!r} (resolved {mode!r})")


def reshape_trial_hw_t(
    trial: np.ndarray,
    spatial_size: tuple[int, int] = DEFAULT_SPATIAL,
) -> np.ndarray:
    """(n_pixels, T) → (H, W, T)."""
    height, width = spatial_size
    n_pixels, n_frames = trial.shape
    if n_pixels != height * width:
        raise ValueError(
            f"Expected {height * width} pixels, got {n_pixels} for spatial_size={spatial_size}"
        )
    return trial.reshape(height, width, n_frames)


def block_mean_std_single(
    trial_hw_t: np.ndarray,
    *,
    block_size: int = DEFAULT_BLOCK,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Megapixel mean ± std for one trial.

    For each 10×10 block and frame: mean and std **across the 100 pixels**.
    Returns (mean, std) with shape (H/block, W/block, T).
    """
    height, width, n_frames = trial_hw_t.shape
    if height % block_size or width % block_size:
        raise ValueError(
            f"spatial {(height, width)} not divisible by block_size={block_size}"
        )
    n_mh, n_mw = height // block_size, width // block_size
    # (n_mh, block, n_mw, block, T) → pixels axis = block*block
    blocks = trial_hw_t.reshape(n_mh, block_size, n_mw, block_size, n_frames)
    pixels = blocks.transpose(0, 2, 1, 3, 4).reshape(
        n_mh, n_mw, block_size * block_size, n_frames
    )
    # Processed frames can contain NaN mask pixels; ignore them in the block.
    with np.errstate(all="ignore"):
        mean = np.nanmean(pixels, axis=2).astype(np.float32)
        std = np.nanstd(pixels, axis=2, ddof=0).astype(np.float32)
    return mean, std


def block_mean_only(
    trial_hw_t: np.ndarray,
    *,
    block_size: int = DEFAULT_BLOCK,
) -> np.ndarray:
    """Megapixel block-mean only → (n_mh, n_mw, T)."""
    mean, _ = block_mean_std_single(trial_hw_t, block_size=block_size)
    return mean


def build_megapixel_stack(
    h5_path,
    trials: list[TrialRef],
    *,
    normalization: str = NORMALIZATION_NONE,
    spatial_size: tuple[int, int] = DEFAULT_SPATIAL,
    block_size: int = DEFAULT_BLOCK,
    baseline_start_frame: int = BASELINE_START,
    baseline_end_frame: int = BASELINE_END,
) -> MegapixelStack:
    """
    Build megapixel mean±std stack.

    - One trial: std across pixels in each block (per frame).
    - Multiple (ALL): for each frame, mean of per-trial block-means;
      shaded std is **across trials** of the block-mean.
    """
    if not trials:
        raise ValueError("No trials to load")

    mode = resolve_normalization(normalization)

    if len(trials) == 1:
        raw = read_trial_by_global_id(h5_path, trials[0].trial_global_id)
        work = apply_normalization(
            raw,
            mode,
            baseline_start_frame=baseline_start_frame,
            baseline_end_frame=baseline_end_frame,
        )
        hw_t = reshape_trial_hw_t(work, spatial_size)
        mean, std = block_mean_std_single(hw_t, block_size=block_size)
        return MegapixelStack(
            mean=mean,
            std=std,
            mode="single",
            n_trials=1,
            spatial_size=spatial_size,
            block_size=block_size,
            normalization=mode,
        )

    block_means: list[np.ndarray] = []
    for ref in trials:
        raw = read_trial_by_global_id(h5_path, ref.trial_global_id)
        work = apply_normalization(
            raw,
            mode,
            baseline_start_frame=baseline_start_frame,
            baseline_end_frame=baseline_end_frame,
        )
        hw_t = reshape_trial_hw_t(work, spatial_size)
        block_means.append(block_mean_only(hw_t, block_size=block_size))

    stacked = np.stack(block_means, axis=0)  # (n_trials, Mh, Mw, T)
    with np.errstate(all="ignore"):
        mean = np.nanmean(stacked, axis=0).astype(np.float32)
        std = np.nanstd(stacked, axis=0, ddof=0).astype(np.float32)
    return MegapixelStack(
        mean=mean,
        std=std,
        mode="all",
        n_trials=len(trials),
        spatial_size=spatial_size,
        block_size=block_size,
        normalization=mode,
    )
