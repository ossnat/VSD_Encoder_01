"""Shared helpers for loading trial-averaged VSD frames from H5."""

from __future__ import annotations

from pathlib import Path

from src.data.averaging import NORMALIZATION_NONE, average_frames
from src.data.h5_io import read_trial_by_global_id
from src.paths import resolve_data_path


def load_h5_mean_frame(
    *,
    target_file: str,
    trial_global_id: int,
    repo: Path,
    spatial_size: tuple[int, int],
    start_frame: int,
    end_frame: int,
    avg_method: str = "mean",
    normalization: str = NORMALIZATION_NONE,
    baseline_start_frame: int = 2,
    baseline_end_frame: int = 26,
    baseline_std_eps: float = 1e-8,
):
    """
    Load one trial from H5 and return the analysis-window mean map.

    When ``normalization`` is ``baseline_zscore``, per-pixel mean/std are
    estimated on ``[baseline_start_frame, baseline_end_frame)`` (half-open,
    0-indexed) before averaging ``[start_frame, end_frame)``.
    """
    h5_path = resolve_data_path(target_file, repo)
    trial = read_trial_by_global_id(h5_path, int(trial_global_id))
    return average_frames(
        trial,
        start_frame,
        end_frame,
        spatial_size=spatial_size,
        method=avg_method,
        normalization=normalization,
        baseline_start_frame=baseline_start_frame,
        baseline_end_frame=baseline_end_frame,
        baseline_std_eps=baseline_std_eps,
    )
