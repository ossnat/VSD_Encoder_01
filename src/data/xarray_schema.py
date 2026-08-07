"""xarray NetCDF schema for averaged trials."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def window_id_from_frames(start_frame: int, end_frame: int) -> str:
    return f"win_{start_frame:04d}_{end_frame:04d}"


def trial_output_path(
    averaged_root: Path,
    monkey: str,
    window_id: str,
    trial_global_id: int,
) -> Path:
    return (
        averaged_root
        / monkey
        / window_id
        / "trials"
        / f"{trial_global_id:06d}.nc"
    )


def build_averaged_dataarray(
    image: np.ndarray,
    *,
    trial_global_id: int,
    monkey: str,
    date: str,
    condition: str,
    condition_code: int,
    trial_index_in_condition: int,
    trial_dataset: str,
    target_file: str,
    window_id: str,
    start_frame: int,
    end_frame: int,
    source_n_frames: int,
    avg_method: str,
    normalization: str,
    split: str,
    baseline_start_frame: int | None = None,
    baseline_end_frame: int | None = None,
    baseline_std_eps: float | None = None,
) -> xr.DataArray:
    height, width = image.shape
    y = np.arange(height, dtype=np.int32)
    x = np.arange(width, dtype=np.int32)

    attrs: dict[str, Any] = {
        "trial_global_id": int(trial_global_id),
        "monkey": monkey,
        "date": date,
        "condition": condition,
        "condition_code": int(condition_code),
        "trial_index_in_condition": int(trial_index_in_condition),
        "trial_dataset": trial_dataset,
        "target_file": target_file,
        "window_id": window_id,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "n_frames_averaged": int(end_frame - start_frame),
        "source_n_frames": int(source_n_frames),
        "avg_method": avg_method,
        "normalization": normalization,
        "split": split,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    if baseline_start_frame is not None:
        attrs["baseline_start_frame"] = int(baseline_start_frame)
    if baseline_end_frame is not None:
        attrs["baseline_end_frame"] = int(baseline_end_frame)
    if baseline_std_eps is not None:
        attrs["baseline_std_eps"] = float(baseline_std_eps)

    return xr.DataArray(
        image.astype(np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="vsd",
        attrs=attrs,
    )


def save_averaged_trial(da: xr.DataArray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(path)
