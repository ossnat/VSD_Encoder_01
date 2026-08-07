"""Read trial arrays from session HDF5 files."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np


def read_trial(h5_path: Path, trial_dataset: str) -> np.ndarray:
    """Load one trial as (n_pixels, n_frames) float32."""
    with h5py.File(h5_path, "r") as f:
        data = f[trial_dataset][...]
    return np.asarray(data, dtype=np.float32)


@lru_cache(maxsize=64)
def _trial_global_id_to_dataset(h5_path_str: str) -> dict[int, str]:
    """
    Map trial_global_id → flat H5 dataset name using trial_metadata_json order.

    Session H5 files store trials sequentially across conditions. The split CSV
    often labels the first trial in each condition as trial_000000, but the
    correct dataset is the trial's position in the file metadata list.
    """
    with h5py.File(h5_path_str, "r") as f:
        if "trial_metadata_json" not in f.attrs:
            raise KeyError(f"No trial_metadata_json attribute in {h5_path_str}")
        meta = json.loads(f.attrs["trial_metadata_json"])
    return {
        int(entry["trial_global_id"]): f"trial_{idx:06d}"
        for idx, entry in enumerate(meta)
    }


def resolve_trial_dataset(h5_path: Path, trial_global_id: int) -> str:
    """Return the H5 dataset name for a global trial id."""
    mapping = _trial_global_id_to_dataset(str(h5_path.resolve()))
    try:
        return mapping[int(trial_global_id)]
    except KeyError as exc:
        raise KeyError(
            f"trial_global_id={trial_global_id} not found in {h5_path}"
        ) from exc


def read_trial_by_global_id(h5_path: Path, trial_global_id: int) -> np.ndarray:
    """Load one trial using trial_global_id (handles split CSV naming quirks)."""
    dataset = resolve_trial_dataset(h5_path, trial_global_id)
    return read_trial(h5_path, dataset)
