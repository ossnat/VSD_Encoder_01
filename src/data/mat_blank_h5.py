"""Extract blankAN trials from Gandalf condsAN .mat files into session H5."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from src.paths import resolve_data_path, workspace_root

BLANK_CONDITION = "blankAN"
DEFAULT_MONKEY = "gandalf"
DEFAULT_RAW_REL = Path("Data/FoundationData/RawData/gandalf")
DEFAULT_OUT_REL = Path("Data/FoundationData/ProcessedData/gandalf")

# Known local mat files → session date tags (match companion session_*_condsAN.h5).
DEFAULT_MAT_SESSIONS: tuple[tuple[str, str], ...] = (
    ("condsAN-002.mat", "240718c"),
    ("condsAN-001.mat", "270618b"),
)


def portable_data_path(abs_path: Path) -> str:
    """Return workspace-relative ``Data/...`` when possible."""
    resolved = abs_path.resolve()
    try:
        return str(resolved.relative_to(workspace_root()))
    except ValueError:
        return str(resolved)


def blank_h5_filename(date: str) -> str:
    return f"session_{date}_blank.h5"


def blank_h5_portable_path(monkey: str, date: str) -> str:
    rel = DEFAULT_OUT_REL.parent / monkey / blank_h5_filename(date)
    return portable_data_path(resolve_data_path(str(rel)))


def load_blank_trials(mat_path: Path) -> np.ndarray:
    """
    Load ``blankAN`` from a condsAN .mat file.

    Returns ``(n_trials, n_pixels, n_frames)`` float32.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    mat = scipy.io.loadmat(str(mat_path))
    if BLANK_CONDITION not in mat:
        keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"{BLANK_CONDITION} not in {mat_path.name}; keys={keys}")

    data = np.asarray(mat[BLANK_CONDITION])
    if data.ndim != 3:
        raise ValueError(
            f"Expected blankAN ndim=3 (pixels, frames, trials), got shape {data.shape}"
        )

    n_pixels, n_frames, n_trials = data.shape
    # (pixels, frames, trials) → (trials, pixels, frames)
    trials = np.transpose(data, (2, 0, 1)).astype(np.float32, copy=False)
    return trials


def build_trial_metadata(
    *,
    monkey: str,
    date: str,
    source_file: Path,
    target_file: Path,
    n_trials: int,
    n_pixels: int,
    n_frames: int,
) -> list[dict]:
    """One metadata dict per blank trial, matching ProcessedData session H5."""
    source = portable_data_path(source_file)
    target = portable_data_path(target_file)
    shape = [int(n_pixels), int(n_frames)]
    return [
        {
            "trial_global_id": trial_idx,
            "monkey": monkey,
            "date": date,
            "condition": BLANK_CONDITION,
            "source_file": source,
            "target_file": target,
            "trial_index_in_condition": trial_idx,
            "shape": shape,
        }
        for trial_idx in range(n_trials)
    ]


def write_blank_session_h5(
    mat_path: Path,
    output_h5: Path,
    *,
    monkey: str = DEFAULT_MONKEY,
    date: str,
) -> int:
    """
    Write ``session_{date}_blank.h5`` from ``blankAN`` in ``mat_path``.

    Returns the number of trials written.
    """
    mat_path = Path(mat_path).resolve()
    output_h5 = Path(output_h5).resolve()
    trials = load_blank_trials(mat_path)
    n_trials, n_pixels, n_frames = trials.shape

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    meta = build_trial_metadata(
        monkey=monkey,
        date=date,
        source_file=mat_path,
        target_file=output_h5,
        n_trials=n_trials,
        n_pixels=n_pixels,
        n_frames=n_frames,
    )

    with h5py.File(output_h5, "w") as f:
        for idx, trial in enumerate(trials):
            f.create_dataset(
                f"trial_{idx:06d}",
                data=trial,
                dtype=np.float32,
            )
        f.attrs["monkey"] = monkey
        f.attrs["date"] = date
        f.attrs["n_trials"] = n_trials
        f.attrs["created"] = datetime.now(timezone.utc).isoformat()
        f.attrs["trial_metadata_json"] = json.dumps(meta)

    return n_trials


def default_output_h5(monkey: str, date: str, *, repo: Path | None = None) -> Path:
    rel = DEFAULT_OUT_REL.parent / monkey / blank_h5_filename(date)
    return resolve_data_path(str(rel), repo)


def default_raw_mat_dir(*, repo: Path | None = None) -> Path:
    return resolve_data_path(str(DEFAULT_RAW_REL), repo)
