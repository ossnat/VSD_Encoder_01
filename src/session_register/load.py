"""Mean VSD maps from session H5 trials, with skip-existing npy cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.splits import load_trial_table
from src.data.trial_frames import load_h5_mean_frame
from src.paths import project_root, resolve_data_path
from src.session_register.config import MixedMonkeyError, SessionRegisterConfig


def mean_cache_paths(cache_dir: Path, session: str) -> tuple[Path, Path]:
    stem = f"mean_vessel__{session}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.yaml"


def _select_conditions(df: pd.DataFrame, conditions: str | tuple[str, ...]) -> pd.DataFrame:
    if conditions == "all":
        return df
    wanted = {str(c) for c in conditions}
    out = df[df["condition"].astype(str).isin(wanted)].copy()
    return out


def _sample_rows(df: pd.DataFrame, n_trials: int | None, seed: int) -> pd.DataFrame:
    ordered = df.sort_values("trial_global_id").reset_index(drop=True)
    if n_trials is None or len(ordered) <= int(n_trials):
        return ordered
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(ordered), size=int(n_trials), replace=False))
    return ordered.iloc[idx].reset_index(drop=True)


def session_trial_rows(
    cfg: SessionRegisterConfig,
    session: str,
    *,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Local H5 trials for ``session`` (``date`` column = h5 session id)."""
    repo = repo or project_root()
    trials = load_trial_table(
        cfg.split_csv,
        cfg.monkey,
        trials_index_csv=cfg.trials_index_csv,
        project_root_path=repo,
    )
    if "monkey" in trials.columns:
        other = set(trials["monkey"].astype(str).str.lower().unique())
        if other - {cfg.monkey.lower()}:
            raise MixedMonkeyError(
                f"Trial table for requested monkey={cfg.monkey!r} also has {sorted(other)}"
            )
    rows = trials[trials["date"].astype(str) == str(session)].copy()
    rows = _select_conditions(rows, cfg.conditions)
    if rows.empty:
        raise ValueError(
            f"No split-CSV trials for session {session!r} monkey={cfg.monkey!r}"
        )
    available = rows["target_file"].apply(lambda p: resolve_data_path(p, repo).exists())
    rows = rows.loc[available].reset_index(drop=True)
    if rows.empty:
        raise FileNotFoundError(
            f"No local H5 files for session {session!r} monkey={cfg.monkey!r}"
        )
    return rows


def load_session_mean_map(
    cfg: SessionRegisterConfig,
    session: str,
    *,
    repo: Path | None = None,
    cache_dir: Path | None = None,
    overwrite: bool | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Average ``[start_frame, end_frame)`` over sampled trials; cache to npy.

    Cache hits only when trial ids, window, normalization, and spatial size
    match the YAML sidecar — a later QC replot does not re-read H5.
    """
    repo = repo or project_root()
    overwrite = cfg.overwrite if overwrite is None else bool(overwrite)
    cache_dir = Path(cache_dir) if cache_dir is not None else cfg.cache_dir(repo)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path, yaml_path = mean_cache_paths(cache_dir, session)

    rows = session_trial_rows(cfg, session, repo=repo)
    sample = _sample_rows(rows, cfg.n_trials, cfg.seed)
    trial_ids = [int(x) for x in sample["trial_global_id"].tolist()]
    expected_meta = {
        "session": str(session),
        "monkey": cfg.monkey,
        "trial_global_ids": trial_ids,
        "n_trials": len(trial_ids),
        "frame_window": cfg.frame_window,
        "n_window_frames": cfg.n_window_frames,
        "normalization": cfg.normalization,
        "spatial_size": [int(cfg.spatial_size[0]), int(cfg.spatial_size[1])],
        "conditions": (
            "all"
            if cfg.conditions == "all"
            else list(cfg.conditions)
            if isinstance(cfg.conditions, tuple)
            else cfg.conditions
        ),
    }

    if npy_path.is_file() and yaml_path.is_file() and not overwrite:
        cached = yaml.safe_load(yaml_path.read_text()) or {}
        keys = (
            "trial_global_ids",
            "frame_window",
            "normalization",
            "spatial_size",
            "monkey",
        )
        if all(cached.get(k) == expected_meta.get(k) for k in keys):
            mean_map = np.load(npy_path)
            if tuple(mean_map.shape) != tuple(cfg.spatial_size):
                raise ValueError(
                    f"Cached {npy_path} has shape {mean_map.shape}, "
                    f"expected {cfg.spatial_size}"
                )
            meta = dict(cached)
            meta["cache_hit"] = True
            meta["npy"] = str(npy_path)
            return np.asarray(mean_map, dtype=np.float32), meta

    maps = [
        load_h5_mean_frame(
            target_file=str(row.target_file),
            trial_global_id=int(row.trial_global_id),
            repo=repo,
            spatial_size=tuple(cfg.spatial_size),
            start_frame=int(cfg.start_frame),
            end_frame=int(cfg.end_frame),
            avg_method=str(cfg.avg_method),
            normalization=str(cfg.normalization),
        )
        for row in sample.itertuples(index=False)
    ]
    mean_map = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
    np.save(npy_path, mean_map)
    expected_meta["cache_hit"] = False
    expected_meta["npy"] = str(npy_path)
    yaml_path.write_text(yaml.safe_dump(expected_meta, sort_keys=False))
    return mean_map, expected_meta


def load_session_trial_stack(
    cfg: SessionRegisterConfig,
    session: str,
    *,
    repo: Path | None = None,
    cache_dir: Path | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Return ``(N, H, W)`` float32 array of per-trial frame means.

    By default uses ``cfg.start_frame / cfg.end_frame`` (the evoked window).
    Pass explicit ``start_frame`` / ``end_frame`` to use a different window —
    most importantly the **pre-stimulus baseline** (e.g. frames 2–26) which
    contains only vessel shadows and no stimulus-driven cortical response.

    Returns ``(stack, trial_global_ids)``.
    """
    repo = repo or project_root()

    rows = session_trial_rows(cfg, session, repo=repo)
    sample = _sample_rows(rows, cfg.n_trials, cfg.seed)
    trial_ids = [int(x) for x in sample["trial_global_id"].tolist()]

    sf = int(cfg.start_frame) if start_frame is None else int(start_frame)
    ef = int(cfg.end_frame) if end_frame is None else int(end_frame)

    maps = [
        load_h5_mean_frame(
            target_file=str(row.target_file),
            trial_global_id=int(row.trial_global_id),
            repo=repo,
            spatial_size=tuple(cfg.spatial_size),
            start_frame=sf,
            end_frame=ef,
            avg_method=str(cfg.avg_method),
            normalization=str(cfg.normalization),
        )
        for row in sample.itertuples(index=False)
    ]
    stack = np.stack(maps, axis=0).astype(np.float32)  # (N, H, W)
    return stack, trial_ids
