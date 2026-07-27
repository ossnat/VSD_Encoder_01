#!/usr/bin/env python3
"""Sanity-check: trial-mean VSD spatial maps per frame for random conditions.

For each selected (date, condition), across all its trials:
  1. At each frame index in [start_frame, end_frame] (inclusive), mean the VSD
     maps across trials → one (H, W) mean image per frame.
  2. Plot those mean frames as a multi-panel spatial heatmap grid (shared
     color scale within the condition), using the project VSD colormap.

Outputs land under this experiment folder (PNGs + index.txt).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.h5_io import read_trial_by_global_id
from src.data.splits import load_trial_table
from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import VSD_CMAP

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = 17
DEFAULT_N_CONDITIONS = 6
DEFAULT_MIN_TRIALS = 3
DEFAULT_START_FRAME = 24
DEFAULT_END_FRAME = 45  # inclusive


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _eligible_groups(
    trials: pd.DataFrame, *, min_trials: int
) -> pd.DataFrame:
    counts = (
        trials.groupby(["date", "condition"], sort=True)
        .size()
        .reset_index(name="n_trials")
    )
    return counts[counts["n_trials"] >= min_trials].reset_index(drop=True)


def _sample_conditions(
    eligible: pd.DataFrame,
    *,
    n_conditions: int,
    seed: int,
) -> pd.DataFrame:
    """Pick up to n_conditions groups, preferring distinct dates."""
    if eligible.empty:
        raise RuntimeError("No eligible (date, condition) groups")

    rng = np.random.default_rng(seed)
    dates = rng.permutation(eligible["date"].unique())
    picked_rows: list[pd.Series] = []
    used_keys: set[tuple[str, str]] = set()

    # One random condition per date first (maximizes date diversity).
    for date in dates:
        if len(picked_rows) >= n_conditions:
            break
        pool = eligible[eligible["date"] == date]
        choice = pool.iloc[int(rng.integers(0, len(pool)))]
        key = (str(choice["date"]), str(choice["condition"]))
        picked_rows.append(choice)
        used_keys.add(key)

    # Fill remaining slots from leftover groups if needed.
    if len(picked_rows) < n_conditions:
        leftover = eligible[
            ~eligible.apply(
                lambda r: (str(r["date"]), str(r["condition"])) in used_keys,
                axis=1,
            )
        ].reset_index(drop=True)
        if not leftover.empty:
            order = rng.permutation(len(leftover))
            for idx in order:
                if len(picked_rows) >= n_conditions:
                    break
                picked_rows.append(leftover.iloc[int(idx)])

    out = pd.DataFrame(picked_rows).reset_index(drop=True)
    if len(out) < n_conditions:
        print(
            f"WARNING: only {len(out)} eligible groups "
            f"(requested {n_conditions})"
        )
    return out


def _trial_frame_maps(
    trial: np.ndarray,
    *,
    spatial_size: tuple[int, int],
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    """Return maps with shape (n_frames, H, W) for inclusive frame range."""
    height, width = spatial_size
    n_pixels, n_frames = trial.shape
    if n_pixels != height * width:
        raise ValueError(
            f"Expected {height * width} pixels, got {n_pixels} "
            f"for spatial_size={spatial_size}"
        )
    if start_frame < 0 or end_frame >= n_frames:
        raise ValueError(
            f"Frame range [{start_frame}, {end_frame}] out of bounds "
            f"for trial with {n_frames} frames"
        )
    frames = np.stack(
        [
            trial[:, frame].reshape(height, width)
            for frame in range(start_frame, end_frame + 1)
        ],
        axis=0,
    )
    return frames.astype(np.float32, copy=False)


def _mean_frame_maps(
    group_trials: pd.DataFrame,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    start_frame: int,
    end_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean across trials per frame → spatial maps.

    Returns
    -------
    frame_indices : (n_frames,)
    mean_maps : (n_frames, H, W)
    """
    stacks: list[np.ndarray] = []
    for row in group_trials.itertuples(index=False):
        h5_path = resolve_data_path(str(row.target_file), repo)
        trial = read_trial_by_global_id(h5_path, int(row.trial_global_id))
        stacks.append(
            _trial_frame_maps(
                trial,
                spatial_size=spatial_size,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )

    # (n_trials, n_frames, H, W) → mean over trials → (n_frames, H, W)
    mean_maps = np.mean(np.stack(stacks, axis=0), axis=0)
    frame_indices = np.arange(start_frame, end_frame + 1)
    return frame_indices, mean_maps.astype(np.float32, copy=False)


def _shared_limits(maps: np.ndarray) -> tuple[float, float]:
    """Percentile-based shared vmin/vmax across all frames in a condition."""
    finite = maps[np.isfinite(maps)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo != 0 else 1e-6
        return lo - pad, hi + pad
    return lo, hi


def _grid_shape(n_panels: int) -> tuple[int, int]:
    """Choose a compact subplot grid for n_panels frames."""
    if n_panels <= 0:
        raise ValueError("n_panels must be positive")
    n_cols = min(6, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    return n_rows, n_cols


def _plot_mean_frame_grid(
    *,
    frame_indices: np.ndarray,
    mean_maps: np.ndarray,
    date: str,
    condition: str,
    n_trials: int,
    output_path: Path,
) -> None:
    n_frames = len(frame_indices)
    n_rows, n_cols = _grid_shape(n_frames)
    vmin, vmax = _shared_limits(mean_maps)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.2 * n_cols, 2.2 * n_rows + 0.6),
        layout="constrained",
    )
    axes_flat = np.atleast_1d(axes).ravel()

    im = None
    for i, (frame_idx, mean_map) in enumerate(zip(frame_indices, mean_maps)):
        ax = axes_flat[i]
        im = ax.imshow(mean_map, cmap=VSD_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(f"f={int(frame_idx)}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n_frames, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if im is not None:
        fig.colorbar(
            im,
            ax=axes_flat[:n_frames].tolist(),
            fraction=0.02,
            pad=0.02,
            label="VSD signal",
        )
    fig.suptitle(
        f"{date} | {condition} | n_trials={n_trials}\n"
        f"trial-mean VSD maps (frames {int(frame_indices[0])}–"
        f"{int(frame_indices[-1])}, shared scale)",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/default.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_DIR,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--n-conditions", type=int, default=DEFAULT_N_CONDITIONS
    )
    parser.add_argument("--min-trials", type=int, default=DEFAULT_MIN_TRIALS)
    parser.add_argument(
        "--start-frame", type=int, default=DEFAULT_START_FRAME
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=DEFAULT_END_FRAME,
        help="Inclusive end frame index",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.end_frame < args.start_frame:
        raise ValueError(
            f"end_frame ({args.end_frame}) must be >= start_frame "
            f"({args.start_frame})"
        )

    repo = project_root()
    cfg = _load_yaml(args.config)
    monkey = cfg["monkey"]
    spatial_size = tuple(int(v) for v in cfg["spatial_size"])

    trials = load_trial_table(
        cfg["split_csv"],
        monkey,
        trials_index_csv=cfg.get("trials_index_csv"),
        project_root_path=repo,
    )
    available = trials["target_file"].apply(
        lambda p: resolve_data_path(p, repo).exists()
    )
    n_missing = int((~available).sum())
    if n_missing:
        print(f"Skipping {n_missing} trials with missing session H5")
        trials = trials[available].reset_index(drop=True)
    if trials.empty:
        raise FileNotFoundError(
            f"No trials with existing H5 for monkey={monkey!r}"
        )

    eligible = _eligible_groups(trials, min_trials=args.min_trials)
    selected = _sample_conditions(
        eligible,
        n_conditions=args.n_conditions,
        seed=args.seed,
    )

    out_dir = args.output_dir
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    index_lines = [
        "Trial-mean VSD spatial maps per frame (sanity check)",
        f"monkey: {monkey}",
        f"seed: {args.seed}",
        f"frame range (inclusive): [{args.start_frame}, {args.end_frame}]",
        "per-frame map: mean across trials of that condition at that frame",
        "plot: spatial heatmaps (shared color scale within condition)",
        f"colormap: {VSD_CMAP}",
        "",
        "selected (date, condition, n_trials, plot):",
    ]

    plotted: list[tuple[str, str, int]] = []
    for row in selected.itertuples(index=False):
        date = str(row.date)
        condition = str(row.condition)
        group = trials[
            (trials["date"].astype(str) == date)
            & (trials["condition"].astype(str) == condition)
        ].reset_index(drop=True)
        n_trials = len(group)
        print(f"Computing {date} / {condition} ({n_trials} trials)...")

        frame_indices, mean_maps = _mean_frame_maps(
            group,
            repo=repo,
            spatial_size=spatial_size,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        plot_name = f"{date}__{condition}.png"
        plot_path = plots_dir / plot_name
        _plot_mean_frame_grid(
            frame_indices=frame_indices,
            mean_maps=mean_maps,
            date=date,
            condition=condition,
            n_trials=n_trials,
            output_path=plot_path,
        )
        plotted.append((date, condition, n_trials))
        index_lines.append(
            f"  {date}\t{condition}\tn_trials={n_trials}\tplots/{plot_name}"
        )
        print(f"  wrote {plot_path.relative_to(repo)}")

    index_path = out_dir / "index.txt"
    index_path.write_text("\n".join(index_lines) + "\n")
    print(f"Index: {index_path}")
    print(f"Plotted {len(plotted)} conditions under {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
