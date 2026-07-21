#!/usr/bin/env python3
"""One-off QC plots of raw VSD frames and their training-target means."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.averaging import average_frames
from src.data.h5_io import read_trial_by_global_id
from src.encoding.schema import encoding_pairs_manifest_path
from src.paths import project_root, resolve_data_path


# Deliberately selected across sessions and stimulus families. The last entry is
# the rare white point whose 22 trials are already test-only in the v3 split.
DEFAULT_PICKS = (
    ("100718a", "condAN1", "black point"),
    ("230518b", "condAN1", "white filled circle (100% contrast)"),
    ("201118a", "condAN1", "letter G"),
    ("290518a", "condAN1", "white point (unseen/test-only)"),
)
CMAPS = ("mapgeog", "OrRd")


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _register_mapgeog(repo: Path) -> None:
    """Register the presentation colormap from the sibling foundation project."""
    module_path = (
        repo.parent
        / "VSD_foundation_model"
        / "src"
        / "utils"
        / "colormap_register.py"
    )
    if not module_path.exists():
        raise FileNotFoundError(
            "mapgeog definition not found at "
            f"{module_path}. Keep VSD_foundation_model beside this repository."
        )
    spec = importlib.util.spec_from_file_location("_vsd_mapgeog", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load mapgeog module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.register_custom_cmaps()


def _robust_limits(images: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(images, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite display limits from selected VSD data")
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def _select_trials(pairs: pd.DataFrame) -> list[tuple[pd.Series, str]]:
    selected: list[tuple[pd.Series, str]] = []
    for date, condition, label in DEFAULT_PICKS:
        rows = pairs[
            (pairs["date"] == date) & (pairs["condition"] == condition)
        ].sort_values("trial_global_id")
        if rows.empty:
            raise ValueError(f"No encoding pair for {date} {condition}")
        selected.append((rows.iloc[0], label))

    holdout = pairs[
        (pairs["date"] == "290518a") & (pairs["condition"] == "condAN1")
    ]
    splits = set(holdout["split"].astype(str))
    if splits != {"test"}:
        raise ValueError(
            "Expected 290518a/condAN1 to be test-only, "
            f"but found splits={sorted(splits)}"
        )
    return selected


def _plot_raw_frames(
    frames: np.ndarray,
    *,
    row: pd.Series,
    label: str,
    start_frame: int,
    end_frame: int,
    cmap: str,
    output_path: Path,
) -> None:
    n_frames = end_frame - start_frame
    n_cols = 5
    n_rows = int(np.ceil(n_frames / n_cols))
    fig = plt.figure(figsize=(2.45 * n_cols + 0.7, 2.35 * n_rows))
    grid = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1] * n_cols + [0.045],
        left=0.02,
        right=0.96,
        bottom=0.04,
        top=0.82,
        wspace=0.22,
        hspace=0.25,
    )
    axes = np.asarray(
        [[fig.add_subplot(grid[row, col]) for col in range(n_cols)]
         for row in range(n_rows)]
    )
    colorbar_ax = fig.add_subplot(grid[:, -1])
    vmin, vmax = _robust_limits(frames)
    image = None
    for offset, ax in enumerate(axes.flat):
        if offset >= n_frames:
            ax.axis("off")
            continue
        image = ax.imshow(
            frames[offset],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"frame {start_frame + offset}", fontsize=9)
        ax.axis("off")
    if image is not None:
        fig.colorbar(
            image,
            cax=colorbar_ax,
            label="VSD signal",
        )
    fig.suptitle(
        f"{row['date']} · {row['condition']} · {label}\n"
        f"trial {int(row['trial_global_id'])} · raw frames "
        f"[{start_frame}, {end_frame}) · {cmap}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_grid(
    samples: list[dict],
    *,
    start_frame: int,
    end_frame: int,
    cmap: str,
    output_path: Path,
) -> None:
    means = np.stack([sample["mean"] for sample in samples])
    vmin, vmax = _robust_limits(means)
    n_samples = len(samples)
    fig = plt.figure(figsize=(4 * n_samples + 0.7, 4.2))
    grid = fig.add_gridspec(
        1,
        n_samples + 1,
        width_ratios=[1] * n_samples + [0.045],
        left=0.02,
        right=0.96,
        bottom=0.06,
        top=0.72,
        wspace=0.22,
    )
    axes = np.asarray([fig.add_subplot(grid[0, col]) for col in range(n_samples)])
    colorbar_ax = fig.add_subplot(grid[0, -1])
    image = None
    for ax, sample in zip(axes, samples):
        row = sample["row"]
        image = ax.imshow(
            sample["mean"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(
            f"{row['date']} · {row['condition']}\n"
            f"{sample['label']}\ntrial {int(row['trial_global_id'])}",
            fontsize=9,
        )
        ax.axis("off")
    if image is not None:
        fig.colorbar(
            image,
            cax=colorbar_ax,
            label="Mean VSD signal",
        )
    fig.suptitle(
        f"Encoder targets: per-trial mean over frames "
        f"[{start_frame}, {end_frame}) · {cmap}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/default.yaml",
    )
    parser.add_argument(
        "--window",
        type=Path,
        default=project_root() / "configs/windows/evoked_32_42.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = project_root()
    cfg = _load_yaml(args.config)
    window_cfg = _load_yaml(args.window)
    start_frame = int(window_cfg["start_frame"])
    end_frame = int(window_cfg["end_frame"])
    spatial_size = tuple(int(v) for v in cfg["spatial_size"])
    window_id = str(window_cfg["window_id"])

    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        cfg["monkey"],
        window_id,
    )
    pairs = pd.read_parquet(pairs_path)
    selected = _select_trials(pairs)
    _register_mapgeog(repo)

    samples: list[dict] = []
    for row, label in selected:
        h5_path = resolve_data_path(row["target_file"], repo)
        trial = read_trial_by_global_id(h5_path, int(row["trial_global_id"]))
        frames = np.stack(
            [
                trial[:, frame].reshape(spatial_size)
                for frame in range(start_frame, end_frame)
            ]
        )
        mean_map = average_frames(
            trial,
            start_frame,
            end_frame,
            spatial_size=spatial_size,
            method=str(window_cfg.get("avg_method", "mean")),
        )
        if not np.allclose(mean_map, frames.mean(axis=0), equal_nan=True):
            raise AssertionError("Displayed mean does not match displayed raw frames")
        samples.append(
            {"row": row, "label": label, "frames": frames, "mean": mean_map}
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for cmap in CMAPS:
        cmap_dir = args.output_dir / cmap
        raw_dir = cmap_dir / "raw_frames"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            row = sample["row"]
            path = raw_dir / (
                f"{row['date']}__{row['condition']}__"
                f"trial_{int(row['trial_global_id']):06d}.png"
            )
            _plot_raw_frames(
                sample["frames"],
                row=row,
                label=sample["label"],
                start_frame=start_frame,
                end_frame=end_frame,
                cmap=cmap,
                output_path=path,
            )
            outputs.append(str(path.relative_to(repo)))

        mean_path = cmap_dir / "averaged_encoder_targets.png"
        _plot_mean_grid(
            samples,
            start_frame=start_frame,
            end_frame=end_frame,
            cmap=cmap,
            output_path=mean_path,
        )
        outputs.append(str(mean_path.relative_to(repo)))

    manifest = {
        "purpose": "one-off raw-frame and averaged-target QC",
        "window": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": list(range(start_frame, end_frame)),
            "convention": "half-open",
        },
        "colormaps": list(CMAPS),
        "samples": [
            {
                "trial_global_id": int(sample["row"]["trial_global_id"]),
                "date": str(sample["row"]["date"]),
                "condition": str(sample["row"]["condition"]),
                "split": str(sample["row"]["split"]),
                "label": sample["label"],
            }
            for sample in samples
        ],
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Frame window: [{start_frame}, {end_frame})")
    print(f"Selected trials: {len(samples)}")
    print(f"Wrote {len(outputs)} figures under {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
