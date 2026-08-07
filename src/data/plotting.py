"""QC plots for averaged trials."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting_colormaps import VSD_CMAP


def select_sample_rows(manifest_rows: list[dict], n_samples: int = 4) -> list[dict]:
    """Pick diverse trials (different conditions / splits when possible)."""
    if not manifest_rows:
        return []
    if len(manifest_rows) <= n_samples:
        return manifest_rows

    df = pd.DataFrame(manifest_rows)
    chosen: list[dict] = []
    used_ids: set[int] = set()

    for _, group in df.groupby(["condition", "split"], sort=False):
        row = group.iloc[0].to_dict()
        tid = int(row["trial_global_id"])
        if tid not in used_ids:
            chosen.append(row)
            used_ids.add(tid)
        if len(chosen) >= n_samples:
            break

    for row in manifest_rows:
        tid = int(row["trial_global_id"])
        if tid in used_ids:
            continue
        chosen.append(row)
        used_ids.add(tid)
        if len(chosen) >= n_samples:
            break

    return chosen[:n_samples]


def plot_averaged_samples(
    samples: list[tuple[dict, np.ndarray]],
    output_dir: Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[Path]:
    """Save one PNG per sample; return written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if samples:
        all_vals = np.concatenate([img.ravel() for _, img in samples])
        if vmin is None:
            vmin = float(np.percentile(all_vals, 1))
        if vmax is None:
            vmax = float(np.percentile(all_vals, 99))

    for meta, image in samples:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(image, cmap=VSD_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(
            f"id={meta['trial_global_id']} | {meta['condition']} | "
            f"{meta['split']}\n{meta['date']} {meta['trial_dataset']}",
            fontsize=9,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        out_path = output_dir / f"sample_{meta['trial_global_id']:06d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    if len(samples) > 1:
        n = len(samples)
        fig, axes = plt.subplots(
            1, n, figsize=(4 * n + 0.7, 4), layout="constrained"
        )
        if n == 1:
            axes = [axes]
        for ax, (meta, image) in zip(axes, samples):
            im = ax.imshow(image, cmap=VSD_CMAP, vmin=vmin, vmax=vmax)
            ax.set_title(
                f"{meta['trial_global_id']} | {meta['condition']} | {meta['split']}",
                fontsize=9,
            )
            ax.axis("off")
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
        fig.suptitle("Averaged trial samples", fontsize=11)
        grid_path = output_dir / "samples_grid.png"
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(grid_path)

    return written
