"""Evaluation figure helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _shared_limits(images: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([img.ravel() for img in images])
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


def plot_pixel_correlation_heatmap(
    corr_map: np.ndarray,
    output_path: Path,
    *,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> Path:
    """Save a BWR heatmap of per-pixel correlation values."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(corr_map, cmap="bwr", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pixel_r2_heatmap(
    r2_map: np.ndarray,
    output_path: Path,
    *,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> Path:
    """Save a BWR heatmap of per-pixel R² values."""
    return plot_pixel_correlation_heatmap(
        r2_map,
        output_path,
        title=title,
        vmin=vmin,
        vmax=vmax,
    )


def plot_pixel_mean_maps(
    mean_original: np.ndarray,
    mean_reconstruction: np.ndarray,
    mean_diff: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Side-by-side trial-mean original, reconstruction, and difference maps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    vmin, vmax = _shared_limits([mean_original, mean_reconstruction])
    diff_lim = float(np.nanpercentile(np.abs(mean_diff), 99))
    diff_lim = diff_lim if diff_lim > 1e-8 else 1.0

    panels = [
        (mean_original, "Trial-mean original", "viridis", vmin, vmax),
        (mean_reconstruction, "Trial-mean reconstruction", "viridis", vmin, vmax),
        (mean_diff, "Mean recon − original", "RdBu_r", -diff_lim, diff_lim),
    ]
    for ax, (img, subtitle, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(subtitle, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_condition_mean_originals(
    conditions: list[dict[str, object]],
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Grid of condition-averaged original VSD maps (one panel per condition)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not conditions:
        return output_path

    maps = [entry["map"] for entry in conditions]
    vmin, vmax = _shared_limits(maps)
    n = len(conditions)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    for ax, entry in zip(axes.ravel(), conditions):
        image = entry["map"]
        ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(
            f"{entry['date']} | {entry['condition']}\n"
            f"n = {entry['n_trials']} trials",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
