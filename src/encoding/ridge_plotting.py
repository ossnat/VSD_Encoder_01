"""QC plots for RidgeCV encoding models."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.plotting import select_sample_rows


def _shared_limits(images: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([img.ravel() for img in images])
    return float(np.percentile(vals, 1)), float(np.percentile(vals, 99))


def plot_bias_map(
    bias: np.ndarray,
    output_path: Path,
    *,
    title: str = "RidgeCV intercept (bias)",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _shared_limits([bias])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(bias, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_reconstruction_pair(
    meta: dict,
    original: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _shared_limits([original, reconstructed])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    titles = ["Original (H5 mean)", "Reconstructed (RidgeCV)"]
    for ax, image, subtitle in zip(axes, [original, reconstructed], titles):
        im = ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(
        f"id={meta['trial_global_id']} | {meta['condition']} | {meta['split']}\n"
        f"{meta['date']} {meta['trial_dataset']}",
        fontsize=10,
    )
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_reconstruction_grid(
    samples: list[tuple[dict, np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    title: str = "RidgeCV reconstructions",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        return output_path

    all_images = [orig for _, orig, _ in samples] + [recon for _, _, recon in samples]
    vmin, vmax = _shared_limits(all_images)
    n = len(samples)

    fig, axes = plt.subplots(n, 2, figsize=(8, 3.5 * n))
    if n == 1:
        axes = np.array([axes])
    col_titles = ["Original (H5 mean)", "Reconstructed (RidgeCV)"]

    for row_idx, (meta, original, reconstructed) in enumerate(samples):
        for col_idx, image in enumerate([original, reconstructed]):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10)
            ax.set_ylabel(
                f"{meta['trial_global_id']}\n{meta['condition']}",
                fontsize=9,
            )
            ax.set_xlabel("x")

    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(right=0.9)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def select_plot_samples(
    manifest_rows: list[dict],
    *,
    n_samples: int = 4,
    prefer_split: str = "test",
) -> list[dict]:
    preferred = [r for r in manifest_rows if r.get("split") == prefer_split]
    if len(preferred) >= n_samples:
        return select_sample_rows(preferred, n_samples=n_samples)
    return select_sample_rows(manifest_rows, n_samples=n_samples)
