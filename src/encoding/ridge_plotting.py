"""QC plots for RidgeCV encoding models."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.plotting import select_sample_rows


def _shared_limits(images: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([img.ravel() for img in images])
    return float(np.percentile(vals, 1)), float(np.percentile(vals, 99))


def select_one_trial_per_condition(
    pairs: pd.DataFrame,
    *,
    prefer_split: str | None = "test",
) -> pd.DataFrame:
    """Pick one representative trial per (date, condition) for QC plots."""
    df = pairs.sort_values(
        ["date", "condition_num", "trial_index_in_condition", "trial_global_id"]
    ).copy()
    if prefer_split and (df["split"] == prefer_split).any():
        df["_prefer"] = (df["split"] == prefer_split).astype(int)
        df = df.sort_values(
            ["date", "condition", "_prefer"],
            ascending=[True, True, False],
        )
        df = df.drop(columns="_prefer")
    return df.drop_duplicates(["date", "condition"], keep="first").reset_index(drop=True)


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
        ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(
        f"{meta['date']} | {meta['condition']} | {meta.get('shape_type', '')}\n"
        f"id={meta['trial_global_id']} | {meta['split']} | {meta['trial_dataset']}",
        fontsize=10,
    )
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
            ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10)
            ax.set_ylabel(
                f"{meta['date']}\n{meta['condition']}",
                fontsize=9,
            )
            ax.set_xlabel("x")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_reconstructed_only_grid(
    samples: list[tuple[dict, np.ndarray]],
    output_path: Path,
    *,
    title: str = "RidgeCV reconstructions by condition",
    ncol: int = 4,
) -> Path:
    """Grid of reconstructed maps only — easy comparison across conditions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        return output_path

    images = [recon for _, recon in samples]
    vmin, vmax = _shared_limits(images)
    n = len(samples)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow))
    axes = np.atleast_2d(axes)

    for idx, (meta, reconstructed) in enumerate(samples):
        row, col = divmod(idx, ncol)
        ax = axes[row, col]
        ax.imshow(reconstructed, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{meta['date']} {meta['condition']}", fontsize=9)
        ax.axis("off")

    for idx in range(n, nrow * ncol):
        row, col = divmod(idx, ncol)
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_reconstruction_grid_pages(
    samples: list[tuple[dict, np.ndarray, np.ndarray]],
    output_dir: Path,
    *,
    title: str = "RidgeCV reconstructions",
    rows_per_page: int = 12,
) -> list[Path]:
    """Paginated orig|recon grids when there are many conditions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not samples:
        return []

    written: list[Path] = []
    n_pages = int(np.ceil(len(samples) / rows_per_page))
    for page in range(n_pages):
        chunk = samples[page * rows_per_page : (page + 1) * rows_per_page]
        suffix = f"_page{page + 1:02d}" if n_pages > 1 else ""
        out_path = output_dir / f"reconstructions_by_condition{suffix}.png"
        plot_reconstruction_grid(
            chunk,
            out_path,
            title=f"{title} ({page + 1}/{n_pages})",
        )
        written.append(out_path)
    return written


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
