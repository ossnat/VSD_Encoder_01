"""QC plots for RidgeCV encoding models."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.plotting import select_sample_rows
from src.plotting_colormaps import VSD_CMAP


def _shared_limits(images: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([img.ravel() for img in images])
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo != 0 else 1e-6
        return lo - pad, hi + pad
    return lo, hi


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


def _plot_spatial_heatmap(
    values: np.ndarray,
    output_path: Path,
    *,
    title: str,
    cmap: str,
    colorbar_label: str,
    underlay: np.ndarray | None = None,
    underlay_alpha: float = 0.45,
    log_scale: bool = False,
    diverging: bool = False,
) -> Path:
    """Heatmap over optional gray VSD underlay (shared by bias / α / weights)."""
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("heatmap has no finite values")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    if underlay is not None:
        u_lo, u_hi = _shared_limits([underlay])
        ax.imshow(underlay, cmap=VSD_CMAP, vmin=u_lo, vmax=u_hi, alpha=underlay_alpha)

    overlay_alpha = 0.85 if underlay is not None else 1.0
    if log_scale:
        positive = finite[finite > 0]
        if positive.size == 0:
            raise ValueError("log-scale heatmap needs positive values")
        vmin = float(positive.min())
        vmax = float(positive.max())
        if vmin == vmax:
            vmax = vmin * 10.0 if vmin > 0 else 1.0
        im = ax.imshow(
            values,
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            alpha=overlay_alpha,
        )
    elif diverging:
        vmax = float(np.percentile(np.abs(finite), 99))
        vmax = max(vmax, 1e-12)
        im = ax.imshow(
            values,
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            alpha=overlay_alpha,
        )
    else:
        vmin, vmax = _shared_limits([values])
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, alpha=overlay_alpha)

    ax.set_title(title, fontsize=10)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_bias_map(
    bias: np.ndarray,
    output_path: Path,
    *,
    title: str = "RidgeCV intercept (bias)",
    underlay: np.ndarray | None = None,
    underlay_alpha: float = 0.45,
) -> Path:
    """Heatmap of per-pixel RidgeCV intercept, optionally over a gray VSD underlay."""
    return _plot_spatial_heatmap(
        bias,
        output_path,
        title=title,
        cmap=VSD_CMAP,
        colorbar_label="bias",
        underlay=underlay,
        underlay_alpha=underlay_alpha,
        diverging=False,
    )


def plot_alpha_map(
    alpha: np.ndarray,
    output_path: Path,
    *,
    title: str = "RidgeCV alpha (per pixel)",
    underlay: np.ndarray | None = None,
    underlay_alpha: float = 0.45,
) -> Path:
    """Heatmap of per-pixel RidgeCV α, optionally over a gray VSD underlay."""
    return _plot_spatial_heatmap(
        alpha,
        output_path,
        title=title,
        cmap=VSD_CMAP,
        colorbar_label="α",
        underlay=underlay,
        underlay_alpha=underlay_alpha,
        log_scale=True,
    )


def plot_weight_norm_map(
    weight_norm: np.ndarray,
    output_path: Path,
    *,
    title: str = "RidgeCV weight L2 norm (per pixel)",
    underlay: np.ndarray | None = None,
    underlay_alpha: float = 0.45,
) -> Path:
    """Heatmap of per-pixel ||w||₂ across features, optionally over a gray VSD underlay."""
    return _plot_spatial_heatmap(
        weight_norm,
        output_path,
        title=title,
        cmap=VSD_CMAP,
        colorbar_label="||w||₂",
        underlay=underlay,
        underlay_alpha=underlay_alpha,
        log_scale=False,
    )


def plot_reconstruction_pair(
    meta: dict,
    original: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _shared_limits([original, reconstructed])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4), layout="constrained")
    titles = ["Original (H5 mean)", "Reconstructed (RidgeCV)"]
    im = None
    for ax, image, subtitle in zip(axes, [original, reconstructed], titles):
        im = ax.imshow(image, cmap=VSD_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03, label="VSD signal")
    fig.suptitle(
        f"{meta['date']} | {meta['condition']} | {meta.get('shape_type', '')}\n"
        f"id={meta['trial_global_id']} | {meta['split']} | {meta['trial_dataset']}",
        fontsize=10,
    )
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

    fig, axes = plt.subplots(
        n, 2, figsize=(8.8, 3.5 * n), layout="constrained"
    )
    if n == 1:
        axes = np.array([axes])
    col_titles = ["Original (H5 mean)", "Reconstructed (RidgeCV)"]

    for row_idx, (meta, original, reconstructed) in enumerate(samples):
        for col_idx, image in enumerate([original, reconstructed]):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(image, cmap=VSD_CMAP, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10)
            ax.set_ylabel(
                f"{meta['date']}\n{meta['condition']}",
                fontsize=9,
            )
            ax.set_xlabel("x")

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.012, pad=0.02, label="VSD signal")
    fig.suptitle(title, fontsize=11)
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
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.5 * ncol + 0.7, 3.5 * nrow),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)

    for idx, (meta, reconstructed) in enumerate(samples):
        row, col = divmod(idx, ncol)
        ax = axes[row, col]
        im = ax.imshow(reconstructed, cmap=VSD_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(f"{meta['date']} {meta['condition']}", fontsize=9)
        ax.axis("off")

    for idx in range(n, nrow * ncol):
        row, col = divmod(idx, ncol)
        axes[row, col].axis("off")

    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        fraction=0.012,
        pad=0.02,
        label="Reconstructed VSD signal",
    )
    fig.suptitle(title, fontsize=11)
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
