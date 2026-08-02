"""Evaluation figure helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from src.evaluation.mask import apply_mask_nan
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


def plot_pixel_correlation_heatmap(
    corr_map: np.ndarray,
    output_path: Path,
    *,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
    underlay: np.ndarray | None = None,
    underlay_alpha: float = 0.45,
) -> Path:
    """Save a mapgeog heatmap of per-pixel correlation values."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    if underlay is not None:
        u_lo, u_hi = _shared_limits([underlay])
        ax.imshow(
            underlay, cmap=VSD_CMAP, vmin=u_lo, vmax=u_hi, alpha=underlay_alpha
        )
    im = ax.imshow(corr_map, cmap=VSD_CMAP, vmin=vmin, vmax=vmax, alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_backbone_correlation_comparison(
    panels: list[tuple[str, np.ndarray]],
    underlay: np.ndarray,
    output_path: Path,
    *,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
    underlay_alpha: float = 0.5,
) -> Path:
    """Side-by-side pixel-r heatmaps with a shared VSD underlay."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(4.5 * n + 0.7, 4.5), layout="constrained"
    )
    if n == 1:
        axes = [axes]

    u_lo, u_hi = _shared_limits([underlay])
    for ax, (label, corr_map) in zip(axes, panels):
        ax.imshow(
            underlay, cmap=VSD_CMAP, vmin=u_lo, vmax=u_hi, alpha=underlay_alpha
        )
        im = ax.imshow(
            corr_map, cmap=VSD_CMAP, vmin=vmin, vmax=vmax, alpha=0.82
        )
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    fig.suptitle(title, fontsize=11)
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
        (mean_original, "Trial-mean original", VSD_CMAP, vmin, vmax),
        (mean_reconstruction, "Trial-mean reconstruction", VSD_CMAP, vmin, vmax),
        (mean_diff, "Mean recon − original", VSD_CMAP, -diff_lim, diff_lim),
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


def _add_mask_outline(ax, spatial_size: tuple[int, int], mask_radius: int) -> None:
    height, width = spatial_size
    cy, cx = height / 2.0, width / 2.0
    ax.add_patch(
        Circle(
            (cx, cy),
            mask_radius,
            fill=False,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.9,
        )
    )


def plot_masked_map_panel(
    ax,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    spatial_size: tuple[int, int],
    mask_radius: int,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> object:
    """Single map panel with NaN outside mask and circle outline."""
    im = ax.imshow(apply_mask_nan(image, mask), cmap=cmap, vmin=vmin, vmax=vmax)
    _add_mask_outline(ax, spatial_size, mask_radius)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return im


def plot_test_conditions_grid(
    conditions: list[dict[str, object]],
    output_path: Path,
    *,
    mask: np.ndarray,
    spatial_size: tuple[int, int],
    mask_radius: int,
    model_label: str,
    split: str,
) -> Path:
    """
    Grid of masked orig|recon|diff triptychs, one row per (date, condition).

    Each entry must include keys: date, condition, original, reconstruction,
    optional trial_r_masked.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not conditions:
        return output_path

    n = len(conditions)
    fig, axes = plt.subplots(
        n, 3, figsize=(11.3, 3.2 * n), layout="constrained"
    )
    if n == 1:
        axes = np.array([axes])

    all_orig = [c["original"] for c in conditions]
    all_recon = [c["reconstruction"] for c in conditions]
    masked_orig = [apply_mask_nan(np.asarray(o), mask) for o in all_orig]
    masked_recon = [apply_mask_nan(np.asarray(r), mask) for r in all_recon]
    vmin, vmax = _shared_limits(masked_orig + masked_recon)
    diffs = [
        apply_mask_nan((np.asarray(r) - np.asarray(o)).astype(np.float32), mask)
        for o, r in zip(all_orig, all_recon)
    ]
    diff_lim = float(np.nanpercentile(np.abs(np.stack(diffs)), 99))
    diff_lim = diff_lim if diff_lim > 1e-8 else 1.0

    col_titles = ["Condition-mean original", "Reconstruction", "Recon − original"]
    for row, entry in enumerate(conditions):
        orig = np.asarray(entry["original"], dtype=np.float32)
        recon = np.asarray(entry["reconstruction"], dtype=np.float32)
        diff = (recon - orig).astype(np.float32)
        row_title = (
            f"{entry['date']} | {entry['condition']} | n={entry['n_trials']}"
        )
        tr = entry.get("trial_r_masked")
        if tr is not None and np.isfinite(tr):
            row_title += f" | mean trial r={tr:.3f}"
        images = [orig, recon, diff]
        specs = [
            (VSD_CMAP, vmin, vmax),
            (VSD_CMAP, vmin, vmax),
            (VSD_CMAP, -diff_lim, diff_lim),
        ]
        for col, (img, (cmap, lo, hi)) in enumerate(zip(images, specs)):
            title = row_title if col == 0 else col_titles[col]
            im = plot_masked_map_panel(
                axes[row, col],
                img,
                mask,
                spatial_size=spatial_size,
                mask_radius=mask_radius,
                title=title,
                cmap=cmap,
                vmin=lo,
                vmax=hi,
            )
            if row == n - 1:
                fig.colorbar(
                    im,
                    ax=axes[:, col].tolist(),
                    fraction=0.02,
                    pad=0.02,
                    label=col_titles[col],
                )

    fig.suptitle(
        f"{model_label} | {split} conditions | masked r={mask_radius}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metrics_bar_comparison(
    df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Bar chart of masked test metrics for each model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return output_path

    labels = df["label"].tolist()
    metrics = [
        ("r_mean_test_masked", "Trial r"),
        ("eval_mean_r_masked", "Pixel r (trials)"),
        ("mean_r_across_conditions_masked", "Pixel r (conditions)"),
        ("eval_mean_r2_masked", "Pixel R² (trials)"),
        ("mean_r2_across_conditions_masked", "Pixel R² (conditions)"),
    ]
    metrics = [(c, n) for c, n in metrics if c in df.columns]
    x = np.arange(len(labels))
    width = 0.8 / max(len(metrics), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 3.5), 4.2))
    for i, (col, ylab) in enumerate(metrics):
        vals = df[col].astype(float).to_numpy()
        offset = (i - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=ylab)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0.0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_condition_trial_r(
    rows: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Grouped bars: per (date, condition) mean trial r for each model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows.empty:
        return output_path

    cond_keys = rows.drop_duplicates(["date", "condition"])[["date", "condition"]]
    cond_labels = [f"{r.date}\n{r.condition}" for r in cond_keys.itertuples(index=False)]
    models = list(rows["model_label"].unique())
    x = np.arange(len(cond_labels))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(cond_labels) * 2), 4))
    for i, model in enumerate(models):
        sub = rows[rows["model_label"] == model]
        vals = []
        for date, condition in cond_keys.itertuples(index=False):
            row = sub[(sub["date"] == date) & (sub["condition"] == condition)]
            vals.append(
                float(row["trial_r_masked"].iloc[0]) if not row.empty else float("nan")
            )
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=model)

    ax.set_xticks(x, cond_labels, fontsize=8)
    ax.set_ylabel("Mean trial r (masked)")
    ax.set_title(title)
    ax.legend()
    ax.axhline(0.0, color="k", linewidth=0.5)
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

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.5 * ncol + 0.7, 3.5 * nrow),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    for ax, entry in zip(axes.ravel(), conditions):
        image = entry["map"]
        im = ax.imshow(image, cmap=VSD_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(
            f"{entry['date']} | {entry['condition']}\n"
            f"n = {entry['n_trials']} trials",
            fontsize=9,
        )
        ax.axis("off")

    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        fraction=0.015,
        pad=0.02,
        label="Mean VSD signal",
    )
    fig.suptitle(title, fontsize=11)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
