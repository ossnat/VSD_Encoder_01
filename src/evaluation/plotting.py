"""Evaluation figure helpers."""

from __future__ import annotations

import json
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


def _disable_colorbar_offset(cbar) -> None:
    """Print 1.000x on the bar, not 0.000x with a matplotlib '+1' offset."""
    formatter = getattr(cbar, "formatter", None)
    if formatter is not None and hasattr(formatter, "set_useOffset"):
        formatter.set_useOffset(False)
        if hasattr(formatter, "set_scientific"):
            formatter.set_scientific(False)
        cbar.update_ticks()
        return
    from matplotlib.ticker import ScalarFormatter

    axis = cbar.ax.yaxis if getattr(cbar, "orientation", "vertical") == "vertical" else cbar.ax.xaxis
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    axis.set_major_formatter(fmt)


def shared_orig_recon_residual_clims(
    orig_maps: list[np.ndarray],
    recon_maps: list[np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Run-wide orig/recon 1–99% clim and a symmetric residual clim.

    Residual is **not** tied to the F/F0 orig/recon range (raw residuals are
    ~10⁻³–10⁻⁴ while orig/recon sit near 1).
    """
    orig_recon = _shared_limits(
        [np.asarray(m) for m in list(orig_maps) + list(recon_maps)]
    )
    diffs = [
        np.asarray(r, dtype=np.float64) - np.asarray(o, dtype=np.float64)
        for o, r in zip(orig_maps, recon_maps)
    ]
    if not diffs:
        return orig_recon, (-1.0, 1.0)
    abs_diff = np.concatenate([np.abs(d.ravel()) for d in diffs])
    finite = abs_diff[np.isfinite(abs_diff)]
    diff_lim = float(np.percentile(finite, 99)) if finite.size else 1.0
    diff_lim = diff_lim if diff_lim > 1e-8 else 1.0
    return orig_recon, (-diff_lim, diff_lim)


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


def plot_loo_all_shapes_orig_recon(
    fold_dirs: list[Path],
    *,
    valid: np.ndarray,
    out_path: Path,
    title: str,
) -> Path:
    """One orig|recon row per LOO fold, from cached fold-mean maps."""
    from src.encoding.ridge_plotting import plot_reconstruction_grid

    samples = []
    for fold_dir in fold_dirs:
        metrics_path = fold_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        sid = str(metrics.get("heldout_stimulus_id") or fold_dir.name)
        date = metrics.get("heldout_date")
        cond = metrics.get("heldout_condition")
        pairs_path = fold_dir / "fold_pairs.parquet"
        if (not date or not cond) and pairs_path.is_file():
            pairs = pd.read_parquet(pairs_path)
            test = (
                pairs[pairs["loo_split"] == "test"]
                if "loo_split" in pairs.columns
                else pairs
            )
            if not test.empty:
                date = date or str(test["date"].iloc[0])
                if "condition" in test.columns:
                    cond = cond or str(test["condition"].iloc[0])
        if date:
            label = f"{sid}\n{date}" + (f"/{cond}" if cond else "")
        else:
            label = sid
        orig = np.load(fold_dir / "fold_mean_orig.npy").astype(np.float32)
        recon = np.load(fold_dir / "fold_mean_recon.npy").astype(np.float32)
        samples.append(
            (
                {
                    "date": str(date or "loo"),
                    "condition": str(cond or sid),
                    "stimulus_label": label,
                    "shape_type": "",
                    "trial_global_id": 0,
                    "split": "loo_test_mean",
                    "trial_dataset": "",
                },
                apply_mask_nan(orig, valid),
                apply_mask_nan(recon, valid),
            )
        )
    return plot_reconstruction_grid(samples, out_path, title=title)


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


def _pixel_mean_map_clims(
    mean_original: np.ndarray,
    mean_reconstruction: np.ndarray,
    mean_diff: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    residual_vmin: float | None = None,
    residual_vmax: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return ``(orig_clim, recon_clim, residual_clim)``.

    Default: independent 1–99% percentiles for orig and recon so identical
    ``y_hat`` keeps the same recon appearance across sessions. Residual uses
    its own symmetric scale. If both ``vmin`` and ``vmax`` are set, orig and
    recon share that fixed scale (residual still independent unless
    ``residual_vmin`` / ``residual_vmax`` are also set).
    """
    if (vmin is None) != (vmax is None):
        raise ValueError("vmin and vmax must both be provided or both omitted")
    if (residual_vmin is None) != (residual_vmax is None):
        raise ValueError(
            "residual_vmin and residual_vmax must both be provided or both omitted"
        )
    if vmin is not None:
        shared = (float(vmin), float(vmax))
        orig_clim = shared
        recon_clim = shared
    else:
        orig_clim = _shared_limits([np.asarray(mean_original)])
        recon_clim = _shared_limits([np.asarray(mean_reconstruction)])
    if residual_vmin is not None:
        resid_clim = (float(residual_vmin), float(residual_vmax))
    else:
        abs_diff = np.abs(np.asarray(mean_diff, dtype=np.float64))
        finite = abs_diff[np.isfinite(abs_diff)]
        diff_lim = float(np.percentile(finite, 99)) if finite.size else 1.0
        diff_lim = diff_lim if diff_lim > 1e-8 else 1.0
        resid_clim = (-diff_lim, diff_lim)
    return orig_clim, recon_clim, resid_clim


def plot_pixel_mean_maps(
    mean_original: np.ndarray,
    mean_reconstruction: np.ndarray,
    mean_diff: np.ndarray,
    output_path: Path,
    *,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    residual_vmin: float | None = None,
    residual_vmax: float | None = None,
    cmap: str | None = None,
) -> Path:
    """Side-by-side trial-mean original, reconstruction, and difference maps.

    Default colormap is ``VSD_CMAP`` (``mapgeog``). Pass ``cmap="mapgeog_gray"``
    for the project grayscale analog. Orig uses 1–99% percentiles of
    the original (sessions may differ). Recon uses **recon-only** percentiles
    so identical ``y_hat`` looks the same across folds (e.g. Protocol C
    100718a vs b). Residual keeps its own symmetric scale.

    Pass both ``vmin`` and ``vmax`` to force a shared fixed scale on orig and
    recon for run-wide comparison. Residual is never tied to that scale;
    pass ``residual_vmin`` / ``residual_vmax`` to share a residual clim
    across folds. Colorbars disable matplotlib offset notation so F/F0 ≈ 1
    prints as 1.000x, not 0.000x + 1.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    cmap_name = cmap or VSD_CMAP

    orig_clim, recon_clim, resid_clim = _pixel_mean_map_clims(
        mean_original,
        mean_reconstruction,
        mean_diff,
        vmin=vmin,
        vmax=vmax,
        residual_vmin=residual_vmin,
        residual_vmax=residual_vmax,
    )
    panels = [
        (mean_original, "Trial-mean original", cmap_name, *orig_clim),
        (mean_reconstruction, "Trial-mean reconstruction", cmap_name, *recon_clim),
        (mean_diff, "Mean recon − original", cmap_name, *resid_clim),
    ]
    for ax, (img, subtitle, panel_cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(img, cmap=panel_cmap, vmin=lo, vmax=hi)
        ax.set_title(subtitle, fontsize=10)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _disable_colorbar_offset(cbar)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _recon_and_orig_trial_clims(
    mean_reconstruction: np.ndarray,
    trial_originals: np.ndarray,
    trial_indices: list[int],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``(recon_clim, orig_clim)`` with independent recon scaling."""
    if (vmin is None) != (vmax is None):
        raise ValueError("vmin and vmax must both be provided or both omitted")
    if vmin is not None:
        orig_clim = (float(vmin), float(vmax))
    else:
        sampled = [np.asarray(trial_originals[i]) for i in trial_indices]
        orig_clim = _shared_limits(sampled)
    recon_clim = _shared_limits([np.asarray(mean_reconstruction)])
    return recon_clim, orig_clim


def plot_recon_with_sample_trial_originals(
    mean_reconstruction: np.ndarray,
    trial_originals: np.ndarray,
    output_path: Path,
    *,
    title: str,
    n_samples: int = 3,
    seed: int = 17,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Reconstruction plus K randomly sampled individual-trial originals.

    Recon uses recon-only percentiles; sampled originals share a separate scale.
    Residual / diff panel is omitted. ``seed`` fixes trial selection per fold.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_trials = int(trial_originals.shape[0])
    n_show = min(max(int(n_samples), 1), n_trials)
    rng = np.random.default_rng(seed)
    if n_trials <= n_show:
        trial_indices = list(range(n_trials))
    else:
        trial_indices = sorted(
            rng.choice(n_trials, size=n_show, replace=False).tolist()
        )

    recon_clim, orig_clim = _recon_and_orig_trial_clims(
        mean_reconstruction,
        trial_originals,
        trial_indices,
        vmin=vmin,
        vmax=vmax,
    )
    n_panels = 1 + len(trial_indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels + 0.5, 3.8))
    if n_panels == 1:
        axes = [axes]

    recon_im = axes[0].imshow(
        mean_reconstruction, cmap=VSD_CMAP, vmin=recon_clim[0], vmax=recon_clim[1]
    )
    axes[0].set_title("Reconstruction", fontsize=10)
    axes[0].axis("off")
    fig.colorbar(recon_im, ax=axes[0], fraction=0.046, pad=0.04)

    for ax, trial_idx in zip(axes[1:], trial_indices):
        orig = np.asarray(trial_originals[trial_idx])
        im = ax.imshow(orig, cmap=VSD_CMAP, vmin=orig_clim[0], vmax=orig_clim[1])
        ax.set_title(f"Original trial {trial_idx + 1}/{n_trials}", fontsize=10)
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
    # Clim from originals only — off-scale recons must not wash out maps.
    vmin, vmax = _shared_limits(masked_orig)
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
