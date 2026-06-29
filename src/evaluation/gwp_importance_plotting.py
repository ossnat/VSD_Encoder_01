"""Plot Ridge-based GWP feature importance maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.gwp_importance import GwpImportanceMaps


def _normalize(img: np.ndarray, percentile: float = 99.0) -> tuple[np.ndarray, float]:
    vmax = float(np.percentile(img, percentile))
    vmax = vmax if vmax > 1e-12 else 1.0
    return img / vmax, vmax


def plot_scale_orientation_heatmap(
    maps: GwpImportanceMaps,
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Heatmap: scales (λ) × orientations (°)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = maps.scale_orientation
    data_n, _ = _normalize(data)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(data_n, aspect="auto", cmap="magma", origin="lower")
    ax.set_xticks(range(maps.n_orientations))
    ax.set_xticklabels([f"{d:.0f}°" for d in maps.orientations_deg], rotation=45)
    ax.set_yticks(range(maps.n_scales))
    ax.set_yticklabels([f"s{s} λ={maps.wavelengths_px[s]:.1f}" for s in range(maps.n_scales)])
    ax.set_xlabel("Orientation")
    ax.set_ylabel("Scale (spatial frequency)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Relative importance (|coef| mass, normalized)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_scale_bar(
    maps: GwpImportanceMaps,
    output_path: Path,
    *,
    title: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(maps.n_scales)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(
        x,
        maps.scale_total,
        color="steelblue",
        tick_label=[f"s{s}\nλ={maps.wavelengths_px[s]:.1f}" for s in range(maps.n_scales)],
    )
    ax.set_ylabel("Total |coef| mass")
    ax.set_xlabel("GWP scale")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_scale_spatial_grid(
    maps: GwpImportanceMaps,
    output_path: Path,
    *,
    title: str,
    upsample_to: int | None = 224,
) -> Path:
    """One spatial importance map per scale (pooled stimulus coordinates)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = maps.n_scales
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow))
    axes = np.atleast_2d(axes)

    for s in range(n):
        ax = axes.ravel()[s]
        img = maps.scale_spatial[s]
        if upsample_to is not None and img.shape[0] != upsample_to:
            import torch
            import torch.nn.functional as F

            t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
            img = (
                F.interpolate(
                    t,
                    size=(upsample_to, upsample_to),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )
        img_n, vmax = _normalize(img)
        ax.imshow(img_n, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"Scale {s} | λ≈{maps.wavelengths_px[s]:.1f}px", fontsize=9)
        ax.axis("off")

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_location_map(
    maps: GwpImportanceMaps,
    output_path: Path,
    *,
    title: str,
    upsample_to: int | None = 224,
) -> Path:
    """Marginal importance over all scales/orientations in pooled feature space."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = maps.feature_location
    if upsample_to is not None:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        img = (
            F.interpolate(
                t, size=(upsample_to, upsample_to), mode="bilinear", align_corners=False
            )
            .squeeze()
            .numpy()
        )
    img_n, _ = _normalize(img)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(img_n, cmap="hot", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Relative importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_vsd_output_importance(
    maps: GwpImportanceMaps,
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Which VSD map pixels are most sensitive to GWP features."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_n, _ = _normalize(maps.vsd_output)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(img_n, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Relative importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_orientation_polar(
    maps: GwpImportanceMaps,
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Polar bar chart of orientation marginals."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    theta = np.deg2rad(maps.orientations_deg)
    widths = np.pi / maps.n_orientations * 0.9
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="polar")
    ax.bar(theta, maps.orientation_total, width=widths, bottom=0.0, color="coral")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_title(title, pad=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_all_gwp_importance(
    maps: GwpImportanceMaps,
    output_dir: Path,
    *,
    prefix: str = "",
) -> dict[str, Path]:
    """Write the full importance figure set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    paths = {
        "scale_orientation": plot_scale_orientation_heatmap(
            maps,
            output_dir / f"{stem}importance_scale_x_orientation.png",
            title="GWP importance | scale × orientation",
        ),
        "scale_bar": plot_scale_bar(
            maps,
            output_dir / f"{stem}importance_by_scale.png",
            title="GWP importance | total per scale",
        ),
        "scale_spatial": plot_scale_spatial_grid(
            maps,
            output_dir / f"{stem}importance_spatial_per_scale.png",
            title="GWP importance | stimulus location per scale",
        ),
        "feature_location": plot_feature_location_map(
            maps,
            output_dir / f"{stem}importance_feature_location.png",
            title="GWP importance | pooled stimulus location (all scales)",
        ),
        "vsd_output": plot_vsd_output_importance(
            maps,
            output_dir / f"{stem}importance_vsd_output.png",
            title="GWP importance | VSD output pixels",
        ),
        "orientation_polar": plot_orientation_polar(
            maps,
            output_dir / f"{stem}importance_orientation_polar.png",
            title="GWP importance | orientation marginal",
        ),
    }
    return paths
