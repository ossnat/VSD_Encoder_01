"""Plot VSD responses alongside rendered stimulus images."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.plotting_colormaps import VSD_CMAP


def plot_vsd_vs_stimulus_pair(
    meta: dict,
    vsd_map: np.ndarray,
    stimulus_rgb: np.ndarray,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vsd_vmin, vsd_vmax = np.percentile(vsd_map, [1, 99])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4), layout="constrained")
    im0 = axes[0].imshow(vsd_map, cmap=VSD_CMAP, vmin=vsd_vmin, vmax=vsd_vmax)
    axes[0].set_title("VSD (H5 mean)", fontsize=10)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="VSD signal")

    axes[1].imshow(stimulus_rgb)
    axes[1].set_title("Rendered stimulus", fontsize=10)
    axes[1].axis("off")

    shape = meta.get("shape_type", "")
    fig.suptitle(
        f"{meta['date']} | {meta['condition']} | {shape}\n{meta.get('stimulus_text', '')}",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_vsd_vs_stimulus_grid(
    samples: list[tuple[dict, np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    title: str = "VSD vs rendered stimulus",
    ncol: int = 2,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        return output_path

    vsd_maps = [vsd for _, vsd, _ in samples]
    vsd_vmin = float(np.percentile(np.concatenate([m.ravel() for m in vsd_maps]), 1))
    vsd_vmax = float(np.percentile(np.concatenate([m.ravel() for m in vsd_maps]), 99))

    n = len(samples)
    fig, axes = plt.subplots(
        n, 2, figsize=(8.8, 3.2 * n), layout="constrained"
    )
    if n == 1:
        axes = np.array([axes])

    for row_idx, (meta, vsd_map, stimulus_rgb) in enumerate(samples):
        im = axes[row_idx, 0].imshow(
            vsd_map, cmap=VSD_CMAP, vmin=vsd_vmin, vmax=vsd_vmax
        )
        axes[row_idx, 0].set_ylabel(
            f"{meta['date']}\n{meta['condition']}", fontsize=9
        )
        if row_idx == 0:
            axes[row_idx, 0].set_title("VSD (H5 mean)", fontsize=10)

        axes[row_idx, 1].imshow(stimulus_rgb)
        if row_idx == 0:
            axes[row_idx, 1].set_title("Rendered stimulus", fontsize=10)
        axes[row_idx, 1].axis("off")

    fig.colorbar(
        im,
        ax=axes[:, 0].tolist(),
        fraction=0.018,
        pad=0.02,
        label="VSD signal",
    )
    fig.suptitle(title, fontsize=11)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
