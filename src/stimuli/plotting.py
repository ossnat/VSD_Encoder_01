"""QC plots for rendered stimuli."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_all_stimuli(
    entries: list[tuple[dict, np.ndarray]],
    output_dir: Path,
    *,
    cols: int = 4,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for meta, image in entries:
        out_path = output_dir / f"{meta['h5_session']}__{meta['condition']}.png"
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(image)
        ax.set_title(
            f"{meta['h5_session']} | {meta['condition']}\n{meta['stimulus_text']}",
            fontsize=8,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    if entries:
        n = len(entries)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (meta, image) in zip(axes.ravel(), entries):
            ax.imshow(image)
            ax.set_title(f"{meta['h5_session']} {meta['condition']}", fontsize=8)
        fig.suptitle("Rendered stimuli", fontsize=12)
        fig.tight_layout()
        grid_path = output_dir / "all_stimuli_grid.png"
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(grid_path)

    return written
