"""Plotting helpers for the channel×space (CM) separable encoder."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from matplotlib.colors import TwoSlopeNorm


def upsample_map(
    m: np.ndarray,
    out_hw: tuple[int, int],
) -> np.ndarray:
    """Bilinear upsample a 2D map to ``(H, W)`` via torch (if available) or PIL."""
    h, w = out_hw
    m = np.asarray(m, dtype=np.float32)
    if m.shape == (h, w):
        return m
    try:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
        out = (
            F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
            .squeeze()
            .numpy()
        )
        return out.astype(np.float32, copy=False)
    except Exception:
        # Fallback: PIL on normalized uint8 loses precision; use float via scipy.
        from scipy.ndimage import zoom

        zh = h / m.shape[0]
        zw = w / m.shape[1]
        return zoom(m, (zh, zw), order=1).astype(np.float32)


def plot_channel_importance(
    a: np.ndarray,
    output_path: Path,
    *,
    title: str = "Channel weights a",
    top_k: int | None = None,
) -> Path:
    """Bar plot of shared channel vector ``a`` (optionally top-|a| only)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(a, dtype=np.float64).ravel()
    idx = np.arange(a.size)
    if top_k is not None and top_k < a.size:
        order = np.argsort(np.abs(a))[::-1][:top_k]
        a = a[order]
        idx = order
        title = f"{title} (top {top_k} by |a|)"

    fig, ax = plt.subplots(figsize=(max(6.0, 0.08 * len(a)), 3.5))
    colors = np.where(a >= 0, "steelblue", "darkorange")
    ax.bar(np.arange(len(a)), a, color=colors, width=0.85)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("channel index" if top_k is None else "channel (ranked)")
    ax.set_ylabel("a_c")
    ax.set_title(title, fontsize=11)
    if top_k is not None:
        ax.set_xticks(np.arange(len(a)))
        ax.set_xticklabels([str(int(i)) for i in idx], rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_M_overlay_on_stimulus(
    M_map: np.ndarray,
    stimulus_rgb: np.ndarray,
    output_path: Path,
    *,
    title: str,
    overlay_alpha: float = 0.55,
    diverging: bool = True,
    cmap: str = "RdBu_r",
) -> Path:
    """
    Upsample ``M_map`` (feature-space H×W) to stimulus resolution and overlay.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stim = np.asarray(stimulus_rgb)
    if stim.ndim == 2:
        stim = np.stack([stim] * 3, axis=-1)
    h, w = stim.shape[:2]
    up = upsample_map(M_map, (h, w))
    finite = up[np.isfinite(up)]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    axes[0].imshow(stim)
    axes[0].set_title("Stimulus", fontsize=10)
    axes[0].axis("off")

    if diverging and finite.size:
        vmax = float(np.percentile(np.abs(finite), 99))
        vmax = max(vmax, 1e-12)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im1 = axes[1].imshow(up, cmap=cmap, norm=norm)
        im2_kwargs: dict = {"cmap": cmap, "norm": norm, "alpha": overlay_alpha}
    else:
        vmin = float(np.percentile(finite, 1)) if finite.size else 0.0
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
        heat_cmap = cmap
        im1 = axes[1].imshow(up, cmap=heat_cmap, vmin=vmin, vmax=vmax)
        im2_kwargs = {
            "cmap": heat_cmap,
            "vmin": vmin,
            "vmax": vmax,
            "alpha": overlay_alpha,
        }
    axes[1].set_title("M (upsampled)", fontsize=10)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(stim)
    im2 = axes[2].imshow(up, **im2_kwargs)
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=11)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_stimulus_rgb(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
