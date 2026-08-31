"""Schira-only cortical plots and VSD overlay helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

from src.plotting_colormaps import VSD_CMAP


def _ink_rgba_overlay(
    ink: np.ndarray,
    *,
    outline_iters: int = 1,
) -> np.ndarray:
    """Black fill + light outline so glyphs stay visible on red VSD hotspots."""
    ink_bool = np.asarray(ink, dtype=bool)
    overlay = np.zeros((*ink_bool.shape, 4), dtype=np.float32)
    if outline_iters > 0 and np.any(ink_bool):
        rim = binary_dilation(ink_bool, iterations=int(outline_iters)) & ~ink_bool
        overlay[rim] = (1.0, 1.0, 1.0, 0.95)
    overlay[ink_bool] = (0.0, 0.0, 0.0, 0.95)
    return overlay


def plot_schira_only(
    forward_u: np.ndarray,
    forward_v: np.ndarray,
    *,
    output_path: Path,
    title: str,
    param_note: str | None = None,
    margin_frac: float = 0.15,
    point_size: float = 4.0,
) -> Path:
    """Plot Schira-mapped **ink pixels only** on Cartesian cortical ``(u, v)``.

    Only black (stimulus) pixels are forward-mapped; gray background is not
    transformed. ``u = Re(w)``, ``v = Im(w)`` are Cartesian model-cortex axes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fu = np.asarray(forward_u, dtype=np.float64).ravel()
    fv = np.asarray(forward_v, dtype=np.float64).ravel()
    ok = np.isfinite(fu) & np.isfinite(fv)
    fu, fv = fu[ok], fv[ok]
    if fu.size == 0:
        raise ValueError("No finite ink points to plot after Schira forward map")

    u0, u1 = float(fu.min()), float(fu.max())
    v0, v1 = float(fv.min()), float(fv.max())
    span = max(u1 - u0, v1 - v0, 1e-3)
    pad = float(margin_frac) * span
    u_lo, u_hi = u0 - pad, u1 + pad
    v_lo, v_hi = v0 - pad, v1 + pad

    fig, ax = plt.subplots(figsize=(5.4, 5.2), layout="constrained")
    ax.set_facecolor("white")
    ax.scatter(fu, fv, s=point_size, c="black", marker="s", linewidths=0, alpha=1.0)
    ax.set_xlim(u_lo, u_hi)
    ax.set_ylim(v_lo, v_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("u (Cartesian cortical)", fontsize=11)
    ax.set_ylabel("v (Cartesian cortical)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.axhline(0.0, color="0.7", lw=0.7, ls="--")
    ax.axvline(0.0, color="0.7", lw=0.7, ls="--")
    if param_note:
        fig.text(0.5, 0.01, param_note, ha="center", fontsize=7, color="0.35")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_schira_only_montage(
    panels: list[tuple[np.ndarray, np.ndarray, str]],
    output_path: Path,
    *,
    title: str,
    ncol: int = 3,
    point_size: float = 2.0,
    margin_frac: float = 0.15,
) -> Path:
    """Grid of ink-only Schira forward maps (Cartesian ``u,v``)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not panels:
        return output_path
    n = len(panels)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow), layout="constrained"
    )
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr:
        ax.axis("off")
    for ax, (fu, fv, label) in zip(axes_arr, panels):
        fu = np.asarray(fu, dtype=np.float64).ravel()
        fv = np.asarray(fv, dtype=np.float64).ravel()
        ok = np.isfinite(fu) & np.isfinite(fv)
        fu, fv = fu[ok], fv[ok]
        ax.set_facecolor("white")
        ax.axis("on")
        if fu.size:
            ax.scatter(fu, fv, s=point_size, c="black", marker="s", linewidths=0)
            span = max(fu.max() - fu.min(), fv.max() - fv.min(), 1e-3)
            pad = float(margin_frac) * span
            ax.set_xlim(fu.min() - pad, fu.max() + pad)
            ax.set_ylim(fv.min() - pad, fv.max() + pad)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("u", fontsize=8)
        ax.set_ylabel("v", fontsize=8)
        ax.axhline(0.0, color="0.75", lw=0.5, ls="--")
        ax.axvline(0.0, color="0.75", lw=0.5, ls="--")
    fig.suptitle(title, fontsize=11)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_vsd_with_warped_stimulus(
    vsd_map: np.ndarray,
    stimulus_rgb: np.ndarray,
    warped_rgb: np.ndarray | None,
    ink: np.ndarray,
    output_path: Path,
    *,
    title: str,
    vsd_clim: tuple[float, float] | None = None,
    affine_note: str | None = None,
    outline_iters: int = 0,
    overlay_title: str = "VSD + warped stimulus",
) -> Path:
    """Three-panel figure: stimulus | raw VSD | overlay (black ink).

    ``warped_rgb`` is optional (kept for API compatibility); the overlay is
    drawn from ``ink`` only. Default ``outline_iters=0`` matches thesis Fig. 13
    thin-line style (no white halo).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vsd = np.asarray(vsd_map, dtype=np.float64)
    if vsd_clim is None:
        finite = vsd[np.isfinite(vsd)]
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99])
        else:
            lo, hi = 0.0, 1.0
        if lo == hi:
            hi = lo + 1e-6
        vsd_clim = (float(lo), float(hi))

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), layout="constrained")
    axes[0].imshow(np.asarray(stimulus_rgb))
    axes[0].set_title("Stimulus (visual field)", fontsize=10)
    axes[0].axis("off")

    im = axes[1].imshow(vsd, cmap=VSD_CMAP, vmin=vsd_clim[0], vmax=vsd_clim[1])
    axes[1].set_title("Raw VSD mean", fontsize=10)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(vsd, cmap=VSD_CMAP, vmin=vsd_clim[0], vmax=vsd_clim[1])
    axes[2].imshow(_ink_rgba_overlay(ink, outline_iters=outline_iters))
    axes[2].set_title(overlay_title, fontsize=10)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=10)
    if affine_note:
        fig.text(0.5, 0.01, affine_note, ha="center", fontsize=7, color="0.35")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_overlay_montage(
    panels: list[tuple[np.ndarray, np.ndarray, str]],
    output_path: Path,
    *,
    title: str,
    vsd_clim: tuple[float, float] | None = None,
    ncol: int = 4,
    outline_iters: int = 0,
) -> Path:
    """Grid of VSD+black-ink overlays (thin ink by default)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not panels:
        return output_path
    if vsd_clim is None:
        stacked = np.concatenate([v.ravel() for v, _, _ in panels])
        finite = stacked[np.isfinite(stacked)]
        lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
        vsd_clim = (float(lo), float(hi))
    n = len(panels)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.1 * ncol, 3.1 * nrow), layout="constrained"
    )
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr:
        ax.axis("off")
    for ax, (vsd, ink, label) in zip(axes_arr, panels):
        ax.imshow(vsd, cmap=VSD_CMAP, vmin=vsd_clim[0], vmax=vsd_clim[1])
        ax.imshow(_ink_rgba_overlay(ink, outline_iters=outline_iters))
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path
