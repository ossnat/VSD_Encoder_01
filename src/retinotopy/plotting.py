"""Schira-only cortical plots and VSD overlay helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

from src.plotting_colormaps import VSD_CMAP
from src.retinotopy.schira import SchiraParams, forward_schira
from src.retinotopy.visual_field import stimulus_ink_mask
from src.stimuli.render import RenderConfig


def draw_vf_cartesian_deg(
    ax,
    stimulus_rgb: np.ndarray,
    cfg: RenderConfig,
) -> None:
    """Show the VF image with Cartesian degree axes (fixation top-left)."""
    extent = float(cfg.quadrant_extent_deg)
    ax.imshow(
        np.asarray(stimulus_rgb),
        origin="upper",
        extent=[0.0, extent, -extent, 0.0],
        interpolation="nearest",
    )
    ax.set_xlabel("x (deg)", fontsize=9)
    ax.set_ylabel("y (deg)", fontsize=9)
    ax.set_xlim(0.0, extent)
    ax.set_ylim(-extent, 0.0)
    ax.set_aspect("equal")


def apply_vf_hm_vm_degree_ticks(
    ax,
    cfg: RenderConfig,
    *,
    step_deg: float = 1.0,
) -> None:
    """Label VF axes as HM / VM with 1° ticks.

    The image extent is already in degrees (``1° = pixels_per_deg`` px, 35 on
    the lab canvas), so tick spacing is ``step_deg`` in those units.
    """
    step = float(step_deg)
    if float(cfg.pixels_per_deg) <= 0:
        raise ValueError("pixels_per_deg must be positive (lab scale: 35 px = 1°)")
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    xt = np.arange(np.ceil(x0 - 1e-9), x1 + 1e-9, step)
    yt = -np.arange(0.0, -y0 + 1e-9, step)
    xt = xt[(xt >= x0 - 1e-9) & (xt <= x1 + 1e-9)]
    yt = yt[(yt >= y0 - 1e-9) & (yt <= y1 + 1e-9)]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t:g}°" for t in xt], fontsize=8)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{t:g}°" for t in yt], fontsize=8)
    ax.set_xlabel("HM", fontsize=9)
    ax.set_ylabel("VM", fontsize=9)


def draw_vf_polar_overlay(
    ax,
    stimulus_rgb: np.ndarray,
    cfg: RenderConfig,
    *,
    ecc_spacing_deg: float = 1.0,
    theta_step_deg: float = 15.0,
) -> None:
    """Same static image as the Cartesian panel; polar ``(E, θ)`` is overlay only.

    Pixel values are not resampled. Iso-E arcs (red) and iso-θ rays (blue) are
    the polar labels of the same ``(x, y)`` grid. Schira consumes those numbers,
    not a new polar raster.
    """
    draw_vf_cartesian_deg(ax, stimulus_rgb, cfg)
    extent = float(cfg.quadrant_extent_deg)
    th_arc = np.linspace(0.0, -0.5 * np.pi, 256)
    for ecc in np.arange(ecc_spacing_deg, extent * np.sqrt(2.0) + 1e-9, ecc_spacing_deg):
        x = ecc * np.cos(th_arc)
        y = ecc * np.sin(th_arc)
        inside = (x >= 0.0) & (x <= extent) & (y <= 0.0) & (y >= -extent)
        if np.any(inside):
            ax.plot(x[inside], y[inside], color="#c0392b", lw=0.7, alpha=0.8)
    for th_deg in np.arange(0.0, -90.01, -theta_step_deg):
        th = np.deg2rad(th_deg)
        r_max = extent / max(abs(np.cos(th)), abs(np.sin(th)), 1e-12)
        r = np.linspace(0.0, r_max, 64)
        ax.plot(r * np.cos(th), r * np.sin(th), color="#2471a3", lw=0.7, alpha=0.8)
    ax.set_xlabel("x (deg)  ·  overlay: E red, θ blue", fontsize=8)


def set_schira_axes_visual_deg(
    ax,
    params: SchiraParams,
    *,
    ecc_ticks_deg: np.ndarray | list[float],
    theta_ticks_deg: np.ndarray | list[float],
    ecc_ref_deg: float,
) -> None:
    """Re-index cortical ``(u, v)`` ticks by visual-field degrees.

    Tick *labels* are VF coordinates; data coordinates stay ``(u, v)``.
    Axis direction (fovea left vs right, HM top vs bottom) is whatever
    ``xlim`` / ``ylim`` already are.

    * **u** → eccentricity ``E`` along the HM
    * **v** → polar angle ``θ`` at ``ecc_ref_deg``
    """
    u_lo, u_hi = ax.get_xlim()
    v_lo, v_hi = ax.get_ylim()
    ecc = np.asarray(ecc_ticks_deg, dtype=np.float64)
    u_ticks = np.real(forward_schira(ecc, np.zeros_like(ecc), params))
    keep_u = np.isfinite(u_ticks) & (u_ticks >= min(u_lo, u_hi)) & (
        u_ticks <= max(u_lo, u_hi)
    )
    if np.any(keep_u):
        ax.set_xticks(u_ticks[keep_u])
        ax.set_xticklabels([f"{val:g}°" for val in ecc[keep_u]], fontsize=8)
    th = np.asarray(theta_ticks_deg, dtype=np.float64)
    v_ticks = np.imag(
        forward_schira(np.full(th.shape, float(ecc_ref_deg)), np.deg2rad(th), params)
    )
    keep_v = np.isfinite(v_ticks) & (v_ticks >= min(v_lo, v_hi)) & (
        v_ticks <= max(v_lo, v_hi)
    )
    if np.any(keep_v):
        ax.set_yticks(v_ticks[keep_v])
        ax.set_yticklabels([f"{val:g}°" for val in th[keep_v]], fontsize=8)
    ax.set_xlabel("eccentricity $E$ (deg)", fontsize=9)
    ax.set_ylabel(r"polar angle $\theta$ (deg)", fontsize=9)


def draw_schira_ayzenshtat_c(
    ax,
    warped_rgb: np.ndarray,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    params: SchiraParams,
    *,
    ecc_ticks_deg: np.ndarray | list[float],
    theta_ticks_deg: np.ndarray | list[float],
    ecc_ref_deg: float,
    fovea_on_right: bool = True,
    hm_on_bottom: bool = True,
) -> None:
    """Draw warped RGB in cortical ``(u, v)`` with degree tick labels.

    Mapping is unchanged (``u = Re(w)``, ``v = Im(w)``). Defaults match
    Ayzenshtat Fig. 2C: **fovea on the right**, **HM on the bottom**.
    For a log-polar reading of the lower-right VF, pass both flags ``False``
    (small ``E`` on the left, ``v = 0`` at the top, like the VF panel).
    """
    u0, u1 = float(u_range[0]), float(u_range[1])
    v0, v1 = float(v_range[0]), float(v_range[1])
    ax.imshow(
        np.asarray(warped_rgb),
        origin="upper",
        extent=[u0, u1, v0, v1],
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_xlim(u1, u0) if fovea_on_right else ax.set_xlim(u0, u1)
    ax.set_ylim(v1, v0) if hm_on_bottom else ax.set_ylim(v0, v1)
    ax.axhline(0.0, color="black", lw=1.6, zorder=3)
    set_schira_axes_visual_deg(
        ax,
        params,
        ecc_ticks_deg=ecc_ticks_deg,
        theta_ticks_deg=theta_ticks_deg,
        ecc_ref_deg=ecc_ref_deg,
    )
    u_left, u_right = ax.get_xlim()
    v_bottom, v_top = ax.get_ylim()
    u_hm = 0.5 * (u_left + u_right)
    # Just inside the map from the HM (v = 0).
    hm_at_bottom = abs(v_bottom) <= abs(v_top)
    v_edge = v_bottom if hm_at_bottom else v_top
    v_interior = v_top if hm_at_bottom else v_bottom
    v_hm = v_edge + 0.04 * (v_interior - v_edge)
    ax.text(
        u_hm,
        v_hm,
        "HM",
        ha="center",
        va="bottom" if hm_at_bottom else "top",
        fontsize=9,
        fontweight="bold",
        color="black",
        zorder=4,
        clip_on=False,
    )


def overlay_schira_polar_grid(
    ax,
    params: SchiraParams,
    *,
    ecc_deg: np.ndarray | list[float],
    theta_deg: np.ndarray | list[float],
    ecc_color: str = "#f4d03f",
    theta_color: str = "#5dade2",
    lw: float = 0.95,
) -> None:
    """Draw iso-E (yellow) and iso-θ (blue) curves in cortical ``(u, v)``."""
    eccs = np.asarray(ecc_deg, dtype=np.float64)
    thetas = np.asarray(theta_deg, dtype=np.float64)
    th_fine = np.deg2rad(
        np.linspace(float(np.min(thetas)), float(np.max(thetas)), 256)
    )
    e_max = float(np.max(eccs))
    e_fine = np.linspace(max(0.05, 0.25 * float(np.min(eccs))), e_max, 256)

    def _polyline(w: np.ndarray, color: str) -> None:
        u = np.asarray(w.real, dtype=np.float64)
        v = np.asarray(w.imag, dtype=np.float64)
        ok = np.isfinite(u) & np.isfinite(v)
        if not np.any(ok):
            return
        idx = np.flatnonzero(ok)
        cuts = np.flatnonzero(np.diff(idx) > 1) + 1
        for sl in np.split(idx, cuts):
            if sl.size > 1:
                ax.plot(u[sl], v[sl], color=color, lw=lw, zorder=5, solid_capstyle="round")

    for e in eccs:
        _polyline(forward_schira(np.full_like(th_fine, float(e)), th_fine, params), ecc_color)
    for th in thetas:
        _polyline(
            forward_schira(e_fine, np.full_like(e_fine, np.deg2rad(float(th))), params),
            theta_color,
        )


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


def ink_rgba_overlay(
    ink: np.ndarray,
    *,
    outline_iters: int = 0,
    alpha: float = 0.55,
    rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Semi-transparent ink overlay (default: see VSD through the stroke)."""
    ink_bool = np.asarray(ink, dtype=bool)
    overlay = np.zeros((*ink_bool.shape, 4), dtype=np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    if outline_iters > 0 and np.any(ink_bool):
        rim = binary_dilation(ink_bool, iterations=int(outline_iters)) & ~ink_bool
        overlay[rim] = (1.0, 1.0, 1.0, a)
    overlay[ink_bool] = (rgb[0], rgb[1], rgb[2], a)
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


def plot_schira_cortex_comparison(
    stimulus_rgb: np.ndarray,
    forward_u: np.ndarray,
    forward_v: np.ndarray,
    warped_rgb: np.ndarray | None = None,
    u_range: tuple[float, float] | None = None,
    v_range: tuple[float, float] | None = None,
    *,
    output_path: Path,
    title: str,
    param_note: str | None = None,
    margin_frac: float = 0.12,
    point_size: float = 3.0,
    show_inverse: bool = True,
) -> Path:
    """VF stimulus | forward ink scatter | optional inverse-warp on ``(u, v)``.

    Cortical panels are zoomed to the forward ink cloud (plus ``margin_frac``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fu = np.asarray(forward_u, dtype=np.float64).ravel()
    fv = np.asarray(forward_v, dtype=np.float64).ravel()
    ok = np.isfinite(fu) & np.isfinite(fv)
    fu, fv = fu[ok], fv[ok]
    if fu.size == 0:
        raise ValueError("No finite ink points for forward scatter panel")

    span = max(float(fu.max() - fu.min()), float(fv.max() - fv.min()), 1e-3)
    pad = float(margin_frac) * span
    u_lo, u_hi = float(fu.min() - pad), float(fu.max() + pad)
    v_lo, v_hi = float(fv.min() - pad), float(fv.max() + pad)

    n_panels = 3 if show_inverse else 2
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.0 * n_panels + 0.6, 4.2), layout="constrained"
    )
    axes[0].imshow(np.asarray(stimulus_rgb))
    axes[0].set_title("Visual field (rendered)", fontsize=10)
    axes[0].axis("off")

    bg = float(np.median(np.asarray(stimulus_rgb))) / 255.0
    axes[1].set_facecolor((bg, bg, bg))
    axes[1].scatter(fu, fv, s=point_size, c="black", marker="s", linewidths=0)
    axes[1].set_xlim(u_lo, u_hi)
    axes[1].set_ylim(v_lo, v_hi)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("u", fontsize=9)
    axes[1].set_ylabel("v", fontsize=9)
    axes[1].set_title("Forward ink → (u, v)", fontsize=10)
    axes[1].axhline(0.0, color="0.75", lw=0.5, ls="--")
    axes[1].axvline(0.0, color="0.75", lw=0.5, ls="--")

    if show_inverse:
        if warped_rgb is None or u_range is None or v_range is None:
            raise ValueError("Inverse panel needs warped_rgb, u_range, and v_range")
        u0, u1 = float(u_range[0]), float(u_range[1])
        v0, v1 = float(v_range[0]), float(v_range[1])
        axes[2].imshow(
            np.asarray(warped_rgb),
            origin="upper",
            extent=[u0, u1, v0, v1],
            interpolation="nearest",
            aspect="equal",
        )
        axes[2].set_xlim(u_lo, u_hi)
        axes[2].set_ylim(v_lo, v_hi)
        axes[2].axhline(0.0, color="0.75", lw=0.5, ls="--")
        axes[2].axvline(0.0, color="0.75", lw=0.5, ls="--")
        axes[2].set_xlabel("u (Cartesian cortical)", fontsize=9)
        axes[2].set_ylabel("v (Cartesian cortical)", fontsize=9)
        axes[2].set_title("Inverse warp on (u, v) grid", fontsize=10)

    fig.suptitle(title, fontsize=11)
    if param_note:
        fig.text(0.5, 0.01, param_note, ha="center", fontsize=7, color="0.35")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
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
