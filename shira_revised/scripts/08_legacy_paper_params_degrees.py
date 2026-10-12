#!/usr/bin/env python3
"""Legacy Schira with Ayzenshtat 2012 paper (a, α, k); axis labels in degrees.

Uses monkey-C / monkey-L fitted parameters from Ayzenshtat et al. 2012.
Paper form: ``P = α·θ`` (multiply). Display: fovea right, HM bottom (Fig. 2C).
Axis *tick positions* stay at the cortical loci of 1°…5°; only the labels
are degree strings (not OM indices).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLBACKEND", "Agg")

from shira_revised.src.legacy_schira import LegacySchiraParams, forward_legacy_schira
from shira_revised.src.legacy_trans_cortex import m1_m2_top_right_origin
from shira_revised.src.vf_top_right import detect_ppd_from_grid

# Ayzenshtat et al. 2012 fitted (a, k, α); k in mm.
PAPER_MONKEY_C = LegacySchiraParams(
    a=0.57, alpha=0.52, k=7.7, alpha_mode="multiply", vf_abs=True, ecc_width=0.76
)
PAPER_MONKEY_L = LegacySchiraParams(
    a=0.46, alpha=0.51, k=4.87, alpha_mode="multiply", vf_abs=True, ecc_width=0.76
)


def warp_forward_splat(
    rgb: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    params: LegacySchiraParams,
    *,
    grid: int = 560,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Push each VF pixel into Re(w)×Im(w) — silhouette = Schira image of the canvas."""
    u0, u1 = u_range
    v0, v1 = v_range
    xx, yy = np.meshgrid(m1, m2, indexing="xy")
    w = forward_legacy_schira(xx, yy, params)
    u = w.real
    v = w.imag
    ok = np.isfinite(u) & np.isfinite(v)

    span_u = max(u1 - u0, 1e-9)
    span_v = max(v1 - v0, 1e-9)
    uj = np.rint((u[ok] - u0) / span_u * (grid - 1)).astype(int)
    # row 0 = top = v1
    vi = np.rint((v1 - v[ok]) / span_v * (grid - 1)).astype(int)
    uj = np.clip(uj, 0, grid - 1)
    vi = np.clip(vi, 0, grid - 1)

    acc = np.zeros((grid, grid, 3), dtype=np.float64)
    wgt = np.zeros((grid, grid), dtype=np.float64)
    colors = rgb[ok].astype(np.float64)
    np.add.at(acc[:, :, 0], (vi, uj), colors[:, 0])
    np.add.at(acc[:, :, 1], (vi, uj), colors[:, 1])
    np.add.at(acc[:, :, 2], (vi, uj), colors[:, 2])
    np.add.at(wgt, (vi, uj), 1.0)

    out = np.full((grid, grid, 3), 255, dtype=np.uint8)
    hit = wgt > 0
    out[hit] = np.clip(acc[hit] / wgt[hit, None], 0, 255).astype(np.uint8)
    return out, (u0, u1), (v0, v1)


def cortical_uv_for_vf(
    x_deg: float, y_deg: float, params: LegacySchiraParams
) -> tuple[float, float]:
    w = complex(forward_legacy_schira(x_deg, y_deg, params))
    return float(w.real), float(w.imag)


def set_degree_axis_labels(
    ax,
    params: LegacySchiraParams,
    *,
    ecc_ticks: list[float],
    theta_ticks_deg: list[float],
    ecc_ref: float = 3.0,
) -> None:
    """Keep tick *positions* at cortical loci of those VF degrees; label in °."""
    # x: eccentricity along HM (y≈0+)
    u_ticks, u_labs = [], []
    for e in ecc_ticks:
        u, _ = cortical_uv_for_vf(e, 0.05, params)
        if np.isfinite(u):
            u_ticks.append(u)
            u_labs.append(f"{e:g}°")
    # y: polar angle at fixed ecc (paper-style θ labels); use abs VF y
    v_ticks, v_labs = [], []
    for th_deg in theta_ticks_deg:
        th = np.deg2rad(abs(th_deg))
        x = ecc_ref * np.cos(th)
        y = ecc_ref * np.sin(th)
        _, v = cortical_uv_for_vf(x, y, params)
        if np.isfinite(v):
            v_ticks.append(v)
            # signed catalog: lower field negative
            v_labs.append(f"{-abs(th_deg):g}°" if th_deg != 0 else "0°")

    if u_ticks:
        ax.set_xticks(u_ticks)
        ax.set_xticklabels(u_labs, fontsize=9)
    if v_ticks:
        ax.set_yticks(v_ticks)
        ax.set_yticklabels(v_labs, fontsize=9)
    ax.set_xlabel("eccentricity (deg VF)", fontsize=10)
    ax.set_ylabel("polar angle (deg VF)", fontsize=10)


def paper_view_window(
    params: LegacySchiraParams, *, ecc_max: float = 5.5
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Crop like Fig. 2C: fovea through ~5°, HM to LVM (avoid flat full-field box)."""
    u_fov, _ = cortical_uv_for_vf(0.02, 0.02, params)
    u_far, _ = cortical_uv_for_vf(ecc_max, 0.05, params)
    _, v_lvm = cortical_uv_for_vf(0.05, ecc_max, params)
    pad_u = 0.06 * abs(u_far - u_fov)
    pad_v = 0.06 * abs(v_lvm)
    return (u_fov - pad_u, u_far + pad_u), (0.0 - pad_v, float(v_lvm) + pad_v)


def overlay_cortical_polar_grid(
    ax,
    params: LegacySchiraParams,
    *,
    ecc_ticks: list[float],
    theta_deg: list[float] | None = None,
) -> None:
    """Draw Schira images of iso-E arcs and iso-θ rays (paper dashed grid)."""
    theta_deg = theta_deg or [0, 15, 30, 45, 60, 75, 90]
    # iso-eccentricity arcs
    for e in ecc_ticks:
        th = np.linspace(0.0, 0.5 * np.pi, 80)
        xs = e * np.cos(th)
        ys = e * np.sin(th)
        uu, vv = [], []
        for x, y in zip(xs, ys):
            u, v = cortical_uv_for_vf(float(x), float(y), params)
            if np.isfinite(u) and np.isfinite(v):
                uu.append(u)
                vv.append(v)
        if uu:
            ax.plot(uu, vv, color="0.25", lw=0.7, alpha=0.85, zorder=4)
    # iso-angle rays
    for th_d in theta_deg:
        th = np.deg2rad(th_d)
        rs = np.linspace(0.05, max(ecc_ticks) + 0.5, 60)
        uu, vv = [], []
        for r in rs:
            u, v = cortical_uv_for_vf(float(r * np.cos(th)), float(r * np.sin(th)), params)
            if np.isfinite(u) and np.isfinite(v):
                uu.append(u)
                vv.append(v)
        if uu:
            ax.plot(uu, vv, color="0.25", lw=0.7, alpha=0.85, zorder=4)


def draw_panel(
    ax,
    warped: np.ndarray,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    params: LegacySchiraParams,
    *,
    title: str,
    ecc_ticks: list[float],
) -> None:
    u0, u1 = u_range
    v0, v1 = v_range
    ax.imshow(
        warped,
        origin="upper",
        extent=[u0, u1, v0, v1],
        aspect="equal",
        interpolation="nearest",
    )
    (uw0, uw1), (vw0, vw1) = paper_view_window(params)
    ax.set_xlim(uw1, uw0)  # far E left, fovea right
    ax.set_ylim(vw1, vw0)  # LVM top, HM bottom
    ax.axhline(0.0, color="k", lw=1.4, zorder=3)
    overlay_cortical_polar_grid(ax, params, ecc_ticks=ecc_ticks)

    # Axis ticks stay at cortical loci of those degrees; labels = degree strings only
    set_degree_axis_labels(
        ax,
        params,
        ecc_ticks=ecc_ticks,
        theta_ticks_deg=[0, 30, 60, 90],
        ecc_ref=3.0,
    )
    # Paper-style rim labels on LVM (top) — same loci, not moved
    for e in ecc_ticks:
        u, v = cortical_uv_for_vf(0.08, e, params)
        if np.isfinite(u) and np.isfinite(v):
            ax.plot(u, v, "k+", markersize=8, markeredgewidth=1.2, zorder=6)
            ax.text(
                u,
                v,
                f"{e:g}°",
                color="k",
                fontsize=9,
                ha="center",
                va="bottom",
                zorder=6,
            )
    ax.set_title(title, fontsize=11)
    ax.text(0.98, 0.03, "HM", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.02, 0.97, "LVM", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(0.98, 0.55, "Fovea", transform=ax.transAxes, ha="right", va="center", fontsize=9, color="darkred")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--ppd", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--grid", type=int, default=520)
    ap.add_argument(
        "--paper-ref",
        type=Path,
        default=REPO
        / "shira_revised"
        / "artifacts"
        / "ayzenshtat_2012_fig2_reference.png",
    )
    args = ap.parse_args()

    out_dir = REPO / "shira_revised" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(out_dir / ".mplconfig")

    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    n_rows, n_cols = rgb.shape[:2]
    ppd = args.ppd or detect_ppd_from_grid(rgb.mean(2)) or 60.0
    m1, m2 = m1_m2_top_right_origin(n_rows, n_cols, ppd)
    ecc_ticks = [1.0, 2.0, 3.0, 4.0, 5.0]

    print(f"ppd={ppd:.1f}; warping with paper (a,α,k)…")
    win_c = paper_view_window(PAPER_MONKEY_C)
    win_l = paper_view_window(PAPER_MONKEY_L)
    warped_c, ur_c, vr_c = warp_forward_splat(
        rgb, m1, m2, PAPER_MONKEY_C, grid=args.grid, u_range=win_c[0], v_range=win_c[1]
    )
    warped_l, ur_l, vr_l = warp_forward_splat(
        rgb, m1, m2, PAPER_MONKEY_L, grid=args.grid, u_range=win_l[0], v_range=win_l[1]
    )

    n_cols_fig = 4 if args.paper_ref.is_file() else 3
    fig, axes = plt.subplots(1, n_cols_fig, figsize=(5.2 * n_cols_fig, 5.8))

    # A: stimulus with degree axes (top-right origin)
    x0, x1 = float(m1.min()), float(m1.max())
    y0, y1 = float(m2.min()), float(m2.max())
    axes[0].imshow(
        rgb,
        origin="upper",
        extent=[x0, x1, -y1, -y0],
        aspect="equal",
        interpolation="nearest",
    )
    axes[0].set_xlim(0, min(7.0, x1))
    axes[0].set_ylim(-min(8.0, y1), 0)
    axes[0].set_xticks(ecc_ticks)
    axes[0].set_xticklabels([f"{e:g}°" for e in ecc_ticks])
    axes[0].set_yticks([-e for e in ecc_ticks])
    axes[0].set_yticklabels([f"{-e:g}°" for e in ecc_ticks])
    axes[0].set_xlabel("HM (deg)")
    axes[0].set_ylabel("VM (deg)")
    axes[0].set_title("A  Visual space\n(fixation top-right)", fontsize=11)

    draw_panel(
        axes[1],
        warped_c,
        ur_c,
        vr_c,
        PAPER_MONKEY_C,
        title=f"B  Schira cortex · monkey C\na={PAPER_MONKEY_C.a}, k={PAPER_MONKEY_C.k}, α={PAPER_MONKEY_C.alpha} (P=αθ)",
        ecc_ticks=ecc_ticks,
    )
    draw_panel(
        axes[2],
        warped_l,
        ur_l,
        vr_l,
        PAPER_MONKEY_L,
        title=f"C  Schira cortex · monkey L\na={PAPER_MONKEY_L.a}, k={PAPER_MONKEY_L.k}, α={PAPER_MONKEY_L.alpha} (P=αθ)",
        ecc_ticks=ecc_ticks,
    )

    if args.paper_ref.is_file():
        ref = np.asarray(Image.open(args.paper_ref).convert("RGB"))
        axes[3].imshow(ref, origin="upper", aspect="equal")
        axes[3].set_title("D  Ayzenshtat 2012 Fig. 2\n(reference scan)", fontsize=11)
        axes[3].axis("off")

    fig.suptitle(
        "Legacy Schira algebra · Ayzenshtat 2012 (a, k, α) · multiply α · "
        "fovea RIGHT / HM BOTTOM · axis labels = VF degrees (tick loci = those °)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = args.out or (out_dir / f"legacy_paper_params_{args.image.stem}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
