#!/usr/bin/env python3
"""Legacy ``trans_cortex`` OM with degree ticks on the Schira-stretched grid.

The polar grid in the stimulus (iso-E arcs + iso-θ rays) is warped by Schira.
Those warped lines *are* the axis ticks:
  · pointy end (fovea) = 0° eccentricity
  · next arc = 1°, then 2°, …
  · HM ray = 0° polar; rays toward LVM = increasing |θ|

No extra landmark markers — ticks sit on the visible grid.
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

from shira_revised.src.legacy_schira import joanna_params
from shira_revised.src.legacy_trans_cortex import m1_m2_top_right_origin, trans_cortex
from shira_revised.src.vf_top_right import detect_ppd_from_grid

# Radial angles drawn on typical lab polar grids (incl. create_polar_stim.m).
GRID_THETA_DEG = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def _om_xy_at_vf(
    xcoord: np.ndarray,
    ycoord: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    x_deg: float,
    y_deg: float,
) -> tuple[float, float] | None:
    """Nearest stimulus pixel → OM (col, row) 0-based."""
    j = int(np.argmin(np.abs(m1 - x_deg)))
    i = int(np.argmin(np.abs(m2 - y_deg)))
    xc, yc = xcoord[i, j], ycoord[i, j]
    if not (np.isfinite(xc) and np.isfinite(yc)):
        return None
    return float(xc) - 1.0, float(yc) - 1.0


def grid_ecc_ticks_om(
    xcoord: np.ndarray,
    ycoord: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    eccs: list[float],
) -> list[tuple[float, float, float]]:
    """Iso-E arcs meet HM near y≈0 → (ecc, om_col, om_row). E=0 ≈ fovea."""
    out = []
    for e in eccs:
        # tiny y so we sit on the HM edge of the lower-field map
        y = 0.05 if e > 0 else 0.02
        x = max(e, 0.02)
        p = _om_xy_at_vf(xcoord, ycoord, m1, m2, x, y)
        if p is not None:
            out.append((e, p[0], p[1]))
    return out


def grid_theta_ticks_om(
    xcoord: np.ndarray,
    ycoord: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    thetas_deg: list[float],
    *,
    ecc_ref: float = 3.0,
) -> list[tuple[float, float, float]]:
    """Iso-θ rays at fixed eccentricity → (theta_deg, om_col, om_row)."""
    out = []
    for th in thetas_deg:
        rad = np.deg2rad(th)
        x = float(ecc_ref * np.cos(rad))
        y = float(ecc_ref * np.sin(rad))
        p = _om_xy_at_vf(xcoord, ycoord, m1, m2, x, y)
        if p is not None:
            out.append((th, p[0], p[1]))
    return out


def _set_om_degree_ticks(
    ax,
    *,
    ecc_marks: list[tuple[float, float, float]],
    theta_marks: list[tuple[float, float, float]],
    rot180: bool,
    om_w: int,
    om_h: int,
) -> None:
    """Place ticks on Schira grid lines; label in VF degrees."""

    def tx(c: float, r: float) -> tuple[float, float]:
        if rot180:
            return om_w - 1 - c, om_h - 1 - r
        return c, r

    xt, xl = [], []
    for e, c, r in ecc_marks:
        xx, _ = tx(c, r)
        xt.append(xx)
        xl.append(f"{e:g}°")
    yt, yl = [], []
    for th, c, r in theta_marks:
        _, yy = tx(c, r)
        yt.append(yy)
        # catalog lower field: θ from HM toward LVM is negative y
        yl.append("0°" if th == 0 else f"{-th:g}°")

    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=9)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=9)
    ax.set_xlabel("eccentricity (deg)  ·  iso-E grid arcs", fontsize=10)
    ax.set_ylabel("polar angle (deg)  ·  iso-θ grid rays", fontsize=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--ppd", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = REPO / "shira_revised" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(out_dir / ".mplconfig")

    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    n_rows, n_cols = rgb.shape[:2]
    ppd = args.ppd or detect_ppd_from_grid(rgb.mean(2)) or 60.0
    m1, m2 = m1_m2_top_right_origin(n_rows, n_cols, ppd)
    params = joanna_params()

    print(f"ppd={ppd:.1f}; running legacy trans_cortex…")
    result = trans_cortex(rgb, m1, m2, params, quick_mode=False, bg=0)
    om = result.om
    om_paper = np.rot90(om, 2)
    h, w = om.shape[:2]

    # Grid lines in the photo: E = 0,1,2,… and θ rays (same as stretched OM lines)
    eccs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ecc_marks = grid_ecc_ticks_om(result.xcoord, result.ycoord, m1, m2, eccs)
    theta_marks = grid_theta_ticks_om(
        result.xcoord, result.ycoord, m1, m2, GRID_THETA_DEG, ecc_ref=3.0
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- A ---
    axes[0].imshow(rgb, origin="upper", aspect="equal", interpolation="nearest")
    axes[0].set_title("A  Visual space\npolar grid = arcs (E) + rays (θ)", fontsize=12)
    xt_a = [n_cols - 1 - int(round(d * ppd)) for d in eccs if d > 0]
    yt_a = [int(round(d * ppd)) for d in eccs if d > 0]
    axes[0].set_xticks(xt_a)
    axes[0].set_xticklabels([f"{d:g}°" for d in eccs if d > 0])
    axes[0].set_yticks(yt_a)
    axes[0].set_yticklabels([f"{-d:g}°" for d in eccs if d > 0])
    axes[0].set_xlabel("x (deg) · concentric arcs")
    axes[0].set_ylabel("y (deg) · from fixation down")

    # --- B raw OM ---
    axes[1].imshow(om, origin="upper", aspect="equal", interpolation="nearest")
    axes[1].set_title(
        "B  Legacy OM (raw)\nfovea LEFT = 0° · HM TOP = 0° polar",
        fontsize=12,
    )
    _set_om_degree_ticks(
        axes[1],
        ecc_marks=ecc_marks,
        theta_marks=theta_marks,
        rot180=False,
        om_w=w,
        om_h=h,
    )

    # --- C paper orient ---
    axes[2].imshow(om_paper, origin="upper", aspect="equal", interpolation="nearest")
    axes[2].set_title(
        "C  Same OM, Fig. 2C orientation\n"
        "pointy end RIGHT = 0° · next arc = 1°, 2°, …",
        fontsize=12,
    )
    _set_om_degree_ticks(
        axes[2],
        ecc_marks=ecc_marks,
        theta_marks=theta_marks,
        rot180=True,
        om_w=w,
        om_h=h,
    )
    axes[2].text(0.98, 0.02, "HM (0°)", transform=axes[2].transAxes, ha="right", va="bottom", fontsize=11, fontweight="bold")
    axes[2].text(0.02, 0.98, "LVM (−90°)", transform=axes[2].transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    axes[2].text(0.98, 0.5, "0° E\n(fovea)", transform=axes[2].transAxes, ha="right", va="center", fontsize=10, color="darkred")

    fig.suptitle(
        "Legacy ``trans_cortex`` · Joanna a="
        f"{params.a}, α={params.alpha} (T=θ/α), k={params.k} · ppd={ppd:.0f}\n"
        "Axis ticks = Schira-stretched polar grid (iso-E arcs → x, iso-θ rays → y). "
        "No extra markers.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = args.out or (out_dir / f"legacy_vs_ayzenshtat_orient_{args.image.stem}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
