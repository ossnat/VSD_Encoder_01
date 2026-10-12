#!/usr/bin/env python3
"""Run revised (legacy) Schira on an image; fixation top-right; 1 tick = 1° VF."""

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
from shira_revised.src.legacy_warp import (
    legacy_folded_cortical_xy,
    legacy_trans_cortex_om,
)
from shira_revised.src.legacy_schira import forward_legacy_schira
from shira_revised.src.plot_vf import imshow_cortical
from shira_revised.src.vf_top_right import (
    detect_ppd_from_grid,
    imshow_input_vf_top_right,
    set_cortical_axes_vf_degrees,
    vf_grid_top_right,
)
from scipy.spatial import cKDTree


def folded_inverse_warp_fast(
    rgb: np.ndarray,
    x_deg: np.ndarray,
    y_deg: np.ndarray,
    params,
    *,
    grid: int = 500,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    n_rows, n_cols = rgb.shape[:2]
    w = forward_legacy_schira(x_deg, y_deg, params)
    cx, cy = legacy_folded_cortical_xy(w)
    ok = np.isfinite(cx) & np.isfinite(cy)
    pts = np.column_stack([cx[ok], cy[ok]])
    tree = cKDTree(pts)
    rows, cols = np.where(ok)
    pad = 0.08 * max(float(np.ptp(cx[ok])), float(np.ptp(cy[ok])), 0.05)
    x0, x1 = float(cx[ok].min()) - pad, float(cx[ok].max()) + pad
    y0, y1 = float(cy[ok].min()) - pad, float(cy[ok].max()) + pad
    u_c = np.linspace(x0, x1, grid, dtype=np.float64)
    v_c = np.linspace(y1, y0, grid, dtype=np.float64)
    uu, vv = np.meshgrid(u_c, v_c, indexing="xy")
    q = np.column_stack([uu.ravel(), vv.ravel()])
    _, idx = tree.query(q, k=1)
    warped = rgb[rows[idx], cols[idx]].reshape(grid, grid, 3)
    return warped, (x0, x1, y0, y1)


def overlay_vf_grid_on_cortical(
    ax,
    params,
    *,
    x_max: float,
    y_max: float,
    step: float = 1.0,
    color: str = "k",
    alpha: float = 0.35,
) -> None:
    """Draw iso-x and iso-y VF lines (1°) in folded cortical space."""
    ys = np.linspace(0.0, y_max, int(y_max / step) + 1)
    xs = np.linspace(0.0, x_max, int(x_max / step) + 1)
    for x in xs:
        cy_list = []
        cx_list = []
        for y in np.linspace(0.0, y_max, 200):
            w = forward_legacy_schira(x, y, params)
            if np.isfinite(w):
                cx, cy = legacy_folded_cortical_xy(w)
                cx_list.append(cx)
                cy_list.append(cy)
        if cx_list:
            ax.plot(cx_list, cy_list, color=color, alpha=alpha, linewidth=0.6)
    for y in ys:
        cx_list = []
        cy_list = []
        for x in np.linspace(0.0, x_max, 200):
            w = forward_legacy_schira(x, y, params)
            if np.isfinite(w):
                cx, cy = legacy_folded_cortical_xy(w)
                cx_list.append(cx)
                cy_list.append(cy)
        if cx_list:
            ax.plot(cx_list, cy_list, color=color, alpha=alpha, linewidth=0.6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument(
        "--ppd",
        type=float,
        default=None,
        help="pixels/deg (default: detect 1° grid from top-right axes)",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--grid", type=int, default=520)
    ap.add_argument("--x-max", type=float, default=None, help="VF x extent for ticks")
    ap.add_argument("--y-max", type=float, default=None, help="VF y extent for ticks")
    args = ap.parse_args()

    out_dir = REPO / "shira_revised" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(out_dir / ".mplconfig")

    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    n_rows, n_cols = rgb.shape[:2]
    gray = rgb.mean(axis=2)

    ppd = args.ppd
    if ppd is None:
        ppd = detect_ppd_from_grid(gray)
    if ppd is None or ppd <= 0:
        ppd = 60.0
        print(f"warning: could not detect grid; using ppd={ppd}", file=sys.stderr)
    else:
        print(f"detected ppd={ppd:.2f} px/deg (1 tick = 1°)")

    m1, m2, x_deg, y_deg = vf_grid_top_right(n_rows, n_cols, ppd)
    x_max = args.x_max if args.x_max is not None else float(np.max(m1))
    y_max = args.y_max if args.y_max is not None else float(np.max(m2))

    params = joanna_params()
    om, _ = legacy_trans_cortex_om(rgb, x_deg, y_deg, params, ink_threshold=40.0)
    smooth, smooth_ext = folded_inverse_warp_fast(
        rgb, x_deg, y_deg, params, grid=args.grid
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    imshow_input_vf_top_right(
        axes[0],
        rgb,
        ppd,
        title="Input (fixation top-right; monkey ≈ x 0–3°, y −2 to −6°)",
    )
    axes[0].set_xlim(0.0, float(x_max))
    axes[0].set_ylim(-float(y_max), 0.0)

    axes[1].imshow(om, origin="upper", aspect="equal", interpolation="nearest")
    axes[1].set_title("Legacy Schira OM (trans_cortex)", fontsize=11)
    axes[1].set_xlabel("OM column (index)")
    axes[1].set_ylabel("OM row (index)")

    imshow_cortical(
        axes[2],
        smooth,
        smooth_ext,
        title="Schira cortex (VF degree ticks on axes)",
        xlabel="x (deg VF)",
        ylabel="y (deg VF, + up)",
    )
    overlay_vf_grid_on_cortical(
        axes[2], params, x_max=x_max, y_max=y_max, step=1.0, alpha=0.4
    )
    set_cortical_axes_vf_degrees(
        axes[2],
        params,
        x_max=x_max,
        y_max=y_max,
        y_ref=4.0,
        x_ref=1.5,
        tick_deg=1.0,
    )

    fig.suptitle(
        f"Revised Schira  a={params.a}, α={params.alpha} (T=θ/α), k={params.k}  |  "
        f"ppd={ppd:.1f} (1 px tick = 1°)  |  "
        f"VF extent to x≈{x_max:.0f}°, y≈{y_max:.0f}° down",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.out or (out_dir / f"schira_{args.image.stem}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(
        f"VF check: y=2° → row {2*ppd:.0f}, y=6° → row {6*ppd:.0f}; "
        f"x=0° → col {n_cols-1}, x=3° → col {n_cols-1-3*ppd:.0f}"
    )


if __name__ == "__main__":
    main()
