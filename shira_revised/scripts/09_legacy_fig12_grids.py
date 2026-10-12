#!/usr/bin/env python3
"""Thesis Fig. 12 style: legacy cart/polar grids · two session parameter sets.

Fovea always top-left in VF and LEFT in OM (no rot180).
Difference between columns is Schira (a, α, k), matching Joanna:

  20/11/18 (201118): a=0.72, alpha=1.5, k=1
  10/07/18 (100718): a=0.74, alpha=1.0, k=1
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

from shira_revised.src.legacy_schira import LegacySchiraParams
from shira_revised.src.legacy_stimuli import PPD, create_cart_stim, create_polar_stim
from shira_revised.src.legacy_stimuli import BG, LEGACY_COLS, LEGACY_ROWS, _gray_rgb
from shira_revised.src.legacy_trans_cortex import trans_cortex

THESIS_FIG12 = (
    REPO / "shira_revised" / "artifacts" / "thesis_fig12" / "thesis_page_030.png"
)

# Joanna thesis optimal parameters (session dates)
PARAMS_201118 = LegacySchiraParams(
    a=0.72, alpha=1.5, k=1.0, alpha_mode="divide", vf_abs=True
)
PARAMS_100718 = LegacySchiraParams(
    a=0.74, alpha=1.0, k=1.0, alpha_mode="divide", vf_abs=True
)


def m1_m2_top_left(n_rows: int, n_cols: int, ppd: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixation at top-left (thesis / Joanna grid convention)."""
    m1 = np.arange(n_cols, dtype=np.float64) / ppd
    m2 = np.arange(n_rows, dtype=np.float64) / ppd
    return m1, m2


def crop_vf_square(rgb: np.ndarray, ppd: float, extent_deg: float = 6.0) -> np.ndarray:
    n = int(round(extent_deg * ppd))
    return rgb[:n, :n].copy()


def make_cart(ppd: float) -> np.ndarray:
    if abs(ppd - PPD) < 1e-9:
        return create_cart_stim()
    stim = np.full((LEGACY_ROWS, LEGACY_COLS), BG, dtype=np.float64)
    step = int(round(ppd))
    for rad in range(0, LEGACY_ROWS, step):
        r = 0 if rad == 0 else rad
        stim[r, :] = 255
        stim[:, r] = 255
    half = max(step // 2, 1)
    for hrad in range(half, LEGACY_ROWS, step):
        stim[hrad, :] = 255
        stim[:, hrad] = 255
    return _gray_rgb(stim)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppd", type=float, default=float(PPD))
    ap.add_argument("--extent-deg", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = REPO / "shira_revised" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(out_dir / ".mplconfig")

    ppd = float(args.ppd)
    extent = float(args.extent_deg)

    polar = create_polar_stim(ppd=int(ppd), is_left_hem=True, extent_deg=extent)
    if abs(ppd - PPD) < 1e-9:
        cart = create_cart_stim(extent_deg=extent, ppd=int(ppd))
    else:
        cart = crop_vf_square(make_cart(ppd), ppd, extent)
        cart[-1, :] = 255
        cart[:, -1] = 255
    n = polar.shape[0]
    m1, m2 = m1_m2_top_left(n, n, ppd)

    print(
        f"{n}×{n} @ {ppd} ppd · "
        f"201118 a={PARAMS_201118.a} α={PARAMS_201118.alpha} · "
        f"100718 a={PARAMS_100718.a} α={PARAMS_100718.alpha}"
    )
    polar_201118 = trans_cortex(polar, m1, m2, PARAMS_201118, quick_mode=False, bg=0).om
    polar_100718 = trans_cortex(polar, m1, m2, PARAMS_100718, quick_mode=False, bg=0).om
    cart_201118 = trans_cortex(cart, m1, m2, PARAMS_201118, quick_mode=False, bg=0).om
    cart_100718 = trans_cortex(cart, m1, m2, PARAMS_100718, quick_mode=False, bg=0).om

    has_thesis = THESIS_FIG12.is_file()
    nrows = 3 if has_thesis else 2
    fig = plt.figure(figsize=(15.5, 4.4 * nrows))
    gs = fig.add_gridspec(nrows, 3, hspace=0.38, wspace=0.28)

    def show_vf(ax, rgb, title):
        ax.imshow(rgb, origin="upper", aspect="equal", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ticks = np.arange(0, extent + 1e-9, 1.0)
        ax.set_xticks(ticks * ppd)
        ax.set_xticklabels([f"{t:g}°" for t in ticks])
        ax.set_yticks(ticks * ppd)
        ax.set_yticklabels([f"{-t:g}°" for t in ticks])
        ax.set_xlabel("x (deg VF)")
        ax.set_ylabel("y (deg VF, + up)")
        ax.text(
            0.02,
            0.98,
            "fovea",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="darkred",
            fontweight="bold",
        )

    def show_om(ax, om, title):
        ax.imshow(om, origin="upper", aspect="equal", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("OM column (fovea ← LEFT)")
        ax.set_ylabel("OM row (HM ← TOP)")

    # Row A — polar
    show_vf(
        fig.add_subplot(gs[0, 0]),
        polar,
        "A(i)  Polar VF · fovea top-left",
    )
    show_om(
        fig.add_subplot(gs[0, 1]),
        polar_201118,
        "A(ii)  Polar → Schira OM\n20/11/18  a=0.72, α=1.5, k=1",
    )
    show_om(
        fig.add_subplot(gs[0, 2]),
        polar_100718,
        "A(iii)  Polar → Schira OM\n10/07/18  a=0.74, α=1.0, k=1",
    )

    # Row B — cartesian
    show_vf(
        fig.add_subplot(gs[1, 0]),
        cart,
        "B(i)  Cartesian VF · fovea top-left",
    )
    show_om(
        fig.add_subplot(gs[1, 1]),
        cart_201118,
        "B(ii)  Cartesian → Schira OM\n20/11/18  a=0.72, α=1.5, k=1",
    )
    show_om(
        fig.add_subplot(gs[1, 2]),
        cart_100718,
        "B(iii)  Cartesian → Schira OM\n10/07/18  a=0.74, α=1.0, k=1",
    )

    if has_thesis:
        ax = fig.add_subplot(gs[2, :])
        ax.imshow(np.asarray(Image.open(THESIS_FIG12).convert("RGB")), origin="upper")
        ax.set_title(
            "Thesis Fig. 12 reference (grids registered onto chamber; "
            "ii=20/11/18, iii=10/07/18)",
            fontsize=11,
        )
        ax.axis("off")

    fig.suptitle(
        "Thesis Fig. 12 · LEGACY ``trans_cortex`` only (no chamber registration)\n"
        "Fovea always top-left in VF / LEFT in OM · columns differ by (a, α, k) only · "
        f"T=θ/α · ppd={ppd:g} · {extent:g}°×{extent:g}°",
        fontsize=13,
    )
    out = args.out or (out_dir / "legacy_fig12_cart_polar_grids.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
