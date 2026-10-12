#!/usr/bin/env python3
"""Thesis Fig. 12 style: legacy cart/polar grids registered onto BV chamber.

Uses Joanna ``trans_and_registration`` path (2-point nonreflective similarity)
with session winners:

  20/11/18: a=0.72, α=1.5, k=1 · VF (49,18)/(15,18) → brain (31,78)/(36,25)
  10/07/18: a=0.74, α=1.0, k=1 · VF (9,21)/(37,21) → brain (46,24)/(36,58)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_mpl = REPO / "shira_revised" / "artifacts" / ".mplconfig"
_mpl.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from shira_revised.src.legacy_registration import (
    blend_on_chamber,
    register_om_align_vm_to_lvm,
    register_om_to_chamber,
)
from shira_revised.src.legacy_schira import LegacySchiraParams
from shira_revised.src.legacy_stimuli import (
    BG,
    LEGACY_COLS,
    LEGACY_ROWS,
    PPD,
    _gray_rgb,
    create_cart_stim,
    create_polar_stim,
)
from shira_revised.src.legacy_trans_cortex import trans_cortex

CFG = yaml.safe_load(
    (REPO / "shira_revised" / "configs" / "gandalf_legacy.yaml").read_text()
)
PATHS = yaml.safe_load((REPO / "shira_revised" / "configs" / "paths.yaml").read_text())
LEGACY_ROOT = Path(
    os.environ.get("SHIRA_REVISED_LEGACY_ROOT", PATHS["legacy_root"])
)
ARTIFACTS = PATHS["artifacts"]

THESIS_FIG12 = (
    REPO / "shira_revised" / "artifacts" / "thesis_fig12" / "thesis_page_030.png"
)
FIG2_PAGE = (
    REPO
    / "shira_revised"
    / "artifacts"
    / "ayzenshtat_fig2_extract"
    / "page6_img0_2065x1371.jpeg"
)


def m1_m2_top_left(n_rows: int, n_cols: int, ppd: float) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(n_cols, dtype=np.float64) / ppd, np.arange(n_rows, dtype=np.float64) / ppd


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


def load_chamber(session: str) -> np.ndarray:
    key = "chamber_201118" if session == "201118" else "chamber_100718"
    path = LEGACY_ROOT / ARTIFACTS[key]
    return np.asarray(Image.open(path))


def session_spec(name: str) -> dict:
    block = CFG["joanna_201118"] if name == "201118" else CFG["joanna_100718"]
    params = LegacySchiraParams(
        a=float(block["a"]),
        alpha=float(block["alpha"]),
        k=float(block["k"]),
        alpha_mode="divide",
        vf_abs=True,
    )
    vf = [tuple(p) for p in block["vf_pix_anchors"]]
    brain = [tuple(p) for p in block["brain_anchors"]]
    return {"params": params, "vf": vf, "brain": brain, "label": name}


def register_grid(
    rgb: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    spec: dict,
    ppd: float,
    *,
    align_vm_lvm: bool,
    lvm_row: float,
) -> np.ndarray:
    tc = trans_cortex(rgb, m1, m2, spec["params"], quick_mode=False, bg=0)
    if align_vm_lvm:
        reg = register_om_align_vm_to_lvm(
            tc.om,
            tc.xcoord,
            tc.ycoord,
            lvm_row=lvm_row,
            ppd=ppd,
        )
    else:
        reg = register_om_to_chamber(
            tc.om,
            tc.xcoord,
            tc.ycoord,
            vf_pix_anchors=spec["vf"],
            brain_anchors=spec["brain"],
            ppd=ppd,
            n_stim_rows=rgb.shape[0],
            n_stim_cols=rgb.shape[1],
        )
    return reg.out_image, tc


def crop_fig2e_lvm_only(src: Path, out: Path) -> None:
    """Crop panel E; keep dashed LVM, remove solid red grid + LUS."""
    im = np.asarray(Image.open(src).convert("RGB"))
    h, w = im.shape[:2]
    y0, y1 = int(0.50 * h), int(0.98 * h)
    x0, x1 = int(0.34 * w), int(0.66 * w)
    panel = im[y0:y1, x0:x1].copy()

    r = panel[:, :, 0].astype(np.int16)
    g = panel[:, :, 1].astype(np.int16)
    b = panel[:, :, 2].astype(np.int16)
    redish = (r - g > 45) & (r - b > 45) & (r > 90)

    # Build a grayscale underlay (desaturate red overlays).
    gray = (0.3 * r + 0.59 * g + 0.11 * b).astype(np.uint8)
    base = np.stack([gray, gray, gray], axis=-1)

    # LVM ≈ middle dashed border: among red pixels, take the row closest to
    # the 35th percentile of red rows per column (below LUS, above V1 grid).
    keep = np.zeros(panel.shape[:2], dtype=bool)
    for c in range(panel.shape[1]):
        rows = np.where(redish[:, c])[0]
        if rows.size < 3:
            continue
        target = np.percentile(rows, 35)
        # keep reds within a thin band around that envelope (dashed LVM)
        band = rows[np.abs(rows.astype(np.float64) - target) <= 2.5]
        if band.size == 0:
            continue
        keep[band, c] = True

    # Dilate keep slightly for visibility
    from scipy.ndimage import binary_dilation

    keep = binary_dilation(keep, iterations=1)
    out_im = base.copy()
    out_im[keep] = (230, 35, 35)
    Image.fromarray(out_im).save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppd", type=float, default=float(PPD))
    # Thesis Fig. 12 A/B(i): MATLAB polar rings only go to 3°; cart is a 6×6
    # of 0.5° cells → 3° square (not the previous 6° crop).
    ap.add_argument("--extent-deg", type=float, default=3.0)
    ap.add_argument(
        "--align-vm-lvm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pin VF y-axis to LVM (default: Joanna 2-pt letter anchors, thesis Fig.12)",
    )
    ap.add_argument("--lvm-row", type=float, default=36.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = REPO / "shira_revised" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(out_dir / ".mplconfig")

    ppd = float(args.ppd)
    extent = float(args.extent_deg)

    # Thesis-matched VF squares (3° @ 35 ppd unless overridden).
    polar = create_polar_stim(ppd=int(ppd), is_left_hem=True, extent_deg=extent)
    if abs(ppd - PPD) < 1e-9:
        cart = create_cart_stim(extent_deg=extent, ppd=int(ppd))
    else:
        cart = crop_vf_square(make_cart(ppd), ppd, extent)
        cart[-1, :] = 255
        cart[:, -1] = 255
    n = polar.shape[0]
    m1, m2 = m1_m2_top_left(n, n, ppd)

    s201118 = session_spec("201118")
    s100718 = session_spec("100718")
    bv201118 = load_chamber("201118")
    bv100718 = load_chamber("100718")

    mode = f"VM→LVM row={args.lvm_row:.0f}" if args.align_vm_lvm else ""
    print(f"{n}×{n} @ {ppd} ppd · {mode or '2-pt reg'} · registering polar/cart → chamber…")
    polar_reg = {
        "201118": register_grid(
            polar, m1, m2, s201118, ppd,
            align_vm_lvm=args.align_vm_lvm, lvm_row=args.lvm_row,
        )[0],
        "100718": register_grid(
            polar, m1, m2, s100718, ppd,
            align_vm_lvm=args.align_vm_lvm, lvm_row=args.lvm_row,
        )[0],
    }
    cart_reg = {
        "201118": register_grid(
            cart, m1, m2, s201118, ppd,
            align_vm_lvm=args.align_vm_lvm, lvm_row=args.lvm_row,
        )[0],
        "100718": register_grid(
            cart, m1, m2, s100718, ppd,
            align_vm_lvm=args.align_vm_lvm, lvm_row=args.lvm_row,
        )[0],
    }

    def decorate(bv, reg_img, params):
        # Dark circular plate + fine white dashed grid (thesis Fig. 12 look).
        return blend_on_chamber(
            bv, reg_img, thin=True, alpha=0.85, dark_chamber=True, ink_threshold=180
        )

    overlays = {
        ("polar", "201118"): decorate(bv201118, polar_reg["201118"], s201118["params"]),
        ("polar", "100718"): decorate(bv100718, polar_reg["100718"], s100718["params"]),
        ("cart", "201118"): decorate(bv201118, cart_reg["201118"], s201118["params"]),
        ("cart", "100718"): decorate(bv100718, cart_reg["100718"], s100718["params"]),
    }

    fig2e_lvm = out_dir / "ayzenshtat_2012_fig2e_lvm_only.png"
    if FIG2_PAGE.is_file():
        crop_fig2e_lvm_only(FIG2_PAGE, fig2e_lvm)
        print("Fig 2E LVM-only →", fig2e_lvm)

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.15, 1.15, 1.0], wspace=0.18, hspace=0.25)

    def show(ax, img, title: str, *, cmap=None):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or "gray", origin="upper")
        else:
            ax.imshow(img, origin="upper")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    show(fig.add_subplot(gs[0, 0]), polar, "A(i)  Polar VF\n(fovea top-left)")
    show(
        fig.add_subplot(gs[0, 1]),
        overlays[("polar", "201118")],
        f"A(ii)  Polar on chamber\n20/11/18{('  ' + mode) if mode else ''}",
    )
    show(
        fig.add_subplot(gs[0, 2]),
        overlays[("polar", "100718")],
        f"A(iii)  Polar on chamber\n10/07/18{('  ' + mode) if mode else ''}",
    )
    if fig2e_lvm.is_file():
        show(
            fig.add_subplot(gs[0, 3]),
            np.asarray(Image.open(fig2e_lvm)),
            "Ref: Ayzenshtat 2012 Fig. 2E\nLVM dashed only",
        )
    elif THESIS_FIG12.is_file():
        show(fig.add_subplot(gs[0, 3]), np.asarray(Image.open(THESIS_FIG12)), "Thesis Fig. 12")

    show(fig.add_subplot(gs[1, 0]), cart, "B(i)  Cartesian VF\n(fovea top-left)")
    show(
        fig.add_subplot(gs[1, 1]),
        overlays[("cart", "201118")],
        f"B(ii)  Cart on chamber\n20/11/18{('  ' + mode) if mode else ''}",
    )
    show(
        fig.add_subplot(gs[1, 2]),
        overlays[("cart", "100718")],
        f"B(iii)  Cart on chamber\n10/07/18{('  ' + mode) if mode else ''}",
    )
    if THESIS_FIG12.is_file():
        show(
            fig.add_subplot(gs[1, 3]),
            np.asarray(Image.open(THESIS_FIG12)),
            "Thesis Fig. 12 (reference)",
        )

    fig.suptitle(
        "Legacy Schira → chamber (dark Fig. 12 plate)",
        fontsize=12,
        y=0.98,
    )
    out = args.out or (out_dir / "legacy_fig12_chamber_overlays.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    for (kind, sess), img in overlays.items():
        p = out_dir / f"legacy_fig12_{kind}_chamber_{sess}.png"
        Image.fromarray(img).save(p)
        print("wrote", p)


if __name__ == "__main__":
    main()
