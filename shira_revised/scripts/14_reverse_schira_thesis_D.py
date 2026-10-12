#!/usr/bin/env python3
"""Reverse-map thesis Fig. 13c D → VF under legacy Joanna Schira+reg.

Top: thesis crop | thin cyan outline | re-forward of *reconstructed* VF D
through Schira → Joanna 2‑pt registration onto chamber.
Bottom: reverse pullback VF vs actual letter D.
"""

from __future__ import annotations

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
from scipy.ndimage import binary_dilation, binary_erosion

from shira_revised.src.legacy_registration import (
    blend_on_chamber,
    om_points_to_chamber,
    register_om_to_chamber,
)
from shira_revised.src.legacy_schira import LegacySchiraParams
from shira_revised.src.legacy_stimuli import PPD, create_letter_stim
from shira_revised.src.legacy_trans_cortex import trans_cortex

CFG = yaml.safe_load(
    (REPO / "shira_revised" / "configs" / "gandalf_legacy.yaml").read_text()
)
PATHS = yaml.safe_load((REPO / "shira_revised" / "configs" / "paths.yaml").read_text())
LEGACY_ROOT = Path(os.environ.get("SHIRA_REVISED_LEGACY_ROOT", PATHS["legacy_root"]))
OUT_DIR = REPO / "shira_revised" / "artifacts"
THESIS_MAP = OUT_DIR / "thesis_pages" / "fig13c_map.png"
THESIS_EDGE = OUT_DIR / "_thesis_D_target_edge100.png"


def thin_thesis_d_line(size: int = 100) -> np.ndarray:
    """1‑px thesis D outline (edge mask only)."""
    if THESIS_EDGE.is_file():
        edge = np.asarray(
            Image.open(THESIS_EDGE).convert("L").resize(
                (size, size), Image.Resampling.NEAREST
            )
        )
        mask = edge > 128
    else:
        thesis = np.asarray(
            Image.open(THESIS_MAP).convert("RGB").resize(
                (size, size), Image.Resampling.BILINEAR
            )
        )
        g = thesis.astype(np.float64).mean(axis=2)
        mask = (g < 25) & (np.arange(size)[:, None] < int(0.5 * size))
    if mask.any() and mask.sum() > 80:
        core = binary_erosion(mask, iterations=1)
        rim = mask & ~core
        if rim.any():
            mask = rim
    return mask


def paint_thin(
    base: np.ndarray, mask: np.ndarray, rgb: tuple[int, int, int]
) -> np.ndarray:
    out = np.asarray(base, dtype=np.uint8).copy()
    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)
    out[mask] = np.array(rgb, dtype=np.uint8)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jo = CFG["joanna_201118"]
    a, alpha, k = float(jo["a"]), float(jo["alpha"]), float(jo["k"])
    params = LegacySchiraParams(
        a=a, alpha=alpha, k=k, alpha_mode="divide", vf_abs=True
    )
    vf_pix = [tuple(p) for p in jo["vf_pix_anchors"]]
    brain = [tuple(p) for p in jo["brain_anchors"]]

    n = int(round(6.0 * PPD))
    m1 = np.arange(n, dtype=np.float64) / float(PPD)
    m2 = np.arange(n, dtype=np.float64) / float(PPD)
    dummy = np.full((n, n, 3), 128, dtype=np.uint8)

    print(f"forward coords n={n} a={a} α={alpha} k={k} …")
    tc = trans_cortex(dummy, m1, m2, params, quick_mode=True, bg=0)
    om_h = int(np.nanmax(tc.ycoord))
    om_w = int(np.nanmax(tc.xcoord))
    dummy_om = np.zeros((om_h, om_w, 3), dtype=np.uint8)
    reg = register_om_to_chamber(
        dummy_om,
        tc.xcoord,
        tc.ycoord,
        vf_pix_anchors=vf_pix,
        brain_anchors=brain,
        ppd=float(PPD),
        n_stim_rows=n,
        n_stim_cols=n,
    )

    ok = np.isfinite(tc.xcoord) & np.isfinite(tc.ycoord)
    om_xy = np.column_stack([tc.xcoord[ok], tc.ycoord[ok]])
    ch_rc = om_points_to_chamber(om_xy, reg.tform)
    rows_i = np.clip(np.rint(ch_rc[:, 0]).astype(int), 0, 99)
    cols_i = np.clip(np.rint(ch_rc[:, 1]).astype(int), 0, 99)

    thesis_rgb = np.asarray(
        Image.open(THESIS_MAP).convert("RGB").resize((100, 100), Image.Resampling.BILINEAR)
    )
    thin_line = thin_thesis_d_line(100)
    hit_mask = binary_dilation(thin_line, iterations=1) if thin_line.any() else thin_line
    hit = hit_mask[rows_i, cols_i]

    vf_mask = np.zeros((n, n), dtype=bool)
    flat_idx = np.flatnonzero(ok.ravel())
    vf_mask.ravel()[flat_idx[hit]] = True
    print(f"thin thesis px={int(thin_line.sum())}  recovered VF px={int(vf_mask.sum())}")

    vf_rec = np.full((n, n, 3), 128, dtype=np.uint8)
    vf_rec[vf_mask] = (255, 220, 40)
    vf_rec[0:3, 0:3] = (255, 0, 0)

    # Actual letter D @ Session a — same path as prior D figures (Schira → reg)
    leg = create_letter_stim("D", (1.0, -0.9))[:n, :n].copy()
    leg_ink = leg[:, :, 0] >= 200
    leg_show = np.full((n, n, 3), 128, dtype=np.uint8)
    leg_show[leg_ink] = (255, 220, 40)
    leg_show[0:3, 0:3] = (255, 0, 0)

    # Last panel: take reverse-reconstructed VF D → fw Schira → Joanna reg → brain
    vf_recon = np.zeros((n, n, 3), dtype=np.uint8)
    vf_recon[vf_mask] = 255
    tc_r = trans_cortex(vf_recon, m1, m2, params, quick_mode=False, bg=0)
    reg_r = register_om_to_chamber(
        tc_r.om,
        tc_r.xcoord,
        tc_r.ycoord,
        vf_pix_anchors=vf_pix,
        brain_anchors=brain,
        ppd=float(PPD),
        n_stim_rows=n,
        n_stim_cols=n,
    )
    bv = np.asarray(Image.open(LEGACY_ROOT / PATHS["artifacts"]["chamber_201118"]))
    recon_on_brain = blend_on_chamber(
        bv,
        reg_r.out_image,
        thin=True,
        alpha=0.95,
        dark_chamber=True,
        ink_threshold=180,
    )
    thesis_thin_cyan = paint_thin(thesis_rgb, thin_line, (0, 255, 255))

    if vf_mask.any() or leg_ink.any():
        ys, xs = np.where(vf_mask | leg_ink)
        pad = 28
        r0, r1 = max(0, int(ys.min()) - pad), min(n, int(ys.max()) + pad + 1)
        c0, c1 = max(0, int(xs.min()) - pad), min(n, int(xs.max()) + pad + 1)
    else:
        r0, r1, c0, c1 = 0, n, 0, n

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white")
    axes[0, 0].imshow(thesis_rgb, origin="upper")
    axes[0, 0].set_title("Thesis Fig. 13c D (chamber crop)")
    axes[0, 1].imshow(thesis_thin_cyan, origin="upper")
    axes[0, 1].set_title("Thesis D outline (cyan=thin line)")
    axes[0, 2].imshow(recon_on_brain, origin="upper")
    axes[0, 2].set_title(
        f"Reconstructed VF D → Schira→reg on brain\n"
        f"a={a} α={alpha} k={k}"
    )

    axes[1, 0].imshow(vf_rec, origin="upper")
    axes[1, 0].set_title("Reverse → VF (full 6°)\nyellow = required ink")
    axes[1, 1].imshow(vf_rec[r0:r1, c0:c1], origin="upper")
    axes[1, 1].set_title("Reverse → VF zoom")
    axes[1, 2].imshow(leg_show[r0:r1, c0:c1], origin="upper")
    axes[1, 2].set_title("Actual letter D @ (1.0, −0.9)\n(same zoom)")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Inverse pullback of thesis thin D · legacy Joanna a={a} α={alpha} k={k}",
        fontsize=11,
    )
    out = OUT_DIR / "legacy_reverse_schira_thesis_D.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    Image.fromarray(vf_rec).save(OUT_DIR / "legacy_reverse_schira_thesis_D_vf.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
