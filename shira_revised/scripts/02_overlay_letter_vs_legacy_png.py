#!/usr/bin/env python3
"""Overlay Schira ink (legacy vs current conventions) on legacy Session PNGs.

Example::

    scripts/py shira_revised/scripts/02_overlay_letter_vs_legacy_png.py --letter A --session a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shira_revised.src.legacy_schira import (
    LegacySchiraParams,
    cortical_pixel_indices_from_w,
    forward_legacy_schira,
    thesis_like_params,
)
from shira_revised.src.legacy_vf import load_gandalf_legacy_yaml, load_paths, make_abs_vf_grid


def _stamp_w_to_chamber(
    x_deg: np.ndarray,
    y_deg: np.ndarray,
    ink: np.ndarray,
    params: LegacySchiraParams,
    brain_anchors: np.ndarray,
    vf_anchor_deg: np.ndarray,
    chamber: int = 100,
) -> np.ndarray:
    """Rough 2-pt similarity: map Schira ``w`` of ink → chamber via anchors.

    Uses continuous ``w`` (not integer xcoord) for a first visual check.
    Full ``trans_cortex`` polygon fill + ``imwarp`` is a later port.
    """
    w_all = forward_legacy_schira(x_deg, y_deg, params)
    w_anch = forward_legacy_schira(vf_anchor_deg[:, 0], vf_anchor_deg[:, 1], params)
    if not (np.isfinite(w_anch).all()):
        raise RuntimeError("anchor w non-finite")

    # Similarity: w → chamber pixel (col, row) from 2 points.
    # Solve for scale, rot, translation in complex plane: z' = s * z + t
    # with s complex. Using first two anchors.
    z = w_anch
    p = brain_anchors[:, 1] + 1j * brain_anchors[:, 0]  # col + i*row
    # s = (p1-p0)/(z1-z0); t = p0 - s*z0
    s = (p[1] - p[0]) / (z[1] - z[0])
    t = p[0] - s * z[0]

    out = np.zeros((chamber, chamber), dtype=np.float64)
    mask = ink.astype(bool) & np.isfinite(w_all)
    mapped = s * w_all[mask] + t
    rr = np.rint(mapped.imag).astype(int)
    cc = np.rint(mapped.real).astype(int)
    ok = (rr >= 0) & (rr < chamber) & (cc >= 0) & (cc < chamber)
    out[rr[ok], cc[ok]] = 1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--letter", default="A")
    ap.add_argument("--session", default="a", choices=["a", "d"])
    ap.add_argument("--params", default="joanna", choices=["joanna", "amit"])
    args = ap.parse_args()

    paths = load_paths()
    cfg = load_gandalf_legacy_yaml()
    letter_dir = paths[f"letters_201118_{args.session}"]
    png = letter_dir / f"{args.letter.upper()}.png"
    if not png.is_file():
        # some files are like Fd_temp; try plain
        alts = list(letter_dir.glob(f"{args.letter.upper()}*.png"))
        if not alts:
            raise SystemExit(f"missing empirical PNG under {letter_dir}")
        png = alts[0]

    emp = np.asarray(Image.open(png).convert("L"), dtype=np.float64)
    if emp.shape[0] != 100:
        emp = np.asarray(Image.open(png).convert("L").resize((100, 100)), dtype=np.float64)
    emp = emp / (emp.max() + 1e-9)

    block = cfg["joanna_201118"] if args.params == "joanna" else cfg["amit_201118"]
    a, alpha, k = float(block["a"]), float(block["alpha"]), float(block["k"])

    if args.params == "joanna":
        # VF anchors: use deg from letter centers of opt path is messy;
        # use Amit-style deg near HM for a first look, plus Joanna brain anchors.
        # Better: convert VF pix anchors with ppd abs convention.
        vf_pix = np.asarray(block["vf_pix_anchors"], dtype=np.float64)
        # opt_points store (row, col); deg ≈ ± pix/ppd with abs top-left
        vf_deg = np.column_stack(
            [vf_pix[:, 1] / 35.0, -vf_pix[:, 0] / 35.0]
        )
        brain = np.asarray(block["brain_anchors"], dtype=np.float64)
    else:
        anch = block["anchors"]
        vf_deg = np.array([a_["vf_deg"] for a_ in anch], dtype=np.float64)
        brain = np.array([a_["brain"] for a_ in anch], dtype=np.float64)

    # Dense abs VF grid (legacy-like) and a synthetic letter blob at center.
    xx, yy = make_abs_vf_grid(120, 120, x_max_deg=6.0, y_max_deg=6.0)
    # signed for thesis path samples:
    xx_s = xx.copy()
    yy_s = -yy.copy()

    centers = cfg["letter_centers_deg"]
    cx, cy = centers[args.letter.upper()]
    # ink on abs grid: distance to (|cx|,|cy|)
    ink = (np.hypot(xx - abs(cx), yy - abs(cy)) < 0.35).astype(np.float64)

    legacy_p = LegacySchiraParams(a=a, alpha=alpha, k=k, alpha_mode="divide", vf_abs=True)
    thesis_p = thesis_like_params(a, alpha, k)

    # For legacy: feed abs coords (xx, yy); for thesis: signed (xx_s, yy_s)
    chamber_leg = _stamp_w_to_chamber(xx, yy, ink, legacy_p, brain, vf_deg)
    chamber_th = _stamp_w_to_chamber(xx_s, yy_s, ink, thesis_p, brain, vf_deg)

    out_dir = REPO / "shira_revised" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes[0].imshow(emp, cmap="gray")
    axes[0].set_title(f"legacy VSD {png.name}")
    axes[1].imshow(emp, cmap="gray")
    axes[1].imshow(chamber_leg, cmap="Reds", alpha=0.55)
    axes[1].set_title("ink: legacy divide+abs")
    axes[2].imshow(emp, cmap="gray")
    axes[2].imshow(chamber_th, cmap="Blues", alpha=0.55)
    axes[2].set_title("ink: thesis multiply+signed")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{args.params} params a={a} α={alpha} k={k} | letter {args.letter}")
    fig.tight_layout()
    out = out_dir / f"overlay_{args.params}_{args.letter}_s{args.session}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    print("note: 2-pt stamp is a first visual; full imwarp port is next.")


if __name__ == "__main__":
    main()
