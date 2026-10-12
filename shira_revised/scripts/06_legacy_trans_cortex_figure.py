#!/usr/bin/env python3
"""One figure: stimulus + legacy ``trans_cortex`` OM (MATLAB output before registration)."""

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
from shira_revised.src.legacy_trans_cortex import (
    m1_m2_top_right_origin,
    trans_cortex,
)
from shira_revised.src.legacy_trans_cortex import m1_m2_from_retino_mat
from shira_revised.src.legacy_warp import load_retino_mat_crop
from shira_revised.src.vf_top_right import detect_ppd_from_grid


def main() -> None:
    ap = argparse.ArgumentParser(description="Legacy trans_cortex only (Schira OM).")
    ap.add_argument("image", type=Path)
    ap.add_argument("--ppd", type=float, default=None)
    ap.add_argument(
        "--vf-origin",
        choices=("top-right", "retino-mat"),
        default="top-right",
        help="How to assign VF deg to pixels (default: user top-right 0,0)",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--a", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--k", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = REPO / "shira_revised" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(out_dir / ".mplconfig")

    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    n_rows, n_cols = rgb.shape[:2]
    gray = rgb.mean(axis=2)

    ppd = args.ppd or detect_ppd_from_grid(gray) or 60.0
    print(f"ppd={ppd:.2f} px/deg")

    if args.vf_origin == "retino-mat":
        retino = load_retino_mat_crop()
        m1, m2 = m1_m2_from_retino_mat(retino, n_rows, n_cols)
    else:
        m1, m2 = m1_m2_top_right_origin(n_rows, n_cols, ppd)

    base = joanna_params()
    params = base
    if args.a is not None or args.alpha is not None:
        from shira_revised.src.legacy_schira import LegacySchiraParams

        params = LegacySchiraParams(
            a=float(args.a if args.a is not None else base.a),
            alpha=float(args.alpha if args.alpha is not None else base.alpha),
            k=float(args.k),
            alpha_mode="divide",
            vf_abs=True,
        )

    print("running trans_cortex polygon fill (legacy MATLAB path)...")
    result = trans_cortex(rgb, m1, m2, params, quick_mode=False, bg=0)
    om = result.om
    print(f"OM shape {om.shape[1]} x {om.shape[0]} (W x H)")

    # One wide figure: input | OM (legacy imshow, no VF axis relabeling).
    om_aspect = om.shape[1] / max(om.shape[0], 1)
    in_aspect = n_cols / n_rows
    fig_h = 10.0
    fig_w = fig_h * (in_aspect + om_aspect) * 0.55 + 2
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    axes[0].imshow(rgb, origin="upper", aspect="equal", interpolation="nearest")
    axes[0].set_title("Stimulus (input to trans_cortex)", fontsize=14)
    axes[0].set_xlabel("column (px)")
    axes[0].set_ylabel("row (px)")

    axes[1].imshow(om, origin="upper", aspect="equal", interpolation="nearest")
    axes[1].set_title(
        "Legacy output: OM from trans_cortex.m\n"
        "(Schira only — no fitgeotrans / 100×100 registration)",
        fontsize=14,
    )
    axes[1].set_xlabel("OM column (1-based index after round(w×1000))")
    axes[1].set_ylabel("OM row")

    fig.suptitle(
        f"Legacy Gandalf Schira  model=2 monopole (b≥200)  |  "
        f"a={params.a}, s(α)={params.alpha}, K={params.k}  |  "
        f"T=θ/α, abs VF  |  vf_origin={args.vf_origin}, ppd={ppd:.1f}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out = args.out or (out_dir / f"legacy_trans_cortex_{args.image.stem}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
