#!/usr/bin/env python3
"""Letter D retinotopic transform (thesis Fig. 9 / 13c style).

Full letter ink (no piece removal). Default = Joanna 20/11/18 winner
a=0.72, α=1.5, k=1, Session A (1.0, −0.9), Joanna 2-pt registration.

Also writes a parameter comparison overlaid on the thesis Fig. 13c D map
(``legacy_letter_D_param_sweep_on_thesis.png``).
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
from scipy.io import loadmat
from scipy.ndimage import binary_erosion

from shira_revised.src.legacy_registration import (
    blend_on_chamber,
    prepare_chamber_display,
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
ARTIFACTS = PATHS["artifacts"]

CENTER_A = (1.0, -0.9)


def m1_m2_top_left(n_rows: int, n_cols: int, ppd: float):
    return np.arange(n_cols, dtype=np.float64) / ppd, np.arange(
        n_rows, dtype=np.float64
    ) / ppd


def load_vf2brain() -> dict[tuple[int, int], tuple[float, float]]:
    op = loadmat(
        LEGACY_ROOT / "VSDI points" / "201118" / "opt_points_8_5_22.mat",
        squeeze_me=True,
    )["opt_points"]
    return {
        (int(r[0]), int(r[1])): (float(r[4]), float(r[5]))
        for r in np.atleast_2d(op)
    }


def transform_letter(
    vf_ink: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    *,
    a: float,
    alpha: float,
    k: float,
    vf_pix: list,
    brain: list,
    alpha_mode: str = "divide",
):
    params = LegacySchiraParams(
        a=a, alpha=alpha, k=k, alpha_mode=alpha_mode, vf_abs=True
    )
    n = vf_ink.shape[0]
    tc = trans_cortex(vf_ink, m1, m2, params, bg=0)
    reg = register_om_to_chamber(
        tc.om,
        tc.xcoord,
        tc.ycoord,
        vf_pix_anchors=[tuple(p) for p in vf_pix],
        brain_anchors=[tuple(p) for p in brain],
        ppd=float(PPD),
        n_stim_rows=n,
        n_stim_cols=n,
    )
    ink = reg.out_image[:, :, 0] >= 180
    thin = ink & ~binary_erosion(ink, iterations=1)
    if not np.any(thin):
        thin = ink
    ys, xs = np.where(ink)
    if ys.size == 0:
        h2 = w2 = 0.0
    else:
        h2 = float(np.percentile(ys, 95) - np.percentile(ys, 5))
        w2 = float(np.percentile(xs, 95) - np.percentile(xs, 5))
    return {
        "params": params,
        "reg": reg,
        "tc": tc,
        "ink": ink,
        "thin": thin,
        "H": h2,
        "W": w2,
        "aspect": h2 / max(w2, 1e-6),
    }


def overlay_thin(base: np.ndarray, thin: np.ndarray, rgb=(0, 255, 255)) -> np.ndarray:
    out = np.asarray(base, dtype=np.float64).copy()
    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)
    color = np.array(rgb, dtype=np.float64)
    out[thin] = 0.15 * out[thin] + 0.85 * color
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extent-deg", type=float, default=6.0)
    ap.add_argument("--a", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument(
        "--anchors",
        choices=["joanna", "0805"],
        default="joanna",
        help="joanna=25-07 VF(49,18)/(15,18); 0805=VF(8,66)/(42,66)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = REPO / "shira_revised" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    jo = CFG["joanna_201118"]
    vf2brain = load_vf2brain()
    if args.anchors == "joanna":
        vf_pix = [list(p) for p in jo["vf_pix_anchors"]]
        brain = [list(p) for p in jo["brain_anchors"]]
    else:
        vf_pix = [[8, 66], [42, 66]]
        brain = [list(vf2brain[(8, 66)]), list(vf2brain[(42, 66)])]

    a = float(args.a if args.a is not None else jo["a"])
    alpha = float(args.alpha if args.alpha is not None else jo["alpha"])
    k = float(args.k if args.k is not None else jo["k"])

    n = int(round(args.extent_deg * PPD))
    rgb = create_letter_stim("D", CENTER_A)
    vf = rgb[:n, :n].copy()
    ink_vf = vf[:, :, 0] >= 200
    vf_ink = np.zeros_like(vf)
    vf_ink[ink_vf] = 255
    vf_ann = vf_ink.copy()
    vf_ann[0:3, 0:3] = (255, 0, 0)
    m1, m2 = m1_m2_top_left(n, n, float(PPD))

    primary = transform_letter(
        vf_ink, m1, m2, a=a, alpha=alpha, k=k, vf_pix=vf_pix, brain=brain
    )
    print(
        f"primary anchors={args.anchors} a={a} α={alpha} k={k} "
        f"H={primary['H']:.1f} W={primary['W']:.1f} asp={primary['aspect']:.3f}"
    )

    bv = np.asarray(Image.open(LEGACY_ROOT / ARTIFACTS["chamber_201118"]))
    overlay = blend_on_chamber(
        bv,
        primary["reg"].out_image,
        thin=True,
        alpha=0.95,
        dark_chamber=True,
        ink_threshold=180,
    )

    emp_path = LEGACY_ROOT / "images" / "201118" / "Session a" / "D_new_resized.png"
    emp = np.asarray(Image.open(emp_path).convert("RGB"))
    emp_d = overlay_thin(np.clip(emp.astype(np.float64) * 0.65, 0, 255), primary["thin"], (255, 255, 255))

    thesis_map = np.asarray(
        Image.open(out_dir / "thesis_pages" / "fig13c_map.png").convert("RGB").resize((100, 100))
    )
    thesis_ov = overlay_thin(thesis_map, primary["thin"], (0, 255, 255))

    user_d = Path(
        "/Users/ossnat/.cursor/projects/Users-ossnat-GondaResearch-VSD-FM-VSD-Encoder-01"
        "/assets/image-af9b8d1b-15d0-4067-ba16-39d31e32806c.png"
    )

    # α sweep on thesis (fixed Joanna a=0.72 k=1, 2-pt anchors)
    jo_vf = [list(p) for p in jo["vf_pix_anchors"]]
    jo_br = [list(p) for p in jo["brain_anchors"]]
    alphas = [1.5, 2.0, 2.5, 3.0, 3.5]
    fig_s, axes_s = plt.subplots(2, 3, figsize=(12, 8), facecolor="white")
    axes_s = axes_s.ravel()
    axes_s[0].imshow(thesis_map, origin="upper")
    axes_s[0].set_title("Thesis Fig. 13c D")
    axes_s[0].axis("off")
    for i, al in enumerate(alphas, start=1):
        r = transform_letter(
            vf_ink, m1, m2, a=0.72, alpha=al, k=1.0, vf_pix=jo_vf, brain=jo_br
        )
        axes_s[i].imshow(
            overlay_thin(thesis_map, r["thin"], (0, 255, 255)), origin="upper"
        )
        axes_s[i].set_title(
            f"a=0.72  α={al:g}  k=1\nH={r['H']:.0f}  W={r['W']:.0f}  asp={r['aspect']:.2f}",
            fontsize=9,
        )
        axes_s[i].axis("off")
        print(f"  α={al:g}: H={r['H']:.1f} W={r['W']:.1f} asp={r['aspect']:.3f}")
    fig_s.suptitle(
        "Full letter D on thesis (cyan=model) · α sweep · Joanna 2-pt a=0.72 k=1",
        fontsize=12,
    )
    sweep_path = out_dir / "legacy_letter_D_param_sweep_on_thesis.png"
    fig_s.savefig(sweep_path, dpi=160, bbox_inches="tight")
    plt.close(fig_s)

    ys, xs = np.where(primary["tc"].om[:, :, 0] > 0)
    pad = 25
    om_view = primary["tc"].om[
        max(0, int(ys.min()) - pad) : int(ys.max()) + pad + 1,
        max(0, int(xs.min()) - pad) : int(xs.max()) + pad + 1,
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white")
    axes[0, 0].imshow(vf_ann, origin="upper")
    axes[0, 0].set_title("VF letter D (fovea TL)\nSession A (1.0, −0.9)")
    axes[0, 1].imshow(np.asarray(Image.open(user_d)), origin="upper")
    axes[0, 1].set_title("Thesis Fig. 9A reference")
    axes[0, 2].imshow(om_view, origin="upper")
    axes[0, 2].set_title(f"Schira OM (full ink)\na={a} α={alpha} k={k}")
    axes[1, 0].imshow(overlay, origin="upper")
    axes[1, 0].set_title(f"Chamber · full letter\n{args.anchors} 2-pt")
    axes[1, 1].imshow(emp_d, origin="upper")
    axes[1, 1].set_title("Full letter on empirical")
    axes[1, 2].imshow(thesis_ov, origin="upper")
    axes[1, 2].set_title("Cyan=model on thesis D")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Letter D · full ink · a={a} α={alpha} k={k} · {args.anchors} · asp={primary['aspect']:.2f}",
        fontsize=12,
    )
    out = args.out or (out_dir / "legacy_letter_D_201118.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    Image.fromarray(overlay).save(out_dir / "legacy_letter_D_chamber_slim.png")
    Image.fromarray(emp_d).save(out_dir / "_D_outline_on_empirical.png")
    Image.fromarray(thesis_ov).save(out_dir / "legacy_letter_D_on_thesis.png")
    print("wrote", out)
    print("wrote", sweep_path)


if __name__ == "__main__":
    main()
