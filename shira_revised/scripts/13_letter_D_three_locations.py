#!/usr/bin/env python3
"""Letter D at 201118a / c / d catalog positions, same Joanna Schira+reg.

Same params as Session a: a=0.72, α=1.5, k=1, Joanna 2-pt anchors.
Positions from encoding catalog (letter_D_white_1).
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
from scipy.ndimage import binary_erosion

from shira_revised.src.legacy_registration import (
    blend_on_chamber,
    register_om_to_chamber,
)
from shira_revised.src.legacy_schira import LegacySchiraParams
from shira_revised.src.legacy_stimuli import PPD
from shira_revised.src.legacy_trans_cortex import trans_cortex
from src.paths import project_root, resolve_data_path
from src.stimuli.catalog import load_full_encoder_catalog, stimulus_spec_from_mapping
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.render import RenderConfig, render_stimulus

CFG = yaml.safe_load(
    (REPO / "shira_revised" / "configs" / "gandalf_legacy.yaml").read_text()
)
PATHS = yaml.safe_load((REPO / "shira_revised" / "configs" / "paths.yaml").read_text())
LEGACY_ROOT = Path(os.environ.get("SHIRA_REVISED_LEGACY_ROOT", PATHS["legacy_root"]))
OUT_DIR = REPO / "shira_revised" / "artifacts"

SESSIONS = ("201118a", "201118c", "201118d")


def ink_from_rgb(rgb: np.ndarray, bg: int = 128, tol: int = 20) -> np.ndarray:
    g = rgb.astype(np.float64).mean(axis=2)
    return np.abs(g - float(bg)) >= float(tol)


def metrics(ink: np.ndarray) -> dict:
    ys, xs = np.where(ink)
    if ys.size == 0:
        return {"H": 0.0, "W": 0.0, "aspect": 0.0, "n": 0}
    h = float(np.percentile(ys, 95) - np.percentile(ys, 5))
    w = float(np.percentile(xs, 95) - np.percentile(xs, 5))
    return {"H": h, "W": w, "aspect": h / max(w, 1e-6), "n": int(ys.size)}


def om_crop(om: np.ndarray, pad: int = 20) -> np.ndarray:
    ys, xs = np.where(om[:, :, 0] > 0)
    if ys.size == 0:
        return om[:80, :80]
    return om[
        max(0, int(ys.min()) - pad) : int(ys.max()) + pad + 1,
        max(0, int(xs.min()) - pad) : int(xs.max()) + pad + 1,
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stim_cfg = yaml.safe_load(
        (project_root() / "configs" / "stimuli" / "default.yaml").read_text()
    )
    render_cfg = RenderConfig(
        canvas_size=int(stim_cfg["canvas_size"]),
        pixels_per_deg=float(stim_cfg["pixels_per_deg"]),
        quadrant_extent_deg=float(stim_cfg["quadrant_extent_deg"]),
        background_gray=int(stim_cfg["background_gray"]),
        bar_length_deg=float(stim_cfg["bar_length_deg"]),
        bar_width_px=int(stim_cfg["bar_width_px"]),
        contour_width_px=int(stim_cfg["contour_width_px"]),
        assume_size_is_diameter=bool(stim_cfg["assume_size_is_diameter"]),
        draw_fixation=False,
        letter_box_deg=float(stim_cfg.get("letter_box_deg", 1.0)),
    )

    cfg = yaml.safe_load((project_root() / "configs" / "default.yaml").read_text())
    root = resolve_data_path(cfg["paths"]["encoder_data_root"])
    cat = attach_stimulus_ids(load_full_encoder_catalog(root, monkey="Gandalf"))
    sub = cat[cat["stimulus_id"].astype(str) == "letter_D_white_1"]

    jo = CFG["joanna_201118"]
    a, alpha, k = float(jo["a"]), float(jo["alpha"]), float(jo["k"])
    vf_pix = [list(p) for p in jo["vf_pix_anchors"]]
    brain = [list(p) for p in jo["brain_anchors"]]
    params = LegacySchiraParams(
        a=a, alpha=alpha, k=k, alpha_mode="divide", vf_abs=True
    )
    bv = np.asarray(Image.open(LEGACY_ROOT / PATHS["artifacts"]["chamber_201118"]))

    rows = []
    for sess in SESSIONS:
        on = sub[sub["h5_session"].astype(str).str.startswith(sess)]
        if on.empty:
            raise RuntimeError(f"no letter_D_white_1 for {sess}")
        row = on.iloc[0]
        spec = stimulus_spec_from_mapping(row.to_dict())
        rgb = render_stimulus(spec, render_cfg)
        ink = ink_from_rgb(rgb, bg=render_cfg.background_gray)
        vf = np.zeros_like(rgb)
        vf[ink] = 255
        n = vf.shape[0]
        m1 = np.arange(n, dtype=np.float64) / float(PPD)
        m2 = np.arange(n, dtype=np.float64) / float(PPD)
        tc = trans_cortex(vf, m1, m2, params, bg=0)
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
        ch_ink = reg.out_image[:, :, 0] >= 180
        thin = ch_ink & ~binary_erosion(ch_ink, iterations=1)
        if not np.any(thin):
            thin = ch_ink
        m = metrics(ch_ink)
        overlay = blend_on_chamber(
            bv,
            reg.out_image,
            thin=True,
            alpha=0.95,
            dark_chamber=True,
            ink_threshold=180,
        )
        vf_ann = rgb.copy()
        vf_ann[0:3, 0:3] = (255, 0, 0)
        if ink.any() and rgb[ink].mean() < 80:
            vf_ann[ink] = (255, 220, 40)
        rows.append(
            {
                "sess": sess,
                "spec": spec,
                "vf": vf_ann,
                "om": om_crop(tc.om),
                "overlay": overlay,
                "m": m,
            }
        )
        print(
            f"{sess}: ({spec.pos_x_deg:g}, {spec.pos_y_deg:g}) "
            f"H={m['H']:.1f} W={m['W']:.1f} asp={m['aspect']:.3f} n={m['n']}"
        )

    fig, axes = plt.subplots(3, 3, figsize=(11, 10), facecolor="white")
    for i, r in enumerate(rows):
        s = r["spec"]
        axes[i, 0].imshow(r["vf"], origin="upper")
        axes[i, 0].set_title(
            f"{r['sess']} VF\nD @ ({s.pos_x_deg:g}, {s.pos_y_deg:g})",
            fontsize=10,
        )
        axes[i, 1].imshow(r["om"], origin="upper")
        axes[i, 1].set_title("Schira OM", fontsize=10)
        axes[i, 2].imshow(r["overlay"], origin="upper")
        m = r["m"]
        axes[i, 2].set_title(
            f"Chamber · H={m['H']:.0f} W={m['W']:.0f} asp={m['aspect']:.2f}",
            fontsize=10,
        )
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"letter_D_white_1 · three 201118 locations · same Joanna params "
        f"a={a} α={alpha} k={k}",
        fontsize=12,
    )
    out = OUT_DIR / "legacy_letter_D_three_locations_201118.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
