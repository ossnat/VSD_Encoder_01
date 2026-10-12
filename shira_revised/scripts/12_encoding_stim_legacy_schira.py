#!/usr/bin/env python3
"""Push encoding-project letter D + triangle through legacy Schira+reg.

Uses ``src.stimuli.render`` (same glyphs / positions as encoding), then
Joanna winners:
  letter_D_white_1 (201118a @ 1.0, −0.9) → a=0.72 α=1.5 k=1
  black_triangle_contour_0.4 (100718 @ 0.6, −0.75) → a=0.74 α=1.0 k=1

Also shows the legacy MATLAB ``create_letter_stim`` D for side-by-side shape.
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
from shira_revised.src.legacy_stimuli import PPD, create_letter_stim
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
ARTIFACTS = PATHS["artifacts"]
OUT_DIR = REPO / "shira_revised" / "artifacts"


def m1_m2_top_left(n_rows: int, n_cols: int, ppd: float):
    return (
        np.arange(n_cols, dtype=np.float64) / ppd,
        np.arange(n_rows, dtype=np.float64) / ppd,
    )


def ink_from_rgb(rgb: np.ndarray, *, bg: int = 128, tol: int = 20) -> np.ndarray:
    """Binary ink = pixels away from gray background (letter BMP ink is dark)."""
    g = rgb.astype(np.float64).mean(axis=2)
    return np.abs(g - float(bg)) >= float(tol)


def metrics(ink: np.ndarray) -> dict:
    ys, xs = np.where(ink)
    if ys.size == 0:
        return {"H": 0.0, "W": 0.0, "aspect": 0.0, "n": 0}
    h = float(np.percentile(ys, 95) - np.percentile(ys, 5))
    w = float(np.percentile(xs, 95) - np.percentile(xs, 5))
    return {"H": h, "W": w, "aspect": h / max(w, 1e-6), "n": int(ys.size)}


def transform_ink(
    vf_ink_rgb: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    *,
    a: float,
    alpha: float,
    k: float,
    vf_pix: list,
    brain: list,
):
    params = LegacySchiraParams(
        a=a, alpha=alpha, k=k, alpha_mode="divide", vf_abs=True
    )
    n = vf_ink_rgb.shape[0]
    tc = trans_cortex(vf_ink_rgb, m1, m2, params, bg=0)
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
    m = metrics(ink)
    return {"params": params, "reg": reg, "tc": tc, "ink": ink, "thin": thin, **m}


def catalog_spec(stimulus_id: str, prefer_prefix: str):
    cfg = yaml.safe_load((project_root() / "configs" / "default.yaml").read_text())
    root = resolve_data_path(cfg["paths"]["encoder_data_root"])
    cat = attach_stimulus_ids(load_full_encoder_catalog(root, monkey="Gandalf"))
    sub = cat[cat["stimulus_id"].astype(str) == stimulus_id]
    if sub.empty:
        raise RuntimeError(f"No catalog rows for {stimulus_id}")
    on = sub[sub["h5_session"].astype(str).str.startswith(prefer_prefix)]
    row = (on if not on.empty else sub).sort_values("h5_session").iloc[0]
    return stimulus_spec_from_mapping(row.to_dict()), row


def vf_panel(rgb: np.ndarray, ink: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[0:3, 0:3] = (255, 0, 0)
    # emphasize ink for dark shapes
    if ink.any() and rgb[ink].mean() < 80:
        hi = out.copy()
        hi[ink] = (255, 220, 40)
        return hi
    return out


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

    # --- encoding letter D (201118 Session a) ---
    d_spec, d_row = catalog_spec("letter_D_white_1", "201118a")
    d_rgb = render_stimulus(d_spec, render_cfg)
    d_ink = ink_from_rgb(d_rgb, bg=render_cfg.background_gray)
    d_vf = np.zeros_like(d_rgb)
    d_vf[d_ink] = 255
    n_d = d_vf.shape[0]
    m1d, m2d = m1_m2_top_left(n_d, n_d, float(PPD))
    jo_d = CFG["joanna_201118"]
    d_res = transform_ink(
        d_vf,
        m1d,
        m2d,
        a=float(jo_d["a"]),
        alpha=float(jo_d["alpha"]),
        k=float(jo_d["k"]),
        vf_pix=[list(p) for p in jo_d["vf_pix_anchors"]],
        brain=[list(p) for p in jo_d["brain_anchors"]],
    )
    bv_d = np.asarray(Image.open(LEGACY_ROOT / ARTIFACTS["chamber_201118"]))
    d_ch = blend_on_chamber(
        bv_d,
        d_res["reg"].out_image,
        thin=True,
        alpha=0.95,
        dark_chamber=True,
        ink_threshold=180,
    )

    # legacy MATLAB letter placement (same BMP, 35×35 crop) for comparison
    leg_rgb = create_letter_stim("D", (1.0, -0.9))[:n_d, :n_d].copy()
    leg_ink = leg_rgb[:, :, 0] >= 200
    leg_vf = np.zeros_like(leg_rgb)
    leg_vf[leg_ink] = 255
    leg_res = transform_ink(
        leg_vf,
        m1d,
        m2d,
        a=float(jo_d["a"]),
        alpha=float(jo_d["alpha"]),
        k=float(jo_d["k"]),
        vf_pix=[list(p) for p in jo_d["vf_pix_anchors"]],
        brain=[list(p) for p in jo_d["brain_anchors"]],
    )
    leg_ch = blend_on_chamber(
        bv_d,
        leg_res["reg"].out_image,
        thin=True,
        alpha=0.95,
        dark_chamber=True,
        ink_threshold=180,
    )

    # --- encoding triangle (100718) ---
    t_spec, t_row = catalog_spec("black_triangle_contour_0.4", "100718a")
    t_rgb = render_stimulus(t_spec, render_cfg)
    t_ink = ink_from_rgb(t_rgb, bg=render_cfg.background_gray)
    t_vf = np.zeros_like(t_rgb)
    t_vf[t_ink] = 255
    n_t = t_vf.shape[0]
    m1t, m2t = m1_m2_top_left(n_t, n_t, float(PPD))
    jo_t = CFG["joanna_100718"]
    t_res = transform_ink(
        t_vf,
        m1t,
        m2t,
        a=float(jo_t["a"]),
        alpha=float(jo_t["alpha"]),
        k=float(jo_t["k"]),
        vf_pix=[list(p) for p in jo_t["vf_pix_anchors"]],
        brain=[list(p) for p in jo_t["brain_anchors"]],
    )
    bv_t = np.asarray(Image.open(LEGACY_ROOT / ARTIFACTS["chamber_100718"]))
    t_ch = blend_on_chamber(
        bv_t,
        t_res["reg"].out_image,
        thin=True,
        alpha=0.95,
        dark_chamber=True,
        ink_threshold=180,
    )

    print(
        f"encoding D VF ink: n={d_ink.sum()} bbox={metrics(d_ink)} "
        f"→ chamber H={d_res['H']:.1f} W={d_res['W']:.1f} asp={d_res['aspect']:.3f}"
    )
    print(
        f"legacy   D VF ink: n={leg_ink.sum()} bbox={metrics(leg_ink)} "
        f"→ chamber H={leg_res['H']:.1f} W={leg_res['W']:.1f} asp={leg_res['aspect']:.3f}"
    )
    print(
        f"encoding △ VF ink: n={t_ink.sum()} bbox={metrics(t_ink)} "
        f"→ chamber H={t_res['H']:.1f} W={t_res['W']:.1f} asp={t_res['aspect']:.3f}"
    )
    print(
        f"D pos=({d_spec.pos_x_deg}, {d_spec.pos_y_deg}) "
        f"session={d_row['h5_session']}  "
        f"△ pos=({t_spec.pos_x_deg}, {t_spec.pos_y_deg}) "
        f"session={t_row['h5_session']} size={t_spec.size_deg}"
    )

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), facecolor="white")

    axes[0, 0].imshow(vf_panel(d_rgb, d_ink), origin="upper")
    axes[0, 0].set_title(
        f"Encoding letter_D_white_1\n"
        f"{d_row['h5_session']} ({d_spec.pos_x_deg:g},{d_spec.pos_y_deg:g}) 1° box"
    )
    axes[0, 1].imshow(om_crop(d_res["tc"].om), origin="upper")
    axes[0, 1].set_title(
        f"Schira OM · Joanna 201118\na={jo_d['a']} α={jo_d['alpha']} k={jo_d['k']}"
    )
    axes[0, 2].imshow(d_ch, origin="upper")
    axes[0, 2].set_title(
        f"Chamber 201118 · encoding D\n"
        f"H={d_res['H']:.0f} W={d_res['W']:.0f} asp={d_res['aspect']:.2f}"
    )
    axes[0, 3].imshow(leg_ch, origin="upper")
    axes[0, 3].set_title(
        f"Chamber · legacy create_letter_stim D\n"
        f"H={leg_res['H']:.0f} W={leg_res['W']:.0f} asp={leg_res['aspect']:.2f}"
    )

    axes[1, 0].imshow(vf_panel(t_rgb, t_ink), origin="upper")
    axes[1, 0].set_title(
        f"Encoding black_triangle_contour_0.4\n"
        f"{t_row['h5_session']} ({t_spec.pos_x_deg:g},{t_spec.pos_y_deg:g}) "
        f"size={t_spec.size_deg:g}°"
    )
    axes[1, 1].imshow(om_crop(t_res["tc"].om), origin="upper")
    axes[1, 1].set_title(
        f"Schira OM · Joanna 100718\na={jo_t['a']} α={jo_t['alpha']} k={jo_t['k']}"
    )
    axes[1, 2].imshow(t_ch, origin="upper")
    axes[1, 2].set_title(
        f"Chamber 100718 · encoding △\n"
        f"H={t_res['H']:.0f} W={t_res['W']:.0f} asp={t_res['aspect']:.2f}"
    )
    # zoom VF crop around triangle for shape inspection
    ys, xs = np.where(t_ink)
    if ys.size:
        pad = 18
        r0, r1 = max(0, ys.min() - pad), min(n_t, ys.max() + pad + 1)
        c0, c1 = max(0, xs.min() - pad), min(n_t, xs.max() + pad + 1)
        axes[1, 3].imshow(vf_panel(t_rgb, t_ink)[r0:r1, c0:c1], origin="upper")
    else:
        axes[1, 3].imshow(t_rgb, origin="upper")
    axes[1, 3].set_title("VF zoom (yellow=ink)\nequilat. tip → +x")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Encoding-project stimuli → legacy Schira (θ/α, vf_abs) + Joanna 2-pt reg",
        fontsize=12,
    )
    out = OUT_DIR / "legacy_encoding_D_and_triangle.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    Image.fromarray(d_ch).save(OUT_DIR / "legacy_encoding_letter_D_chamber.png")
    Image.fromarray(t_ch).save(OUT_DIR / "legacy_encoding_triangle_chamber.png")
    Image.fromarray(d_rgb).save(OUT_DIR / "encoding_letter_D_vf.png")
    Image.fromarray(t_rgb).save(OUT_DIR / "encoding_triangle_vf.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
