"""Schira dashboard: set Schira's a, alpha, k; the registration to the camera is fitted (inner loop); see the mapped
stimulus on top of the measured VSD response and the landmark RMSE.

Run from the repo root:   scripts/py -m streamlit run schira_dashboard/schira_dash_app.py
(first time: scripts/py -m pip install streamlit;  and build the cache:  scripts/py schira_dashboard/prepare_data.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import schira_geometry as G                      # noqa: E402
from src.paths import resolve_data_path          # noqa: E402

st.set_page_config(page_title="Schira dashboard", layout="wide")
CACHE = HERE / "cache"


# ---------------------------------------------------------------- data
@st.cache_data
def load(monkey: str):
    maps = dict(np.load(CACHE / f"{monkey}_maps.npz"))
    cat = pd.read_csv(CACHE / f"{monkey}_catalog.csv")
    lm = pd.read_csv(CACHE / f"{monkey}_landmarks.csv").fillna({"pair_id": "", "note": "", "condition": ""})
    return maps, cat, lm

@st.cache_data
def stim_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(resolve_data_path(path, HERE.parent)).convert("RGB"))

monkeys = sorted(p.name.replace("_maps.npz", "") for p in CACHE.glob("*_maps.npz"))
if not monkeys:
    st.error("No cache found. Run:  scripts/py schira_dashboard/prepare_data.py"); st.stop()

# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header("Data")
monkey = sb.selectbox("Monkey", monkeys, format_func=str.capitalize)
maps, cat, LM = load(monkey)
session = sb.selectbox("Session", sorted(cat.session.unique()))
cs = cat[cat.session == session].reset_index(drop=True)
def label(i):
    txt = f"{cs.condition[i]} · {cs.stimulus_text[i].split(':', 1)[-1].strip()}"
    if "drawn_diameter_deg" in cs and np.isfinite(cs.drawn_diameter_deg[i]):
        txt += f"  [drawn Ø {cs.drawn_diameter_deg[i]:g}°" + (", radius reading]" if cs.get("variant", pd.Series([""] * len(cs)))[i] == "radius" else "]")
    return txt
si = sb.selectbox("Stimulus", range(len(cs)), format_func=label)
stim = cs.iloc[si]

sb.header("Schira parameters")
conv = sb.radio("α convention", ["repo", "thesis"], horizontal=True,
                format_func=lambda c: "P = α·θ (repo, Ayzenshtat)" if c == "repo" else "P = θ/α (thesis, legacy)")
for key, val in (("a", 0.74), ("alpha", 1.0), ("k", 1.0)):
    st.session_state.setdefault(key, val)
if "_optimum" in st.session_state:            # result of the outer loop, applied before the widgets are drawn
    st.session_state.update(st.session_state.pop("_optimum"))
def num(name, key, lo, hi, step, help_=None):
    return sb.number_input(name, min_value=lo, max_value=hi, step=step, key=key, format="%.3f", help=help_)
a = num("a (deg)", "a", 0.01, 50.0, 0.05, "foveal offset; small a = strong foveal magnification")
alpha = num("α", "alpha", 0.05, 20.0, 0.05, "polar-angle compression; see the convention switch above")

sb.header("Registration (inner loop)")
mode = sb.radio("Mode", ["similarity", "rigid"], horizontal=True,
                help="similarity: shift + rotation + scale (as in the thesis / Ayzenshtat); k has no effect.\n"
                     "rigid: shift + rotation at a fixed 0.17 mm/px; w is read in mm, so k sets the cortical scale.")
k = num("k (mm in rigid mode)", "k", 0.01, 200.0, 0.5,
        "scale of w. In similarity mode the fitted scale absorbs it completely.")
if mode == "similarity":
    sb.caption("k is inactive in similarity mode (the fitted scale absorbs it).")
mirror = sb.radio("Mirror", ["auto", False, True], horizontal=True,
                  format_func=lambda m: {"auto": "auto", False: "no", True: "yes"}[m])

sb.header("Overlays")
show_vm = sb.toggle("Vertical meridian (V1/V2 border)", True, key="t_vm")
show_hm = sb.toggle("Horizontal meridian", False, key="t_hm")

sb.header("Landmarks")
use_auto = sb.checkbox("automatic (response peaks)", True)
use_man = sb.checkbox("manual (Joanna)", True)
qmin = sb.slider("min. quality of automatic peaks (peak / noise SD)", 0.0, 20.0, 3.0, 0.5)

# ---------------------------------------------------------------- landmarks for this session
L = LM[(LM.session == session)].copy()
L = L[((L.source == "auto") & use_auto & ((L.quality >= qmin) | L.quality.isna())) | ((L.source == "manual") & use_man)]
L = L.reset_index(drop=True)
L.insert(0, "use", True)
with st.expander(f"Landmarks of {session} ({len(L)}) — untick to exclude", expanded=False):
    L = st.data_editor(L, hide_index=True, disabled=[c for c in L.columns if c != "use"], key=f"ed_{session}_{use_auto}_{use_man}_{qmin}")
U = L[L.use].reset_index(drop=True)
x, y = U.x_deg.values, U.y_deg.values
cam = (U.cam_col + 1j * U.cam_row).values

if sb.button("Optimise a, α" + (", k" if mode == "rigid" else "") + " (outer loop)", type="primary", disabled=len(U) < 3):
    with st.spinner("nested optimisation…"):
        o = G.optimize_schira(x, y, cam, U.pair_id.values, conv, mode, mirror, start=(a, alpha, k))
    st.session_state["_optimum"] = dict(a=round(o["a"], 4), alpha=round(o["alpha"], 4), k=round(o["k"], 4))
    st.session_state["_opt_msg"] = f"Outer loop: a {o['a']:.3f}, α {o['alpha']:.3f}, k {o['k']:.3f} → RMSE {o['rmse_px']:.2f} px"
    st.rerun()

# ---------------------------------------------------------------- fit
res = G.fit_session(x, y, cam, U.pair_id.values, a, alpha, k, conv, mode, mirror) if len(U) else None
img = stim_image(stim.image_path)
ix, iy = G.ink_to_deg(img)
resp = maps[f"{session}|{stim.condition}"]

st.title("Schira mapping: stimulus → V1 camera")
if "_opt_msg" in st.session_state:
    st.success(st.session_state.pop("_opt_msg"))
c = st.columns(6)
if res is not None and res.reg is not None:
    c[0].metric("RMSE (fit)", f"{res.rmse_px:.2f} px", f"{res.rmse_mm:.2f} mm", delta_color="off")
    c[1].metric("RMSE (leave-one-out)", f"{res.loo_rmse_px:.2f} px", f"{res.loo_rmse_px * G.MM_PER_PX:.2f} mm", delta_color="off")
    c[2].metric("Landmarks used", f"{res.n_used} / {len(U)}", "outside Schira's range: " + str(len(U) - res.n_used) if res.n_used < len(U) else None, delta_color="off")
    c[3].metric("Registration scale", f"{res.reg.px_per_w:.1f} px/w", f"rot {res.reg.rotation_deg:.0f}°, mirror {'yes' if res.reg.mirror else 'no'}", delta_color="off")
    ink_cam = res.reg.apply(G.schira_w(ix, iy, a, alpha, k, conv))
    c[4].metric("Overlap r (this stimulus)", f"{G.overlap_r(resp, G.ink_mask_on_camera(ink_cam)):.2f}",
                "landmark-free score", delta_color="off")
    if res.message:
        st.warning(res.message)
else:
    ink_cam = None
    st.warning("No registration: select at least 2 landmarks (similarity) or 1 (rigid)." if res is None else res.message)
alpha_repo = alpha if conv == "repo" else 1 / alpha
c[5].metric("α in both conventions", f"repo {alpha_repo:.3f}", f"thesis {1 / alpha_repo:.3f}", delta_color="off")

# ---------------------------------------------------------------- figure
cc = st.columns(2)
show_grid = cc[0].checkbox("0.5° Cartesian grid (0–3°)", False)
show_all = cc[1].checkbox("Landmarks of all stimuli", True)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw=dict(width_ratios=[1, 1.15]))
crop = 140
ax0.imshow(img[:crop, :crop], extent=(0, crop / G.PPD, -crop / G.PPD, 0))
ax0.set_xlabel("x (deg)"); ax0.set_ylabel("y (deg)"); ax0.set_title(f"Stimulus: {stim.stimulus_text}", fontsize=10, loc="left")
if True:
    here = U if show_all else U[U.condition == stim.condition]
    ax0.plot(here.x_deg, here.y_deg, "o", mfc="none", mec="#e34948", ms=8, mew=1.6)

v = np.nanpercentile(np.abs(resp), 99)
im = ax1.imshow(resp, cmap="RdBu_r", vmin=-v, vmax=v)
plt.colorbar(im, ax=ax1, fraction=0.046, label="ΔF/F (frames 35–46 minus 5–25)")
if ink_cam is not None:
    ax1.plot(ink_cam.real, ink_cam.imag, ".", ms=1.6, color="#111111", alpha=0.6)
    if show_vm or show_hm:   # model's lower vertical meridian (x = 0) = V1/V2 border; drawn from 0.02 to 6 deg eccentricity
        e = np.geomspace(0.02, 6, 300)
        vm = res.reg.apply(G.schira_w(np.full_like(e, 1e-6), -e, a, alpha, k, conv))
        if show_vm:
            ax1.plot(vm.real, vm.imag, "--", color="#333333", lw=1.4, alpha=0.55, label="vertical meridian (V1/V2), model")
        hm = res.reg.apply(G.schira_w(e, np.full_like(e, -1e-6), a, alpha, k, conv)) if show_hm else None
        if hm is not None:
            ax1.plot(hm.real, hm.imag, ":", color="#333333", lw=1.2, alpha=0.5, label="horizontal meridian, model")
        ax1.legend(loc="lower right", fontsize=8, framealpha=0.7)
    if show_grid:
        t = np.linspace(0.01, 3, 200)
        for g in np.arange(0, 3.01, 0.5):
            for gx, gy in ((np.full_like(t, max(g, 0.01)), -t), (t, np.full_like(t, -g))):
                p = res.reg.apply(G.schira_w(gx, gy, a, alpha, k, conv)); ax1.plot(p.real, p.imag, "-", lw=0.6, color="#555555")
    if res is not None and res.reg is not None:
        sel = np.arange(len(U)) if show_all else np.where(U.condition == stim.condition)[0]
        for i in sel:
            col = "#1f77b4" if U.source[i] == "manual" else "#2ca02c"
            ax1.plot([cam[i].real, res.pred[i].real], [cam[i].imag, res.pred[i].imag], "-", color="#e34948", lw=1.2)
            ax1.plot(cam[i].real, cam[i].imag, "o", color=col, ms=6, mec="white", mew=0.8)
            ax1.plot(res.pred[i].real, res.pred[i].imag, "x", color="#e34948", ms=7, mew=2)
ax1.set_xlim(-0.5, 99.5); ax1.set_ylim(99.5, -0.5)
ax1.set_title(f"{session} · {stim.condition}: VSD response with the mapped stimulus (black)\n"
              "landmarks: ● measured (green auto, blue manual) · ✕ model · red line = error", fontsize=10, loc="left")
plt.tight_layout()
st.pyplot(fig)
vtag = "_radius" if stim.get("variant", "") == "radius" else ""
fname = f"{session}_{stim.condition}{vtag}_{conv}_a{a:.3f}_alpha{alpha:.3f}_k{k:.3f}_{mode}.png"
if st.button("Save figure to schira_dashboard/figures/"):
    out = HERE / "figures" / fname; out.parent.mkdir(exist_ok=True)
    fig.suptitle(f"{monkey} {session} {stim.condition} · {conv} a={a:.3f} α={alpha:.3f} k={k:.3f} · {mode}"
                 + (f" · RMSE {res.rmse_px:.2f} px ({res.rmse_mm:.2f} mm), LOO {res.loo_rmse_px:.2f} px" if res is not None and res.reg is not None else ""),
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(out, dpi=130, bbox_inches="tight"); st.success(f"saved {out.relative_to(HERE.parent)}")

if res is not None and res.reg is not None:
    T = U[["source", "label", "condition", "x_deg", "y_deg", "cam_row", "cam_col", "quality"]].copy()
    T["model_row"] = res.pred.imag.round(1); T["model_col"] = res.pred.real.round(1)
    T["error_px"] = res.resid_px.round(2); T["error_mm"] = (res.resid_px * G.MM_PER_PX).round(3)
    T["LOO_error_px"] = np.round(res.loo_px, 2) if res.loo_px.size else np.nan
    st.subheader("Landmark errors")
    st.dataframe(T, hide_index=True, width="stretch")
st.caption("Schira: repo src/retinotopy/schira.py (double-sech). Camera px = 0.17 mm. "
           "Bar ends are unordered: the fit picks the pairing that gives the lower error. "
           "Manual 20/11/18 and 10/07/18 points are day-level and shown on every session of that day.")
