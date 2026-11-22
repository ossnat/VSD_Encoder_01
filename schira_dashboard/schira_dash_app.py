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
import thesis_fig12 as T                         # noqa: E402
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

def _img_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else resolve_data_path(p, HERE.parent)

@st.cache_data
def stim_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(_img_path(path)).convert("RGB"))

def _shape_label(r) -> str:
    """Readable name for one stimulus: shape · colour · size · position."""
    parts = r.uid.split("__")                        # e.g. filled_circle__rgb182__d0.8__x0.6_y-0.75__bg186
    bg = f"bg {parts[-1][2:]}" if parts[-1].startswith("bg") else ""
    if r.shape_type == "blank":
        return f"blank · {bg}"
    if str(r.shape_type).startswith("letter"):
        return " · ".join((f"letter {parts[0].split('_', 1)[1]}", f"({r.pos_x_deg:g}, {r.pos_y_deg:g})°", bg))
    col = parts[1].replace("rgb", "rgb ")
    size = "" if pd.isna(r.size_deg) else (f"{r.size_deg:g}° long" if "bar" in r.shape_type else f"Ø {r.size_deg:g}°")
    return " · ".join(s for s in (str(r.shape_type).replace("_", " "), col, size,
                                  f"({r.pos_x_deg:g}, {r.pos_y_deg:g})°", bg) if s)

@st.cache_data
def load_shapes(monkey: str, map_keys: tuple[str, ...]) -> pd.DataFrame:
    """One row per distinct stimulus image: the stimulus library (Data/.../stimuli/<monkey>/library) plus images that
    only the dashboard has (renderings under schira_dashboard/_retired/stimuli/<session>/<condition>__<tag>.png and any
    cached catalog image outside the library), de-duplicated by pixels. 'vsds' = the 'session|condition' keys with a
    response map for that image."""
    keys = set(map_keys)
    root = f"Data/VSD_Encoder_01/stimuli/{monkey}"
    lib = pd.read_parquet(resolve_data_path(f"{root}/library.parquet", HERE.parent))
    lk = pd.read_parquet(resolve_data_path(f"{root}/lookup.parquet", HERE.parent))
    rows = []
    for r in lib.rename(columns={"stimulus_uid": "uid"}).itertuples(index=False):
        sc = lk[lk.stimulus_uid == r.uid].sort_values(["h5_session", "condition_num"])
        vs = [f"{s}|{c}" for s, c in zip(sc.h5_session, sc.condition) if f"{s}|{c}" in keys]
        if r.shape_type == "blank":   # blank-trial recordings (session_<s>_blank.h5) of the sessions that list this blank
            vs += [f"{s}|blank" for s in sorted(set(sc.h5_session)) if f"{s}|blank" in keys]
        rows.append(dict(uid=r.uid, shape_type=r.shape_type, label=_shape_label(r), image_path=r.image_path,
                         source="library", vsds=vs, sessions=sorted(set(sc.h5_session))))
    # dashboard-only images
    cat = pd.read_csv(CACHE / f"{monkey}_catalog.csv")
    cand = [(str(p), p.parent.name, p.stem.split("__")[0], p.stem.split("__", 1)[-1])
            for p in sorted((HERE / "_retired/stimuli").glob("*/*.png"))]
    cand += [(r.image_path, r.session, r.condition, "catalog") for r in cat.itertuples(index=False)
             if "/library/" not in str(r.image_path)]
    lib_px = {r["uid"]: stim_image(r["image_path"]).tobytes() for r in rows}
    extra: dict[bytes, dict] = {}
    for path, s, c, tag in cand:
        px = stim_image(path).tobytes()
        if px in lib_px.values():
            continue                                         # identical to a library image: nothing new
        hit = cat[(cat.session == s) & (cat.condition == c)]
        e = extra.setdefault(px, dict(uid=f"dashboard__{s}_{c}_{tag}", shape_type=hit.shape_type.iloc[0] if len(hit) else "other",
                                      label=(hit.stimulus_text.iloc[0].split(":", 1)[-1].strip() if len(hit) else Path(path).stem)
                                      + f" [dashboard rendering: {tag.replace('_', ' ')} reading, pre-fix renderer]",
                                      image_path=path, source="dashboard", vsds=[], sessions=[]))
        if f"{s}|{c}" in keys and f"{s}|{c}" not in e["vsds"]:
            e["vsds"].append(f"{s}|{c}"); e["sessions"].append(s)
    # Joanna's thesis grids (make_thesis_fig12.py): never recorded, so they can be laid on ANY recording; the two thesis
    # Fig. 12 days come first.
    grid_rows = []
    if T.available():
        # '<session>|thesis' = the thesis Fig. 12 plate of that day as background, registered with that session
        thesis_bg = [f"{s}|thesis" for s in sorted({kk.split("|")[0] for kk in keys if kk[:6] in T.DATES})]
        any_vsd = thesis_bg + sorted(keys, key=lambda kk: (kk[:6] not in T.DATES, not kk.endswith("|blank"), kk))
        for g in T.grids().itertuples(index=False):
            grid_rows.append(dict(uid=g.uid, shape_type=g.shape_type, label=g.label, image_path=g.image_path,
                                  source="thesis grid", vsds=any_vsd, sessions=[], kind=g.kind))
    S = pd.DataFrame(rows + list(extra.values()) + grid_rows)
    S["n_vsd"] = S.vsds.map(len)
    S["_lib_first"] = S.source != "library"
    return S.sort_values(["shape_type", "_lib_first", "label"]).drop(columns="_lib_first").reset_index(drop=True)

monkeys = sorted(p.name.replace("_maps.npz", "") for p in CACHE.glob("*_maps.npz"))
if not monkeys:
    st.error("No cache found. Run:  scripts/py schira_dashboard/prepare_data.py"); st.stop()

# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header("Data")
monkey = sb.selectbox("Monkey", monkeys, format_func=str.capitalize)
maps, cat, LM = load(monkey)
SH = load_shapes(monkey, tuple(sorted(maps)))
types = sorted(SH.shape_type.unique())
n_rec = ((SH.n_vsd > 0) & (SH.source != "thesis grid")).sum()
sb.caption(f"{len(SH)} shapes ({(SH.source == 'library').sum()} library + {(SH.source == 'dashboard').sum()} dashboard-only"
           f" + {(SH.source == 'thesis grid').sum()} thesis grids), {n_rec} with a VSD recording")
stype = sb.selectbox("Shape type", ["all"] + types, key="shape_type", format_func=lambda t: t.replace("_", " "))
SHt = SH if stype == "all" else SH[SH.shape_type == stype]
def shape_label(u: str) -> str:
    r = SH[SH.uid == u].iloc[0]
    if r.source == "thesis grid":
        return f"{r.label}  — not recorded, any VSD"
    return f"{r.label}  — {r.n_vsd} VSD" + ("" if r.n_vsd == 1 else "s")
shape_uid = sb.selectbox("Shape", SHt.uid.tolist(), key="shape", format_func=shape_label)
shape = SH[SH.uid == shape_uid].iloc[0]
is_grid = shape.source == "thesis grid"
if shape.n_vsd == 0:
    st.title("Schira mapping: stimulus → V1 camera")
    why = ("no blank-trial file session_<s>_blank.h5 for its sessions; scripts/extract_blank_trials_h5.py makes one "
           "from the raw condsAN .mat" if shape.shape_type == "blank"
           else f"no processed recording for {', '.join(shape.sessions)}")
    st.info(f"**{shape.label}** has no VSD response in the dashboard cache ({why}). Pick another shape.")
    st.image(stim_image(shape.image_path)[:140, :140], width=280, caption=shape.uid)
    st.stop()

def vsd_label(key: str) -> str:
    s, c = key.split("|")
    if c == "thesis":
        return f"THESIS Fig. 12 plate, {s[:2]}/{s[2:4]}/{s[4:6]} (registered with {s})"
    if c == "blank":
        return f"{s} · blank trials"
    txt = cat[(cat.session == s) & (cat.condition == c)].stimulus_text
    return f"{s} · {c}" + (f" · {txt.iloc[0].split(':', 1)[-1].strip()}" if len(txt) else "")
vsd = sb.selectbox("Background to draw it on" if is_grid else "VSD recording of this shape", shape.vsds,
                   key="vsd", format_func=vsd_label,
                   help="The grid was never shown to the monkey: draw it on the thesis Fig. 12 plate, a blank-trial "
                        "recording or any stimulus recording. 20/11/18 and 10/07/18 (the thesis days) come first. "
                        "Blank trials exist only where session_<s>_blank.h5 was extracted." if is_grid else None)
session, condition = vsd.split("|")
on_thesis = condition == "thesis"
hit = cat[(cat.session == session) & (cat.condition == condition)]
stim = (hit.iloc[0].copy() if len(hit) else
        pd.Series(dict(session=session, condition=condition, shape_type=shape.shape_type)))
stim["image_path"], stim["stimulus_text"] = shape.image_path, shape.label    # draw the chosen shape on this VSD

sb.header("Schira parameters")
for key, val in (("a", 0.74), ("alpha", 1.0), ("k", 1.0), ("r", 1.0), ("conv", "repo"), ("mode", "similarity"),
                 ("pix_grid", "35ppd"), ("model", "free")):
    st.session_state.setdefault(key, val)
if "_optimum" in st.session_state:            # result of a fit / button, applied before the widgets are drawn
    st.session_state.update({"model": "free", **st.session_state.pop("_optimum")})
JF = G.JOINT_FIT
JOINT_STATE = dict(a=JF["a"], alpha=JF["alpha_thesis"], r=JF["r"], conv="thesis", mode="anchors", pix_grid="35ppd")

def _apply_model():
    if st.session_state["model"] == "joint":
        st.session_state.update(JOINT_STATE)
        st.session_state["_opt_msg"] = (f"Model: our joint fit — a {JF['a']}, α {JF['alpha_thesis']} (P = θ/α), "
                                        f"r {JF['r']}, Joanna's 2-anchor registration, 35 px/°")

model = sb.radio("Model", ["free", "joint"], horizontal=True, key="model", on_change=_apply_model,
                 format_func=lambda m: "Set by hand" if m == "free" else "Our joint fit (stretched Schira)",
                 help=f"Our joint fit (registration_revised/fit_stretch.py): one a, α, r shared by 10/07/18 and "
                      f"20/11/18, each day registered exactly through Joanna's two anchors, 35 px/° stimulus geometry: "
                      f"a {JF['a']}, α {JF['alpha_thesis']} (P = θ/α), r {JF['r']}. Its settings are locked while "
                      f"selected; switch to 'Set by hand' to change them (a fit result switches back automatically).")
lock = model == "joint"
if lock:   # keep the locked values even if a stale state slipped in
    st.session_state.update(JOINT_STATE)
conv = sb.radio("α convention", ["repo", "thesis"], horizontal=True, key="conv", disabled=lock,
                format_func=lambda c: "P = α·θ (repo, Ayzenshtat)" if c == "repo" else "P = θ/α (thesis, legacy)")
def num(name, key, lo, hi, step, help_=None, locked=False):
    return sb.number_input(name, min_value=lo, max_value=hi, step=step, key=key, format="%.3f", help=help_,
                           disabled=locked)
a = num("a (deg)", "a", 0.01, 50.0, 0.05, "foveal offset; small a = strong foveal magnification", lock)
alpha = num("α", "alpha", 0.05, 20.0, 0.05, "polar-angle compression; see the convention switch above", lock)
r_str = num("horizontal stretch r", "r", 0.05, 5.0, 0.01,
            "Stretched Schira: the visual field is read as (r·x, y) before the Schira equation, i.e. the matrix "
            "[[r, 0], [0, 1]]. r = 1 is the true geometry. Fitted on 10/07/18 + 20/11/18 with Joanna's anchors: "
            "r ≈ 0.54, the same factor her retino_mat grid has built in (0.415 / 0.77). Do not combine r < 1 with "
            "the retino_mat grid below: that would apply the squeeze twice.", lock)
G.set_stretch(r_str)
pix_grid = sb.radio("Stimulus px → degrees", ["35ppd", "retino"], horizontal=True, key="pix_grid", disabled=lock,
                    format_func=lambda g: "35 px/° (true)" if g == "35ppd" else "Joanna's retino_mat (thesis)",
                    help="35 px/°: the true geometry. Joanna's retino_mat: how her pipeline (and the thesis figures) read "
                         "the stimulus, ~84 px/° horizontally and ~45 px/° vertically. Applies to the stimulus, the "
                         "landmarks and the meridians.")
G.set_pixel_grid(pix_grid)

sb.header("Registration (inner loop)")
if pix_grid == "retino" and r_str < 0.99:
    sb.warning("retino_mat grid AND r < 1: the horizontal squeeze is applied twice.")
mode = sb.radio("Mode", ["similarity", "rigid", "anchors"], horizontal=True, key="mode", disabled=lock,
                format_func=lambda m: {"anchors": "2 anchors (Joanna)"}.get(m, m),
                help="similarity: shift + rotation + scale, least squares over all landmarks (Ayzenshtat); k has no effect.\n"
                     "rigid: shift + rotation at a fixed 0.17 mm/px; w is read in mm, so k sets the cortical scale.\n"
                     "2 anchors (Joanna): shift + rotation + scale solved exactly through two chosen landmarks; the others "
                     "only score the fit (her 'RMSD' = mean distance over the non-anchor landmarks). k has no effect.")
k = num("k (mm in rigid mode)", "k", 0.01, 200.0, 0.5,
        "scale of w. In similarity mode the fitted scale absorbs it completely.")
if mode != "rigid":
    sb.caption("k is inactive in this mode (the fitted scale absorbs it).")
mirror = sb.radio("Mirror", ["auto", False, True], horizontal=True,
                  format_func=lambda m: {"auto": "auto", False: "no", True: "yes"}[m])
thesis_ok = is_grid and session[:6] in T.DATES
camera = "landmarks"
if thesis_ok:
    camera = sb.radio("Camera for the grid", ["landmarks", "thesis"], horizontal=True, key="camera",
                      format_func=lambda c_: "session landmarks" if c_ == "landmarks" else "fit to thesis lines",
                      help="session landmarks: the registration fitted on this session's landmarks (as for every stimulus). "
                           "fit to thesis lines: for the current a, α, k, a similarity camera (shift, rotation, scale) "
                           "fitted to the thesis Fig. 12 lines, so only the Schira shape is compared.")

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
anchors = None
if mode == "anchors":
    sb.header("Anchors (2 landmarks)")
    if len(U) < 2:
        sb.warning("anchor mode needs at least 2 landmarks")
    else:
        lab = [f"{i}: {U.label[i]} ({U.x_deg[i]:.2f}, {U.y_deg[i]:.2f})°" for i in range(len(U))]
        dflt = G.day_anchors(session, x, y, U.source.values) or (0, 1)
        akey = f"anc_{session}_{len(U)}"
        ia = sb.selectbox("anchor A", range(len(U)), index=dflt[0], format_func=lab.__getitem__, key=akey + "_A")
        ib = sb.selectbox("anchor B", range(len(U)), index=dflt[1], format_func=lab.__getitem__, key=akey + "_B",
                          help="Default: Joanna's winning pair for this day (10/07/18: vertical-bar upper + circle "
                               "bottom; 20/11/18: stimulus px (49,18) + (15,18)), when the manual landmarks are selected.")
        anchors = (int(ia), int(ib))

fit_r = sb.checkbox("outer loop also fits r", mode == "anchors", key="fit_r")
if sb.button("Optimise a, α" + (", k" if mode == "rigid" else "") + (", r" if fit_r else "") + " (outer loop)",
             type="primary", disabled=len(U) < (4 if mode == "anchors" else 3)):
    with st.spinner("nested optimisation…"):
        o = G.optimize_schira(x, y, cam, U.pair_id.values, conv, mode, mirror, start=(a, alpha, k), r=r_str,
                              fit_r=fit_r, anchors=anchors)
    st.session_state["_optimum"] = dict(a=round(o["a"], 4), alpha=round(o["alpha"], 4), k=round(o["k"], 4),
                                        r=round(o["r"], 4))
    score = "mean non-anchor distance" if mode == "anchors" else "RMSE"
    st.session_state["_opt_msg"] = (f"Outer loop: a {o['a']:.3f}, α {o['alpha']:.3f}, k {o['k']:.3f}, r {o['r']:.3f} "
                                    f"→ {score} {o['cost_px']:.2f} px")
    st.rerun()

# ---- joint fit over days (shared a, alpha, r; each day its own anchors)
with sb.expander("Joint fit over days (shared a, α, r)"):
    st.caption("Our suggested method (registration_revised/fit_stretch.py): one a, α, r for all chosen sessions; each "
               "session is registered on its own manual (Joanna) landmarks, exactly through its day's two anchors. "
               "Score = mean distance over all non-anchor landmarks. α is reported in the convention chosen above.")
    man_sess = sorted(LM[(LM.source == "manual")].session.unique())
    first_of_day = [next(s_ for s_ in man_sess if s_.startswith(d)) for d in sorted({s_[:6] for s_ in man_sess})]
    jsess = st.multiselect("sessions (one per day is enough: the points are day-level)", man_sess, first_of_day, key="jsess")
    jr = st.checkbox("fit r (else keep the current r)", True, key="jfit_r")
    if st.button("Run joint fit", disabled=not jsess):
        days, bad = [], []
        for s_ in jsess:
            M = LM[(LM.session == s_) & (LM.source == "manual")].reset_index(drop=True)
            an = G.day_anchors(s_, M.x_deg.values, M.y_deg.values, M.source.values)
            if an is None:
                bad.append(s_); continue
            days.append(dict(x=M.x_deg.values, y=M.y_deg.values, cam=(M.cam_col + 1j * M.cam_row).values,
                             pair_ids=M.pair_id.values, anchors=an))
        if bad:
            st.error(f"no default anchor pair for {', '.join(bad)}")
        else:
            with st.spinner("joint fit…"):
                j = G.optimize_joint(days, convention="thesis", mirror=mirror, fit_r=jr, r=r_str)
            al_set = j["alpha"] if conv == "thesis" else 1 / j["alpha"]
            st.session_state["_optimum"] = dict(a=round(j["a"], 4), alpha=round(al_set, 4), r=round(j["r"], 4),
                                                mode="anchors")
            st.session_state["_opt_msg"] = (
                f"Joint fit over {', '.join(jsess)}: a {j['a']:.3f}, α {j['alpha']:.3f} (P = θ/α) = {1 / j['alpha']:.3f} "
                f"(repo), r {j['r']:.3f} → mean distance {j['cost_px']:.2f} px (per session: "
                + ", ".join(f"{s_} {v:.2f} px [{n} pts]" for s_, v, n in zip(jsess, j["per_day"], j["n_scored"])) + ")")
            st.rerun()

# ---------------------------------------------------------------- fit
res = G.fit_session(x, y, cam, U.pair_id.values, a, alpha, k, conv, mode, mirror, anchors=anchors) if len(U) else None
img = stim_image(stim.image_path)
ix, iy = G.ink_to_deg(img)
if on_thesis:   # background = the thesis plate (camera frame); grey values, not ΔF/F
    resp = T.thesis_image_cam(shape.kind, session[:6], up=1)
else:
    resp = maps[f"{session}|{stim.condition}"]
reg_used, thesis_cost = (res.reg if res is not None else None), np.nan
if thesis_ok and camera == "thesis" and reg_used is not None:
    _cen, _rad = T.chamber_circle(shape.kind, session[:6])
    reg_used, thesis_cost = G.fit_camera_to_lines(ix[::2], iy[::2], T.thesis_lines_cam(shape.kind, session[:6]), _cen, _rad,
                                                  a, alpha, k, conv, res.reg)

st.title("Schira mapping: stimulus → V1 camera")
if "_opt_msg" in st.session_state:
    st.success(st.session_state.pop("_opt_msg"))
c = st.columns(6)
if res is not None and res.reg is not None:
    if mode == "anchors":
        c[0].metric("Mean distance (non-anchor)", f"{res.mean_px:.2f} px", "Joanna's 'RMSD'", delta_color="off")
        c[1].metric("RMSE (non-anchor)", f"{res.rmse_px:.2f} px", f"{res.rmse_mm:.2f} mm", delta_color="off")
    else:
        c[0].metric("RMSE (fit)", f"{res.rmse_px:.2f} px", f"{res.rmse_mm:.2f} mm", delta_color="off")
        c[1].metric("RMSE (leave-one-out)", f"{res.loo_rmse_px:.2f} px", f"{res.loo_rmse_px * G.MM_PER_PX:.2f} mm", delta_color="off")
    c[2].metric("Landmarks used", f"{res.n_used} / {len(U)}", "outside Schira's range: " + str(len(U) - res.n_used) if res.n_used < len(U) else None, delta_color="off")
    c[3].metric("Registration scale", f"{res.reg.px_per_w:.1f} px/w", f"rot {res.reg.rotation_deg:.0f}°, mirror {'yes' if res.reg.mirror else 'no'}", delta_color="off")
    ink_cam = reg_used.apply(G.schira_w(ix, iy, a, alpha, k, conv))
    if thesis_ok:
        _c, _r = T.chamber_circle(shape.kind, session[:6])
        _d = T.chamfer(T.thesis_lines_cam(shape.kind, session[:6]), T.inside(ink_cam, _c, _r))
        c[4].metric("Distance to thesis lines", f"{np.mean(_d):.2f} px",
                    "camera fitted to thesis" if camera == "thesis" else "landmark camera", delta_color="off")
    else:
        c[4].metric("Overlap r (this stimulus)", f"{G.overlap_r(resp, G.ink_mask_on_camera(ink_cam)):.2f}",
                    "landmark-free score", delta_color="off")
    if res.message:
        st.warning(res.message)
else:
    ink_cam = None
    st.warning("No registration: select at least 2 landmarks (similarity) or 1 (rigid)." if res is None else res.message)
alpha_repo = alpha if conv == "repo" else 1 / alpha
c[5].metric("α in both conventions", f"repo {alpha_repo:.3f}", f"thesis {1 / alpha_repo:.3f} · r {r_str:.3f}", delta_color="off")

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

if on_thesis:
    im = ax1.imshow(resp, cmap="gray", vmin=0, vmax=255)
    plt.colorbar(im, ax=ax1, fraction=0.046, label="thesis Fig. 12 plate (grey level)")
else:
    v = np.nanpercentile(np.abs(resp), 99)
    im = ax1.imshow(resp, cmap="RdBu_r", vmin=-v, vmax=v)
    plt.colorbar(im, ax=ax1, fraction=0.046, label="ΔF/F (frames 35–46 minus 5–25)")
if ink_cam is not None:
    ax1.plot(ink_cam.real, ink_cam.imag, ".", ms=1.6 if not on_thesis else 2.2,
             color="#ff9f1c" if on_thesis else "#111111", alpha=0.6 if not on_thesis else 0.75)
    if show_vm or show_hm:   # model's lower vertical meridian (x = 0) = V1/V2 border; drawn from 0.02 to 6 deg eccentricity
        e = np.geomspace(0.02, 6, 300)
        vm = reg_used.apply(G.schira_w(np.full_like(e, 1e-6), -e, a, alpha, k, conv))
        if show_vm:
            ax1.plot(vm.real, vm.imag, "--", color="#333333", lw=1.4, alpha=0.55, label="vertical meridian (V1/V2), model")
        hm = reg_used.apply(G.schira_w(e, np.full_like(e, -1e-6), a, alpha, k, conv)) if show_hm else None
        if hm is not None:
            ax1.plot(hm.real, hm.imag, ":", color="#333333", lw=1.2, alpha=0.5, label="horizontal meridian, model")
        ax1.legend(loc="lower right", fontsize=8, framealpha=0.7)
    if show_grid:
        t = np.linspace(0.01, 3, 200)
        for g in np.arange(0, 3.01, 0.5):
            for gx, gy in ((np.full_like(t, max(g, 0.01)), -t), (t, np.full_like(t, -g))):
                p = reg_used.apply(G.schira_w(gx, gy, a, alpha, k, conv)); ax1.plot(p.real, p.imag, "-", lw=0.6, color="#555555")
    if res is not None and res.reg is not None and camera == "landmarks":
        sel = np.arange(len(U)) if show_all else np.where(U.condition == stim.condition)[0]
        for i in sel:
            col = "#1f77b4" if U.source[i] == "manual" else "#2ca02c"
            ax1.plot([cam[i].real, res.pred[i].real], [cam[i].imag, res.pred[i].imag], "-", color="#e34948", lw=1.2)
            ax1.plot(cam[i].real, cam[i].imag, "o", color=col, ms=6, mec="white", mew=0.8)
            ax1.plot(res.pred[i].real, res.pred[i].imag, "x", color="#e34948", ms=7, mew=2)
        for i in (res.anchors or ()):
            ax1.plot(cam[i].real, cam[i].imag, "*", color="#ffd400", ms=15, mec="k", mew=0.8)
ax1.set_xlim(-0.5, 99.5); ax1.set_ylim(99.5, -0.5)
if on_thesis:
    ax1.set_title(f"Thesis Fig. 12{T.PANEL[(shape.kind, session[:6])]} plate with the mapped grid (orange)\n"
                  + ("camera fitted to the thesis lines" if camera == "thesis" else f"camera = {session} landmark registration"),
                  fontsize=10, loc="left")
else:
    ax1.set_title(f"{session} · {stim.condition}: VSD response with the mapped stimulus (black)\n"
                  + ("landmarks: ● measured (green auto, blue manual) · ✕ model · red line = error"
                     + (" · ★ anchor" if mode == "anchors" else "") if camera == "landmarks"
                     else "camera fitted to the thesis lines (landmark errors hidden)"), fontsize=10, loc="left")
plt.tight_layout()
st.pyplot(fig)
vtag = {"dashboard": "_dashboard-shape", "thesis grid": f"_{shape.shape_type}"}.get(shape.source, "")
fname = f"{session}_{stim.condition}{vtag}_{conv}_a{a:.3f}_alpha{alpha:.3f}_k{k:.3f}_r{r_str:.3f}_{mode}.png"
if st.button("Save figure to schira_dashboard/figures/"):
    out = HERE / "figures" / fname; out.parent.mkdir(exist_ok=True)
    fig.suptitle(f"{monkey} {session} {stim.condition} · {conv} a={a:.3f} α={alpha:.3f} k={k:.3f} r={r_str:.3f} · {mode}"
                 + (f" · RMSE {res.rmse_px:.2f} px ({res.rmse_mm:.2f} mm), LOO {res.loo_rmse_px:.2f} px" if res is not None and res.reg is not None else ""),
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(out, dpi=130, bbox_inches="tight"); st.success(f"saved {out.relative_to(HERE.parent)}")

# ---------------------------------------------------------------- thesis Fig. 12 (grids only)
if is_grid:
    kind, date = shape.kind, session[:6]
    st.subheader(f"Thesis Fig. 12{T.PANEL.get((kind, date), '')}: thesis vs Joanna's pipeline vs this dashboard")
    if date not in T.DATES:
        st.info("Thesis Fig. 12 shows the chambers of 20/11/18 and 10/07/18 only: pick a VSD recording from one of those "
                "days (they are listed first) to compare.")
    else:
        jp = T.joanna_params(date)
        cen, rad = T.chamber_circle(kind, date)
        th_pts = T.thesis_lines_cam(kind, date)
        jo_pts = T.inside(T.joanna_lines_cam(kind, date), cen, rad)
        our_pts = T.inside(ink_cam, cen, rad) if ink_cam is not None else np.array([], complex)
        d_jo, d_our = T.chamfer(th_pts, jo_pts), T.chamfer(th_pts, our_pts)
        m = st.columns([1, 1, 1.3])
        m[0].metric("Thesis ↔ Joanna's pipeline (re-run)", f"{d_jo[0]:.2f} px", f"back {d_jo[1]:.2f} px", delta_color="off")
        m[1].metric("Thesis ↔ this dashboard", f"{d_our[0]:.2f} px" if np.isfinite(d_our[0]) else "–",
                    f"back {d_our[1]:.2f} px" if np.isfinite(d_our[1]) else None, delta_color="off")
        alpha_here = jp["alpha_thesis"] if conv == "thesis" else 1 / jp["alpha_thesis"]
        if m[2].button(f"Use Joanna's {date} values: a {jp['a']:g}, α {jp['alpha_thesis']:g} (thesis) = {1 / jp['alpha_thesis']:.3f} (repo)"):
            st.session_state["_optimum"] = dict(a=float(jp["a"]), alpha=round(float(alpha_here), 4), k=float(jp["k"]))
            st.session_state["_opt_msg"] = f"Set a = {jp['a']:g}, α = {alpha_here:.4g} ({conv} convention), k = {jp['k']:g}"
            st.rerun()
        fb = st.columns(2)
        fit_cam = None
        if fb[0].button("Fit a, α to the thesis lines — session landmark camera", disabled=res is None or res.reg is None,
                        help="Searches a and α (and k in rigid mode) so that the grid, registered with this session's "
                             "landmarks, lies on the thesis lines. Uses the current 'Stimulus px → degrees' setting."):
            fit_cam = "session"
        if fb[1].button("Fit a, α to the thesis lines — free camera", disabled=res is None or res.reg is None,
                        help="Searches a and α together with a similarity camera fitted to the thesis lines: the best the "
                             "Schira shape alone can do. Switches 'Camera for the grid' to 'fit to thesis lines'."):
            fit_cam = "free"
        if fit_cam:
            ar = alpha if conv == "repo" else 1 / alpha
            starts = [(a, ar), (jp["a"], 1 / jp["alpha_thesis"]), (0.3, 1.0), (1.5, 0.7)]
            with st.spinner("fitting to the thesis lines…"):
                fr = G.fit_schira_to_lines(ix[::2], iy[::2], th_pts, cen, rad,
                                           (x, y, cam, U.pair_id.values), starts, k, camera=fit_cam, mode=mode,
                                           mirror=mirror, fit_k=(mode == "rigid" and fit_cam == "session"), anchors=anchors)
            if fr["ok"]:
                al_set = fr["alpha_repo"] if conv == "repo" else 1 / fr["alpha_repo"]
                upd = dict(a=round(fr["a"], 4), alpha=round(al_set, 4), k=round(fr["k"], 4))
                if fit_cam == "free":
                    upd["camera"] = "thesis"
                st.session_state["_optimum"] = upd
                st.session_state["_opt_msg"] = (
                    f"Fit to thesis {T.PANEL[(kind, date)]} ({'free camera' if fit_cam == 'free' else session + ' landmark camera'}, "
                    f"{'retino_mat' if pix_grid == 'retino' else '35 px/°'}): a {fr['a']:.3f}, α {fr['alpha_repo']:.3f} (repo) = "
                    f"{1 / fr['alpha_repo']:.3f} (thesis), k {fr['k']:.3g} → {fr['cost']:.2f} px (mean of both directions)")
                st.rerun()
            else:
                st.error("Fit failed: no registration for the start values.")
        bvi = T.bv(date); ext = (-0.5, 99.5, 99.5, -0.5)
        lo, hi = np.percentile(bvi, [1, 99]); bv_kw = dict(cmap="gray", vmin=lo, vmax=lo + 2.2 * (hi - lo))   # darkened
        f12, ax12 = plt.subplots(2, 2, figsize=(11, 11)); ax12 = ax12.ravel()
        ax12[0].imshow(T.thesis_image_cam(kind, date), cmap="gray", extent=ext, vmin=0, vmax=255)
        ax12[0].set_title(f"Thesis Fig. 12{T.PANEL[(kind, date)]}\n(registered onto the chamber image)", fontsize=9, loc="left")
        for ax, pts, col, ttl in ((ax12[1], jo_pts, "white", f"Joanna's prediction: her pipeline re-run\n"
                                   f"a {jp['a']:g}, α {jp['alpha_thesis']:g} (thesis), 2-point registration"),
                                  (ax12[2], our_pts, "#ff9f1c", f"This dashboard: a {a:.3g}, α {alpha:.3g} ({conv}), "
                                   f"k {k:.3g}, {'retino_mat' if pix_grid == 'retino' else '35 px/°'}\n"
                                   + ("camera fitted to the thesis lines" if camera == "thesis"
                                      else f"{session} landmark registration ({mode})"))):
            ax.imshow(bvi, extent=ext, **bv_kw)
            ax.plot(pts.real, pts.imag, ".", ms=4.5 if col == "white" else 3, color=col, mec="none")
            ax.set_title(ttl, fontsize=9, loc="left")
        ax12[3].imshow(bvi, extent=ext, **bv_kw)
        for pts, col, lab in ((th_pts, "#e34948", "thesis"), (jo_pts, "#17becf", "Joanna re-run"), (our_pts, "#ff9f1c", "dashboard")):
            ax12[3].plot(pts.real, pts.imag, ".", ms=2.4, color=col, mec="none", alpha=0.85, label=lab)
        ax12[3].add_patch(plt.Circle((cen.real, cen.imag), rad, fill=False, ls=":", color="#dddddd", lw=0.8))
        ax12[3].legend(loc="lower left", fontsize=7, markerscale=6, framealpha=0.8)
        ax12[3].set_title("All three on the chamber image", fontsize=9, loc="left")
        for ax in ax12:
            ax.set_xlim(-0.5, 99.5); ax.set_ylim(99.5, -0.5); ax.set_xticks([]); ax.set_yticks([])
        f12.tight_layout(); st.pyplot(f12)
        st.caption("Distances: mean distance (camera px, 1 px = 0.17 mm) from each thesis line pixel to the nearest model "
                   "line, and back (inside the chamber circle). Each distance is capped at 6 px, and lines are not matched by "
                   "identity. Joanna's pipeline converts stimulus pixels to degrees with her retino_mat grid "
                   "(about 84 px/° horizontally and 45 px/° vertically). The dashboard uses 35 px/°, so the same a and α "
                   "give a different map (see docs/schira_history/schira_audit.md). The thesis plate is registered "
                   "onto BV_resized.png by its blood vessels (schira_dashboard/make_thesis_fig12.py).")

# ---------------------------------------------------------------- same shape on every VSD it was shown in
cmp_keys = list(shape.vsds)
if is_grid:   # never recorded: one panel per session, the selected condition where that session has it
    cmp_keys = []
    in_maps = [kk for kk in shape.vsds if kk in maps]
    for s2 in sorted({kk.split("|")[0] for kk in in_maps}, key=lambda s_: (s_[:6] not in T.DATES, s_)):
        k2 = f"{s2}|{condition}" if f"{s2}|{condition}" in maps else next(kk for kk in in_maps if kk.startswith(s2 + "|"))
        cmp_keys.append(k2)
n_cmp = len(cmp_keys)
if n_cmp > 1 and st.toggle(f"Compare this shape on {'one recording of each of the ' + str(n_cmp) + ' sessions' if is_grid else 'all ' + str(n_cmp) + ' of its VSD recordings'}",
                           True, key="t_cmp"):
    st.subheader("Same shape, same Schira parameters, each session's own registration")
    st.caption("Each panel fits the registration on that session's landmarks with the sidebar filters. "
               "Rows you untick in the Landmarks table apply only to the selected session. ★ = selected VSD."
               + (f" Grids: each session shows {condition} where it has it, otherwise its first condition." if is_grid else ""))
    ncol = min(4, n_cmp); nrow = -(-n_cmp // ncol)
    fg, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.7 * nrow), squeeze=False)
    w_ink = G.schira_w(ix, iy, a, alpha, k, conv)
    for ax, key in zip(axs.flat, cmp_keys):
        s2, c2 = key.split("|")
        if s2 == session:
            U2 = U
        else:
            L2 = LM[LM.session == s2]
            U2 = L2[((L2.source == "auto") & use_auto & ((L2.quality >= qmin) | L2.quality.isna()))
                    | ((L2.source == "manual") & use_man)].reset_index(drop=True)
        an2 = anchors if s2 == session else (G.day_anchors(s2, U2.x_deg.values, U2.y_deg.values, U2.source.values)
                                             if len(U2) else None)
        mode2 = mode if (mode != "anchors" or an2 is not None) else "similarity"
        r2 = G.fit_session(U2.x_deg.values, U2.y_deg.values, (U2.cam_col + 1j * U2.cam_row).values, U2.pair_id.values,
                           a, alpha, k, conv, mode2, mirror, anchors=an2) if len(U2) else None
        R2 = maps[key]; v2 = np.nanpercentile(np.abs(R2), 99)
        ax.imshow(R2, cmap="RdBu_r", vmin=-v2, vmax=v2)
        info = f"no registration ({len(U2)} landmarks)"
        if r2 is not None and r2.reg is not None:
            ic = r2.reg.apply(w_ink)
            ax.plot(ic.real, ic.imag, ".", ms=1.2, color="#111111", alpha=0.6)
            info = ((f"mean {r2.mean_px:.1f} px (anchors)" if mode2 == "anchors" else f"RMSE {r2.rmse_px:.1f} px")
                    + (" [similarity: no anchors]" if mode2 != mode else "") + f" · n={r2.n_used} · "
                    f"overlap r {G.overlap_r(R2, G.ink_mask_on_camera(ic)):.2f}")
        ax.set_xlim(-0.5, 99.5); ax.set_ylim(99.5, -0.5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(("★ " if key == vsd else "") + f"{s2} · {c2}\n{info}", fontsize=9, loc="left")
    for ax in list(axs.flat)[n_cmp:]:
        ax.axis("off")
    fg.tight_layout(); st.pyplot(fg)

if res is not None and res.reg is not None:
    TE = U[["source", "label", "condition", "x_deg", "y_deg", "cam_row", "cam_col", "quality"]].copy()
    TE["model_row"] = res.pred.imag.round(1); TE["model_col"] = res.pred.real.round(1)
    TE["error_px"] = res.resid_px.round(2); TE["error_mm"] = (res.resid_px * G.MM_PER_PX).round(3)
    TE["LOO_error_px"] = np.round(res.loo_px, 2) if res.loo_px.size and mode != "anchors" else np.nan
    if mode == "anchors":
        TE.insert(0, "anchor", [i in (res.anchors or ()) for i in range(len(TE))])
        TE = TE.drop(columns="LOO_error_px")
    st.subheader("Landmark errors")
    st.dataframe(TE, hide_index=True, width="stretch")
st.caption("Schira: repo src/retinotopy/schira.py (double-sech), visual field read as (r·x, y) first (horizontal "
           "stretch r). Camera px = 0.17 mm. "
           "Bar ends are unordered: the fit picks the pairing that gives the lower error. "
           "Manual 20/11/18 and 10/07/18 points are day-level and shown on every session of that day.")
