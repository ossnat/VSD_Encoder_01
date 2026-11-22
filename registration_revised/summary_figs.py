"""Summary figures for the 10/07/18 registration points.

  figs/summary_points_on_thesis.png   the xlsx points (top-left reading) on thesis Fig. 11B, paired with the thesis marks,
                                      distances, Joanna's metric and the RMSE; and Joanna's model prediction (thesis
                                      winner) of her optimisation points on the same plate
  figs/summary_points_on_stimuli.png  per stimulus: the VF points on the stimulus | the measured VSD point (red) and
                                      Joanna's model prediction (cyan) on the 100718b response map, with the error

Joanna's procedure (legacy Code/: leftHem_trans_opt.m, find_optimization_pts.m, calc_rmsd.m, mean_rmsd.m), re-run
with the Python port of trans_cortex (shira_revised/src/legacy_trans_cortex.py):
  1. trans_cortex on the 301 x 512 canvas with her retino_mat grid -> cortical coords (xcoord, ycoord) per canvas pixel
  2. 2 anchor points: stimulus pixel -> (xcoord, ycoord); camera point (col, 101 - row) [low hemifield flip];
     exact 2-point 'nonreflectivesimilarity' (shift, rotation, scale)
  3. every other point: round((xcoord, ycoord) * tform), flip back -> predicted camera (row, col), 1-based
  4. distance d_i = |predicted - measured| (camera px). Her reported 'RMSD' (thesis: 2.56 px for 10/07/18) is
     mean_rmsd = sum(d) / (n - 2), the MEAN distance over the non-anchor points (anchors are 0 by construction);
     calc_rmsd's own value is sqrt(sum(d) / n). We also give the true RMSE, sqrt(mean(d^2)).
Thesis winner for 10/07/18: a 0.74, alpha 1.0 (P = theta/alpha), k 1, anchors = vertical-bar upper (stim 9,21 -> cam
46,24) and circle bottom (37,21 -> 36,58), points file opt_points_25_09_23_23_01.mat.

Run:  scripts/py registration_revised/summary_figs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
import points                                     # noqa: E402
import make_figs as MF                            # noqa: E402
from shira_revised.src.legacy_trans_cortex import m1_m2_from_retino_mat, trans_cortex   # noqa: E402
from shira_revised.src.legacy_warp import load_retino_mat_crop                          # noqa: E402
from shira_revised.src.legacy_schira import LegacySchiraParams                          # noqa: E402

WINNER = dict(a=0.74, alpha=1.0, k=1.0)
WINNER_FILE = MF.LEG / "VSDI points/100718/opt_points_25_09_23_23_01.mat"
ANCHORS = [(9, 21), (37, 21)]                     # stimulus (row, col), 1-based
SESSION = "100718b"
NAMES = {(26, 21): "0.05/0.1 point", (9, 21): "vertical bar upper", (43, 21): "vertical bar bottom",
         (26, 38): "horizontal bar right", (26, 35): "triangle right", (37, 21): "circle bottom",
         (26, 4): "horizontal bar left", (14, 14): "triangle top", (38, 14): "triangle bottom",
         (14, 21): "circle top"}


# ---------------------------------------------------------------- Joanna's pipeline
def joanna_model(a=WINNER["a"], alpha=WINNER["alpha"], k=WINNER["k"]):
    m1, m2 = m1_m2_from_retino_mat(load_retino_mat_crop(), 301, 512)
    r = trans_cortex(np.zeros((301, 512, 3), np.uint8), m1, m2, LegacySchiraParams(a=a, alpha=alpha, k=k), quick_mode=True)
    return r.xcoord, r.ycoord


def joanna_predict(X, Y, stim_rc, anchor_stim_rc, anchor_cam_rc):
    """Predicted camera (row, col), 1-based, for stimulus pixels (row, col), 1-based."""
    u = lambda rc: X[rc[0] - 1, rc[1] - 1] + 1j * Y[rc[0] - 1, rc[1] - 1]
    ua = np.array([u(p) for p in anchor_stim_rc])
    va = np.array([c + 1j * (100 - r + 1) for r, c in anchor_cam_rc])
    c1 = (va[1] - va[0]) / (ua[1] - ua[0]); c0 = va[0] - c1 * ua[0]
    t = c1 * np.array([u(p) for p in stim_rc]) + c0
    t = np.round(t.real) + 1j * np.round(t.imag)
    return 100 - t.imag + 1, t.real


MEAS = "lime"                                     # measured points (xlsx)


def our_predict(stim_rc, anchor_stim_rc, anchor_cam_rc, ref_r=None, ref_c=None, mirror=None):
    """Our pipeline (schira_dashboard/schira_geometry.py): stimulus px -> degrees at the true 35 px/deg -> repo Schira
    (same a, alpha [thesis convention], k) -> similarity camera fitted exactly on the same 2 anchors. No rounding.
    The mirror (a 2-point fit cannot tell) is the one closest to Joanna's own predictions (ref), not to the measured
    points. Returns predicted camera (row, col), 1-based, and the mirror used."""
    sys.path.insert(0, str(REPO / "schira_dashboard")); import schira_geometry as G   # noqa: E402
    G.set_pixel_grid("35ppd")
    deg = lambda rcs: (np.array([c for _, c in rcs]) / G.PPD, -np.array([r for r, _ in rcs]) / G.PPD)
    w = G.schira_w(*deg(stim_rc), WINNER["a"], WINNER["alpha"], WINNER["k"], "thesis")
    wa = G.schira_w(*deg(anchor_stim_rc), WINNER["a"], WINNER["alpha"], WINNER["k"], "thesis")
    ca = np.array([(c - 1) + 1j * (r - 1) for r, c in anchor_cam_rc])
    best = None
    for mr in ((False, True) if mirror is None else (mirror,)):
        u = G.fit_registration(wa, ca, "similarity", mr).apply(w)
        e = 0.0 if ref_r is None else np.nansum(np.hypot(u.imag + 1 - ref_r, u.real + 1 - ref_c))
        if best is None or e < best[0]:
            best = (e, u, mr)
    return best[1].imag + 1, best[1].real + 1, best[2]


def metrics(d, n_anchor=2):
    d = np.asarray(d, float)
    return dict(joanna=d.sum() / (len(d) - n_anchor) if n_anchor else d.mean(),
                calc_rmsd=np.sqrt(d.sum() / len(d)), rmse=np.sqrt(np.mean(d[d > 0] ** 2)) if n_anchor else np.sqrt(np.mean(d ** 2)))


def mark(ax, r, c, color="red", size=3, **kw):
    for rr, cc in zip(np.atleast_1d(r), np.atleast_1d(c)):
        ax.add_patch(plt.Rectangle((cc - size / 2, rr - size / 2), size, size, color=color, lw=0, **kw))


# ---------------------------------------------------------------- main
def main():
    stim_pts, reg_pts = points.load()
    X, Y = joanna_model()
    W = sio.loadmat(WINNER_FILE)["opt_points"].astype(int)          # stim row, col, 0, 0, cam row, cam col (1-based)
    w_stim = [tuple(p[:2]) for p in W]; w_cam = [tuple(p[4:6]) for p in W]
    is_anchor = np.array([s in ANCHORS for s in w_stim])
    anc_cam = [w_cam[w_stim.index(a)] for a in ANCHORS]
    pr, pc = joanna_predict(X, Y, w_stim, ANCHORS, anc_cam)
    d_w = np.hypot(pr - W[:, 4], pc - W[:, 5])
    M_w = metrics(d_w)
    print("thesis winner re-run: distances", np.round(d_w, 2), {k: round(v, 3) for k, v in M_w.items()})

    # thesis plate + its marks (camera, 0-based)
    rgb, grey, red_xy, blue_xy = MF.fig11b()
    reg = MF.register_plate(grey, MF.np.asarray(Image.open(MF.LEG / "images/100718/BV_resized.png").convert("L")).astype(float))
    to_plate = lambda r0, c0: ((np.asarray(c0) - reg["col0"]) * reg["s"], (np.asarray(r0) - reg["row0"]) * reg["s"])
    tr, tc = MF.plate_to_cam(reg, red_xy); br, bc = MF.plate_to_cam(reg, blue_xy)
    marks = pd.DataFrame(dict(row=np.r_[tr, br], col=np.r_[tc, bc], kind=["red dot"] * len(tr) + ["blue cross"] * len(br)))

    # ---- figure 1: on the thesis plate
    allp = pd.concat([reg_pts.assign(table="right"), stim_pts.assign(table="left")]).drop_duplicates(["vsd_x", "vsd_y"])
    allp["row"], allp["col"] = allp.vsd_y - 1, allp.vsd_x - 1                 # top-left reading, 0-based
    pairs = []
    for _, m in marks.iterrows():
        dd = np.hypot(allp.row - m.row, allp.col - m.col); j = int(np.argmin(dd))
        pairs.append(dict(mark=m.kind, mark_row=m.row, mark_col=m.col, point=allp.iloc[j].stimulus,
                          in_right_table=allp.iloc[j].table == "right", row=allp.iloc[j].row, col=allp.iloc[j].col, d=dd.iloc[j]))
    PR = pd.DataFrame(pairs)
    M_t = metrics(PR.d, n_anchor=0)
    PR.to_csv(MF.FIGS / "summary_thesis_pairs.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(17, 8.6))
    for a in ax:
        a.imshow(rgb); a.set_xlim(0, rgb.shape[1]); a.set_ylim(rgb.shape[0], 0); a.set_xticks([]); a.set_yticks([])
    a = ax[0]
    rp = reg_pts.assign(row=reg_pts.vsd_y - 1, col=reg_pts.vsd_x - 1)
    x, y = to_plate(rp.row, rp.col)
    a.scatter(x, y, s=170, marker="s", facecolors="none", edgecolors="red", lw=2.2, label="xlsx registration point (right table)")
    for i, (xx, yy) in enumerate(zip(x, y)):
        a.text(xx + 9, yy - 7, str(rp.point.iloc[i]), color="red", fontsize=12, weight="bold")
    ex = PR[~PR.in_right_table]
    x2, y2 = to_plate(ex.row, ex.col)
    a.scatter(x2, y2, s=170, marker="s", facecolors="none", edgecolors="orange", lw=2.2, label="left-table point (vertical bar upper)")
    for _, p in PR.iterrows():
        (x0, y0), (x1, y1) = to_plate(p.row, p.col), to_plate(p.mark_row, p.mark_col)
        a.plot([x0, x1], [y0, y1], color="k", lw=1)
        a.text(x1 - 10, y1 + 22, f"{p.d:.1f}", fontsize=10, color="k", weight="bold",
               bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    a.legend(loc="lower left", fontsize=9)
    tbl = "\n".join(f"{p.mark:10s} ↔ {p.point:28s} {p.d:4.1f} px" for _, p in PR.iterrows())
    a.set_title("thesis Fig. 11B with the xlsx points (VSD (x, y), (1,1) = top-left)\n"
                f"each thesis mark ↔ nearest xlsx point · mean distance (Joanna's metric) {M_t['joanna']:.2f} px · "
                f"RMSE {M_t['rmse']:.2f} px", fontsize=10, loc="left")
    a.text(4, rgb.shape[0] - 70, tbl, fontsize=8, family="monospace", va="top",
           bbox=dict(fc="white", ec="none", alpha=0.8))
    a.text(4, 14, "table point 1 (triangle top) has no thesis mark", fontsize=9, color="red",
           bbox=dict(fc="white", ec="none", alpha=0.8))

    a = ax[1]
    mx, my = to_plate(W[:, 4] - 1, W[:, 5] - 1); qx, qy = to_plate(pr - 1, pc - 1)
    a.scatter(mx[~is_anchor], my[~is_anchor], s=150, marker="s", facecolors="none", edgecolors="red", lw=2.2,
              label="measured VSD point (Joanna's file)")
    a.scatter(mx[is_anchor], my[is_anchor], s=170, marker="X", color="blue", label="anchor (measured = predicted)")
    a.scatter(qx[~is_anchor], qy[~is_anchor], s=150, facecolors="none", edgecolors="cyan", lw=2.2, label="model prediction")
    for i in np.where(~is_anchor)[0]:
        a.plot([mx[i], qx[i]], [my[i], qy[i]], color="k", lw=1.2)
        a.text(mx[i] + 10, my[i] - 8, f"{NAMES.get(w_stim[i], w_stim[i])}\n{d_w[i]:.2f} px", fontsize=9, weight="bold",
               color="k", bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
    a.legend(loc="lower left", fontsize=9)
    a.set_title(f"Joanna's thesis winner re-run (a {WINNER['a']}, α {WINNER['alpha']} [θ/α], k {WINNER['k']}; "
                f"anchors: vertical-bar upper, circle bottom)\nmean distance (her 'RMSD') {M_w['joanna']:.2f} px "
                f"(thesis: 2.56) · true RMSE {M_w['rmse']:.2f} px · calc_rmsd {M_w['calc_rmsd']:.2f}", fontsize=10, loc="left")
    fig.suptitle("10/07/18 registration points on thesis Fig. 11B (plate registered to Joanna's vessel image, "
                 f"{reg['s']:.2f} thesis px per camera px; distances in camera px)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(MF.FIGS / "summary_points_on_thesis.png", dpi=95); plt.close(fig)

    # ---- figure 2: on the stimuli and the VSD maps (Joanna's model, every stimulus point of the left table)
    maps = np.load(MF.MAPS); cat = pd.read_csv(MF.CATALOG)
    S = stim_pts.copy()
    S["stim_rc"] = list(zip(S.vf_y_px.astype(int), S.vf_x_px.astype(int)))
    S["cam_r"], S["cam_c"] = S.vsd_y, S.vsd_x                                   # 1-based measured
    S["pred_r"], S["pred_c"] = joanna_predict(X, Y, list(S.stim_rc), ANCHORS, anc_cam)
    S["d"] = np.hypot(S.pred_r - S.cam_r, S.pred_c - S.cam_c)
    S["anchor"] = S.stim_rc.isin(ANCHORS)
    S["our_r"], S["our_c"], our_mirror = our_predict(list(S.stim_rc), ANCHORS, anc_cam, S.pred_r.values, S.pred_c.values)
    S["d_our"] = np.hypot(S.our_r - S.cam_r, S.our_c - S.cam_c)
    S["in_thesis_rmsd"] = S.stim_rc.isin([s for s, an in zip(w_stim, is_anchor) if not an])
    S.drop(columns="stim_rc").to_csv(MF.FIGS / "summary_stimulus_points.csv", index=False)
    nz = S[~S.anchor].drop_duplicates(["vf_x_px", "vf_y_px"])
    M_all = dict(joanna=nz.d.mean(), rmse=np.sqrt((nz.d ** 2).mean()))
    M_our = dict(joanna=nz.d_our.mean(), rmse=np.sqrt((nz.d_our ** 2).mean()))
    # our model on the thesis RMSD points with Joanna's measured values (her .mat), for a like-for-like number
    o_r, o_c, _ = our_predict([w_stim[i] for i in np.where(~is_anchor)[0]], ANCHORS, anc_cam, mirror=our_mirror)
    d_our_th = np.hypot(o_r - W[~is_anchor, 4], o_c - W[~is_anchor, 5])
    print("our model, thesis RMSD points:", np.round(d_our_th, 2), "mean", d_our_th.mean().round(3),
          "| all non-anchor:", {k: round(v, 3) for k, v in M_our.items()}, "mirror", our_mirror)
    print("all left-table points (non-anchor, unique):", len(nz), {k: round(v, 3) for k, v in M_all.items()})

    conds = sorted(S.cond.unique())
    fig, axes = plt.subplots(len(conds), 2, figsize=(11, 4.4 * len(conds)))
    for (a0, a1), c in zip(axes, conds):
        P = S[S.cond == c]
        st = cat[(cat.session == SESSION) & (cat.condition == f"condAN{c}")].iloc[0]
        img = np.asarray(Image.open(MF.DATA.parent / st.image_path).convert("RGB"))
        a0.imshow(img[:56, :56], interpolation="nearest")
        mark(a0, P.vf_y_px.values, P.vf_x_px.values, color=MEAS)
        a0.set_title(f"cond {c}: {st.stimulus_text.split(': ', 1)[-1]}\nVF points (x,y px): "
                     + ", ".join(f"({x:g},{y:g})" for x, y in zip(P.vf_x_px, P.vf_y_px)), fontsize=9, loc="left")
        m = maps[f"{SESSION}|condAN{c}"]; lo, hi = np.nanpercentile(m, [2, 99.5])
        a1.imshow(gaussian_filter(np.nan_to_num(m, nan=np.nanmedian(m)), 1.0), cmap="jet", vmin=lo, vmax=hi)
        mark(a1, P.cam_r.values - 1, P.cam_c.values - 1, color=MEAS)
        lab = []
        for _, p in P.iterrows():
            a1.scatter(p.pred_c - 1, p.pred_r - 1, s=90, facecolors="none", edgecolors="yellow" if p.anchor else "cyan", lw=2)
            if not p.anchor:
                a1.scatter(p.our_c - 1, p.our_r - 1, s=80, marker="x", color="white", lw=2.2)
            a1.plot([p.cam_c - 1, p.pred_c - 1], [p.cam_r - 1, p.pred_r - 1], color="k", lw=1)
            tag = " (anchor)" if p.anchor else (" *" if p.in_thesis_rmsd else "")
            if p.stim_rc in w_stim:                         # Joanna's .mat has a different VSD point than the xlsx
                i = w_stim.index(p.stim_rc)
                if w_cam[i] != (p.cam_r, p.cam_c):
                    a1.scatter(w_cam[i][1] - 1, w_cam[i][0] - 1, s=60, marker="s", facecolors="none", edgecolors="magenta", lw=2)
                    tag += f" [her .mat: ({w_cam[i][1]},{w_cam[i][0]}) → {d_w[i]:.1f} px, magenta]"
            lab.append(f"{p.stimulus.split('(')[-1].rstrip(')') if '(' in p.stimulus else 'centre'}: Joanna {p.d:.1f} px"
                       + ("" if p.anchor else f", ours {p.d_our:.1f} px") + tag)
        a1.set_xlim(0, 99); a1.set_ylim(80, 10)
        a1.set_title(f"{SESSION} · green = measured, cyan ○ = Joanna, white × = ours\n" + "\n".join(lab), fontsize=9, loc="left")
    fig.suptitle("Stimulus points → cortex with Joanna's thesis winner (a 0.74, α 1, k 1, her retino_mat grid; anchors "
                 "vertical-bar upper + circle bottom)\n"
                 f"thesis RMSD points (*): mean distance {M_w['joanna']:.2f} px (thesis 2.56), RMSE {M_w['rmse']:.2f} px · "
                 f"all {len(nz)} non-anchor points: mean {M_all['joanna']:.2f} px, RMSE {M_all['rmse']:.2f} px\n"
                 f"ours (true 35 px/°, repo Schira, same a/α/k + anchors, no rounding):\nthesis RMSD points mean "
                 f"{d_our_th.mean():.2f} px · all {len(nz)} non-anchor points mean {M_our['joanna']:.2f} px, "
                 f"RMSE {M_our['rmse']:.2f} px",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(MF.FIGS / "summary_points_on_stimuli.png", dpi=90); plt.close(fig)
    print(PR[["mark", "point", "d"]].round(2).to_string(index=False), "\n", {k: round(v, 3) for k, v in M_t.items()})


if __name__ == "__main__":
    main()
