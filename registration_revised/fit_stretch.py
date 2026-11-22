"""Joanna's registration with one extra parameter: a horizontal stretch r of the visual field.

Model (all points of one day, one fixed anchor pair):
    VF (x, y) deg  ->  (r * x, y)                       stretch, i.e. the matrix [[r, 0], [0, 1]]
                   ->  Schira w (a, alpha [P = theta/alpha], k = 1; repo code, schira_geometry.schira_w)
                   ->  camera: exact similarity through the 2 anchors (shift, rotation, scale), flip fixed
    score = Joanna's 'RMSD' = mean distance over the non-anchor points (camera px); RMSE also reported.
Free parameters: a, alpha, r. A uniform scale of the VF is not needed: scaling (x, y) by s is exactly a -> a/s plus a
shift of w that the anchors absorb, so only the ratio r is identifiable. Joanna's retino_mat grid reads
x' = 0.007 + 0.415 x, y' = 0.014 + 0.77 y, i.e. r = 0.415 / 0.77 = 0.54 and a_true = a_joanna / 0.77.

VF points are stimulus-canvas pixels at the true 35 px/deg (x = col / 35, y = -row / 35); camera points 0-based.

Point sets (10/07/18):
  thesis   opt_points_25_09_23_23_01.mat (the thesis 2.56 px), anchors vertical-bar upper + circle bottom
  A        xlsx right table (= opt_points_26_09_23_11_33.mat; horizontal-bar right as in her .mat), anchors as in her
           winner for that file (26-09-2023-11-37: triangle top + vertical-bar bottom, a 0.77, alpha 1.1, 2.618 px)
  B        both xlsx tables (10 distinct points), same anchors as A

  20/11/18  opt_points_8_5_22.mat (8 letter points), her winner anchors (49,18), (15,18)
  joint    shared a, alpha, r over both days, each day with its own exact 2-anchor camera

Writes figs/stretch_profile.png, figs/stretch_fits.csv.   Run: scripts/py registration_revised/fit_stretch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "schira_dashboard"))
import points                                     # noqa: E402
import schira_geometry as G                       # noqa: E402

PTS = REPO.parent / "legacy_code/Retinotopic model Joanna/Gandalf/VSDI points/100718"
FIGS = HERE / "figs"
R_JOANNA = 0.415 / 0.77
MIRROR = False                                    # orientation that matches Joanna's flipped frame (see summary_figs)


def mat_points(name):
    """(stim row, col), (cam row, col), 1-based, from one of Joanna's opt_points files."""
    P = sio.loadmat(PTS / name)["opt_points"].astype(int)
    return [tuple(p[:2]) for p in P], [tuple(p[4:6]) for p in P]


def xlsx_points(tables=("right",)):
    s, r = points.load()
    T = pd.concat([d for t, d in (("right", r), ("left", s)) if t in tables]).drop_duplicates(["vf_x_px", "vf_y_px"])
    stim = [(int(y), int(x)) for x, y in zip(T.vf_x_px, T.vf_y_px)]
    cam = [(int(y), int(x)) for x, y in zip(T.vsd_x, T.vsd_y)]
    fix = {(26, 38): (48, 51)}                    # horizontal-bar right: xlsx has x/y swapped vs her .mat and the thesis
    cam = [fix.get(s_, c_) for s_, c_ in zip(stim, cam)]
    return stim, cam


def predict(stim, cam, anchors, a, alpha, r, k=1.0):
    """Predicted camera points (complex col + i row, 0-based) for every point."""
    G.set_pixel_grid("35ppd")
    x = np.array([c for _, c in stim]) / G.PPD * r
    y = -np.array([rw for rw, _ in stim]) / G.PPD
    w = G.schira_w(x, y, a, alpha, k, "thesis")
    ia = [stim.index(p) for p in anchors]
    cam_c = np.array([(c - 1) + 1j * (rw - 1) for rw, c in cam])
    wv = np.conj(w) if MIRROR else w
    if not np.all(np.isfinite(wv[ia])) or wv[ia[0]] == wv[ia[1]]:
        return np.full(len(w), np.nan + 0j), cam_c, np.array(ia)
    c1 = (cam_c[ia[1]] - cam_c[ia[0]]) / (wv[ia[1]] - wv[ia[0]])             # exact similarity through 2 points
    return c1 * (wv - wv[ia[0]]) + cam_c[ia[0]], cam_c, np.array(ia)


def score(stim, cam, anchors, a, alpha, r, mask=None):
    u, c, ia = predict(stim, cam, anchors, a, alpha, r)
    d = np.abs(u - c)
    keep = np.ones(len(d), bool); keep[ia] = False
    if mask is not None:
        keep &= mask
    return d, keep


def fit(stim, cam, anchors, fit_r=True, mask=None, r_fixed=1.0, starts=None):
    def cost(p):
        a, al = np.exp(p[0]), np.exp(p[1]); r = np.exp(p[2]) if fit_r else r_fixed
        if not (0.05 < a < 5 and 0.3 < al < 4 and 0.15 < r < 3):
            return 1e3
        d, keep = score(stim, cam, anchors, a, al, r, mask)
        v = d[keep].mean()
        return v if np.isfinite(v) else 1e3
    starts = starts or [(a, al, r) for a in (0.3, 0.7, 1.2) for al in (0.7, 1.0, 1.5) for r in ((0.4, 0.7, 1.0) if fit_r else (r_fixed,))]
    best = min((minimize(cost, np.log(s_), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-5, maxiter=3000))
                for s_ in starts), key=lambda o: o.fun)
    a, al = np.exp(best.x[:2]); r = np.exp(best.x[2]) if fit_r else r_fixed
    d, keep = score(stim, cam, anchors, a, al, r, mask)
    return dict(a=a, alpha=al, r=r, mean=d[keep].mean(), rmse=np.sqrt((d[keep] ** 2).mean()), d=d, keep=keep)


def loo(stim, cam, anchors, fit_r=True, r_fixed=1.0):
    """Leave one scored point out: refit on the rest, predict it."""
    _, keep = score(stim, cam, anchors, 1, 1, 1)
    out = []
    for i in np.where(keep)[0]:
        m = keep.copy(); m[i] = False
        f = fit(stim, cam, anchors, fit_r, mask=m, r_fixed=r_fixed)
        d, _ = score(stim, cam, anchors, f["a"], f["alpha"], f["r"])
        out.append(d[i])
    return np.array(out)


PTS_201118 = REPO.parent / "legacy_code/Retinotopic model Joanna/Gandalf/VSDI points/201118"


def set_201118():
    """Joanna's 20/11/18 letter points (opt_points_8_5_22.mat, 8 points) and her winner anchors (49,18), (15,18)
    (Opt_results/201118/12-06-2022-16: a 0.72, alpha 1.5, 1.52 px)."""
    P = sio.loadmat(PTS_201118 / "opt_points_8_5_22.mat")["opt_points"].astype(int)
    return [tuple(p[:2]) for p in P], [tuple(p[4:6]) for p in P], [(49, 18), (15, 18)]


def fit_joint(sets, starts=None):
    """Shared a, alpha, r over several days; each day keeps its own exact 2-anchor camera. Score = mean distance over
    all non-anchor points of all days."""
    def dists(a, al, r):
        out = []
        for stim, cam, anc in sets:
            d, keep = score(stim, cam, anc, a, al, r); out.append(d[keep])
        return np.concatenate(out)
    def cost(p):
        a, al, r = np.exp(p)
        if not (0.05 < a < 5 and 0.3 < al < 4 and 0.15 < r < 3):
            return 1e3
        v = dists(a, al, r).mean()
        return v if np.isfinite(v) else 1e3
    starts = starts or [(a, al, r) for a in (0.3, 0.7, 1.2) for al in (0.7, 1.0, 1.5) for r in (0.4, 0.7, 1.0)]
    best = min((minimize(cost, np.log(s_), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-5, maxiter=3000))
                for s_ in starts), key=lambda o: o.fun)
    a, al, r = np.exp(best.x); d = dists(a, al, r)
    return dict(a=a, alpha=al, r=r, mean=d.mean(), rmse=np.sqrt((d ** 2).mean()), per_day=[
        float(np.mean(score(s, c, an, a, al, r)[0][score(s, c, an, a, al, r)[1]])) for s, c, an in sets])


def fit_shared_r(sets, starts_r=(0.4, 0.55, 0.7, 1.0), starts_aal=((0.3, 0.7), (0.7, 1.0), (1.2, 1.5), (3.0, 1.5))):
    """One r shared by all days; a and alpha per day; each day its own exact 2-anchor camera.
    Score = mean distance over all non-anchor points of all days. Also returns every local optimum found, so the
    spread of near-equal solutions is visible."""
    n = len(sets)
    def unpack(p):
        e = np.exp(p); return e[:n], e[n:2 * n], e[-1]
    def dists(a, al, r):
        return [score(s, c, an, a[i], al[i], r)[0][score(s, c, an, a[i], al[i], r)[1]] for i, (s, c, an) in enumerate(sets)]
    def cost(p):
        a, al, r = unpack(p)
        if not (np.all((0.05 < a) & (a < 5)) and np.all((0.3 < al) & (al < 4)) and 0.15 < r < 3):
            return 1e3
        v = np.concatenate(dists(a, al, r)).mean()
        return v if np.isfinite(v) else 1e3
    runs = []
    for r0 in starts_r:
        for aal in starts_aal:
            p0 = np.log(np.r_[[aal[0]] * n, [aal[1]] * n, r0])
            o = minimize(cost, p0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-5, maxiter=8000, adaptive=True))
            runs.append(o)
    best = min(runs, key=lambda o: o.fun)
    a, al, r = unpack(best.x); d = dists(a, al, r)
    sols = sorted({tuple(np.round(np.exp(o.x), 2)) + (round(o.fun, 3),) for o in runs if o.fun < 1e3}, key=lambda t: t[-1])
    return dict(a=a, alpha=al, r=r, mean=np.concatenate(d).mean(), per_day=[float(x.mean()) for x in d], solutions=sols)


def main():
    FIGS.mkdir(exist_ok=True)
    sets = {
        "thesis set (opt_points_25_09_23_23_01)": (*mat_points("opt_points_25_09_23_23_01.mat"), [(9, 21), (37, 21)]),
        "A: xlsx right table (6 pts)": (*xlsx_points(("right",)), [(14, 14), (43, 21)]),
        "B: both xlsx tables (10 pts)": (*xlsx_points(("right", "left")), [(14, 14), (43, 21)]),
        "20/11/18: opt_points_8_5_22 (8 pts)": set_201118(),
    }
    rows, prof = [], {}
    for name, (stim, cam, anchors) in sets.items():
        n_sc = len(stim) - 2
        # sanity: Joanna's own grid as a stretch (r 0.54, a = a_joanna / 0.77)
        jw = (0.74, 1.0) if "thesis" in name else (0.72, 1.5) if "20/11/18" in name else (0.77, 1.1)
        for lab, (a0, al0) in {"Joanna winner, r 0.54": jw}.items():
            d, keep = score(stim, cam, anchors, a0 / 0.77, al0, R_JOANNA)
            rows.append(dict(set=name, model=lab + f" (a {a0}/0.77, α {al0}, fixed)", n_scored=n_sc, a=a0 / 0.77, alpha=al0,
                             r=R_JOANNA, mean=d[keep].mean(), rmse=np.sqrt((d[keep] ** 2).mean()), loo_mean=np.nan))
        f1 = fit(stim, cam, anchors, fit_r=False, r_fixed=1.0)
        f2 = fit(stim, cam, anchors, fit_r=True)
        l1 = loo(stim, cam, anchors, fit_r=False) if n_sc >= 3 else np.array([np.nan])
        l2 = loo(stim, cam, anchors, fit_r=True) if n_sc >= 4 else np.array([np.nan])
        for lab, f, l in (("fit a, α (r = 1, true geometry)", f1, l1), ("fit a, α, r", f2, l2)):
            rows.append(dict(set=name, model=lab, n_scored=n_sc, a=f["a"], alpha=f["alpha"], r=f["r"], mean=f["mean"],
                             rmse=f["rmse"], loo_mean=np.nanmean(l)))
        # profile: best mean distance for each fixed r
        rr = np.linspace(0.25, 1.4, 47)
        prof[name] = (rr, [fit(stim, cam, anchors, fit_r=False, r_fixed=r_,
                                starts=[(a, al, r_) for a in (0.4, 0.9) for al in (0.8, 1.2)])["mean"] for r_ in rr], f2)
        print(f"{name}: r = {f2['r']:.3f}, a {f2['a']:.3f}, α {f2['alpha']:.3f}, mean {f2['mean']:.2f} px", flush=True)
    # joint fit over both days: shared a, alpha, r; per-day anchors
    for lab, keys in (("joint: B + 20/11/18", ["B: both xlsx tables (10 pts)", "20/11/18: opt_points_8_5_22 (8 pts)"]),
                      ("joint: thesis set + 20/11/18", ["thesis set (opt_points_25_09_23_23_01)", "20/11/18: opt_points_8_5_22 (8 pts)"])):
        fj = fit_joint([sets[k] for k in keys])
        rows.append(dict(set=lab, model="fit a, α, r (shared)", n_scored=sum(len(sets[k][0]) - 2 for k in keys),
                         a=fj["a"], alpha=fj["alpha"], r=fj["r"], mean=fj["mean"], rmse=fj["rmse"], loo_mean=np.nan,
                         per_day=" / ".join(f"{v:.2f}" for v in fj["per_day"])))
        print(f"{lab}: r = {fj['r']:.3f}, a {fj['a']:.3f}, α {fj['alpha']:.3f}, mean {fj['mean']:.2f} px, per day {fj['per_day']}")
    R = pd.DataFrame(rows)
    R.to_csv(FIGS / "stretch_fits.csv", index=False)
    with pd.option_context("display.width", 200):
        print(R.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, len(prof), figsize=(5.2 * len(prof), 4.6), sharey=True)
    for a_, (name, (rr, mm, f2)) in zip(ax, prof.items()):
        a_.plot(rr, mm, "k-")
        a_.axvline(R_JOANNA, color="tab:orange", ls="--", label=f"Joanna's grid r = {R_JOANNA:.2f}")
        a_.axvline(1, color="tab:blue", ls=":", label="true geometry r = 1")
        a_.plot(f2["r"], f2["mean"], "ro", label=f"best r = {f2['r']:.2f} ({f2['mean']:.2f} px)")
        a_.set_title(name, fontsize=10, loc="left"); a_.set_xlabel("stretch r (x-scale / y-scale)")
        a_.legend(fontsize=8)
    ax[0].set_ylabel("mean distance, non-anchor points (px)\n(a, α re-fitted at each r)")
    fig.suptitle("How well each horizontal stretch r fits (Joanna's method: 2 fixed anchors, score = mean distance)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIGS / "stretch_profile.png", dpi=95); plt.close(fig)


if __name__ == "__main__":
    main()
