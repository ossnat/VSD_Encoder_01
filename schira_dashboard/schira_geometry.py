"""Geometry for the Schira dashboard: visual field -> Schira w -> camera, with a fitted registration (inner loop)
and an optional search over Schira's a, alpha, k (outer loop).

Conventions
  * Schira is the repo implementation (src/retinotopy/schira.py, double-sech, S1 0.76, S2 0.1821):
        w = k * log(E * exp(i P fa) + a)
    'repo'   : P = alpha * theta   (Ayzenshtat 2012, the repo)
    'thesis' : P = theta / alpha   (legacy MATLAB / thesis values, e.g. alpha 1.5 == repo 0.667)
  * Visual field: x right, y up, degrees from fixation (lower field y < 0).
  * Camera points are complex u = col + i*row (0-based pixels of the 100 x 100 VSD map); 1 px = 0.17 mm.
  * Horizontal stretch r (registration_revised/fit_stretch.py): the visual field is read as (r*x, y) before Schira,
    i.e. the matrix [[r, 0], [0, 1]]. r = 1 is the true geometry; Joanna's retino_mat grid amounts to r = 0.415/0.77
    = 0.54 (plus a uniform 0.77 that only rescales a). Fitted on 10/07/18 + 20/11/18: r ~ 0.54.
  * Registration (inner loop), u = c1 * w' + c0 with w' = w or conj(w) (mirror):
      'similarity' : c1 free (rotation + scale) -> k has NO effect (absorbed by the scale)
      'rigid'      : |c1| = 1 / 0.17 px per mm, only rotation free -> w is read in mm, so k is the mm scale
      'anchors'    : Joanna's method: the similarity solved EXACTLY through 2 chosen anchor landmarks
                     (c1 = (u_B - u_A) / (w_B - w_A), c0 = u_A - c1 w_A); the other landmarks only score the fit,
                     with her 'RMSD' = mean distance over the non-anchor landmarks
"""
from __future__ import annotations

import itertools, sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.retinotopy.schira import SchiraParams, forward_schira, cartesian_to_polar  # noqa: E402

MM_PER_PX = 0.17
PPD = 35.0


# ---------------------------------------------------------------- stimulus px -> degrees
# '35ppd'  : the true geometry, 35 px per degree on both axes (default).
# 'retino' : Joanna's pipeline (thesis): trans_cortex reads stimulus pixel (row r, col c) of the 301 x 512 canvas at
#            x = linspace(0.00714, 6.06429, 512)[c], y = linspace(0.01429, 6.61429, 301)[r]  (retino_mat_20_03_22.mat),
#            i.e. ~84.4 px/deg horizontally and ~45.5 px/deg vertically. Positions given in degrees (landmarks) go
#            deg -> px at 35 px/deg -> that grid, as in her trans_and_registration.m (conv_pix_deg, then xcoord/ycoord).
RETINO_X, RETINO_Y = (0.0071428571428571175, 6.064285714285714, 512), (0.014285714285714235, 6.614285714285714, 301)
PIXEL_GRID = "35ppd"          # set by the dashboard before any mapping
STRETCH_R = 1.0               # horizontal stretch of the visual field (set by the dashboard)

# Joanna's winning anchor pairs (thesis): visual-field position (deg, 35 px/deg) of the two manual landmarks per day
DAY_ANCHORS = {"100718": ((21 / PPD, -9 / PPD), (21 / PPD, -37 / PPD)),     # vertical-bar upper, circle bottom
               "201118": ((18 / PPD, -49 / PPD), (18 / PPD, -15 / PPD))}    # her 20/11/18 winner (49,18), (15,18)
# our joint fit of both days (fit_stretch.py, thesis set + 20/11/18): shared a, alpha (thesis convention), r
JOINT_FIT = dict(a=1.43, alpha_thesis=1.50, r=0.539)


def set_stretch(r: float) -> None:
    global STRETCH_R
    if not r > 0:
        raise ValueError(r)
    STRETCH_R = float(r)


def day_anchors(session: str, x_deg, y_deg, source) -> tuple[int, int] | None:
    """Indices of the day's default anchor pair among manual landmarks (x_deg, y_deg, source arrays), or None."""
    pair = DAY_ANCHORS.get(session[:6])
    if pair is None:
        return None
    x, y, src = np.asarray(x_deg, float), np.asarray(y_deg, float), np.asarray(source)
    out = []
    for xa, ya in pair:
        hit = np.where((src == "manual") & (np.abs(x - xa) < 1e-3) & (np.abs(y - ya) < 1e-3))[0]
        if not hit.size:
            return None
        out.append(int(hit[0]))
    return tuple(out)


def set_pixel_grid(mode: str) -> None:
    global PIXEL_GRID
    if mode not in ("35ppd", "retino"):
        raise ValueError(mode)
    PIXEL_GRID = mode


def model_deg(x_deg, y_deg, r: float | None = None):
    """Visual-field degrees as the model reads them: the pixel grid, then the horizontal stretch r (default STRETCH_R)."""
    r = STRETCH_R if r is None else r
    x, y = np.asarray(x_deg, float), np.asarray(y_deg, float)
    if PIXEL_GRID != "35ppd":
        (x0, x1, nx), (y0, y1, ny) = RETINO_X, RETINO_Y
        x, y = x0 + x * PPD * (x1 - x0) / (nx - 1), -(y0 + (-y) * PPD * (y1 - y0) / (ny - 1))
    return r * x, y


# ---------------------------------------------------------------- Schira
def schira_w(x_deg, y_deg, a: float, alpha: float, k: float, convention: str = "repo",
             r: float | None = None) -> np.ndarray:
    alpha_repo = alpha if convention == "repo" else 1.0 / alpha
    E, th = cartesian_to_polar(*model_deg(x_deg, y_deg, r))
    return forward_schira(E, th, SchiraParams(a=a, alpha=alpha_repo, k=k))


# ---------------------------------------------------------------- registration (inner loop)
@dataclass
class Registration:
    c1: complex
    c0: complex
    mirror: bool
    mode: str

    def apply(self, w: np.ndarray) -> np.ndarray:
        w = np.conj(w) if self.mirror else w
        return self.c1 * w + self.c0

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(np.angle(self.c1)))

    @property
    def px_per_w(self) -> float:
        return float(abs(self.c1))


def fit_registration(w: np.ndarray, cam: np.ndarray, mode: str, mirror: bool) -> Registration:
    """Least-squares registration of w onto camera points (complex)."""
    wv = np.conj(w) if mirror else w
    if mode == "similarity":
        M = np.column_stack([wv, np.ones_like(wv)])
        c1, c0 = np.linalg.lstsq(M, cam, rcond=None)[0]
    elif mode == "rigid":
        wc, uc = wv - wv.mean(), cam - cam.mean()
        phi = np.angle(np.sum(np.conj(wc) * uc)) if len(w) > 1 else 0.0
        c1 = np.exp(1j * phi) / MM_PER_PX
        c0 = cam.mean() - c1 * wv.mean()
    else:
        raise ValueError(mode)
    return Registration(complex(c1), complex(c0), mirror, mode)


@dataclass
class FitResult:
    reg: Registration | None
    order: np.ndarray                 # landmark index used for each row (bar-end pairings resolved)
    pred: np.ndarray                  # predicted camera points (complex), per landmark row
    resid_px: np.ndarray
    rmse_px: float
    loo_px: np.ndarray = field(default_factory=lambda: np.array([]))
    n_used: int = 0
    message: str = ""
    anchors: tuple | None = None      # 'anchors' mode: landmark rows used as anchors (exact)
    mean_px: float = np.nan           # 'anchors' mode: Joanna's 'RMSD' = mean distance over the non-anchor landmarks

    @property
    def rmse_mm(self) -> float:
        return self.rmse_px * MM_PER_PX

    @property
    def loo_rmse_px(self) -> float:
        v = self.loo_px[np.isfinite(self.loo_px)]
        return float(np.sqrt(np.mean(v ** 2))) if v.size else np.nan


def _pairings(pair_ids: list[str]) -> list[np.ndarray]:
    """All ways to swap the two ends of each unordered bar pair (index permutations)."""
    idx = np.arange(len(pair_ids))
    groups = {}
    for i, p in enumerate(pair_ids):
        if p:
            groups.setdefault(p, []).append(i)
    groups = [g for g in groups.values() if len(g) == 2][:10]
    out = []
    for flips in itertools.product((False, True), repeat=len(groups)):
        o = idx.copy()
        for f, (i, j) in zip(flips, groups):
            if f:
                o[i], o[j] = j, i
        out.append(o)
    return out


def anchor_registration(w: np.ndarray, cam: np.ndarray, anchors, mirror: bool) -> Registration | None:
    """Exact similarity through the two anchor rows (Joanna's 2-point fitgeotrans)."""
    i, j = anchors
    wv = np.conj(w) if mirror else w
    if not (np.isfinite(wv[i]) and np.isfinite(wv[j])) or wv[i] == wv[j]:
        return None
    c1 = (cam[j] - cam[i]) / (wv[j] - wv[i])
    return Registration(complex(c1), complex(cam[i] - c1 * wv[i]), mirror, "anchors")


def _fit_anchors(x, cam, w_all, pair_ids, anchors, mirror) -> FitResult:
    n = len(x)
    if anchors is None or len(anchors) != 2 or anchors[0] == anchors[1]:
        return FitResult(None, np.arange(n), np.full(n, np.nan + 0j), np.full(n, np.nan), np.nan,
                         message="anchor mode: choose two different anchor landmarks")
    scored = np.ones(n, bool); scored[list(anchors)] = False
    best = None
    for order in _pairings(list(pair_ids)):
        w = w_all[order]
        for mr in ((False, True) if mirror == "auto" else (bool(mirror),)):
            reg = anchor_registration(w, cam, anchors, mr)
            if reg is None:
                continue
            pred = reg.apply(w); r = np.abs(pred - cam)
            ok = scored & np.isfinite(r)
            m = float(r[ok].mean()) if ok.any() else np.nan
            if best is None or (np.isfinite(m) and (not np.isfinite(best.mean_px) or m < best.mean_px)):
                best = FitResult(reg, order, pred, r, float(np.sqrt(np.mean(r[ok] ** 2))) if ok.any() else np.nan,
                                 loo_px=np.where(scored, r, np.nan), n_used=int(np.isfinite(r).sum()),
                                 anchors=tuple(anchors), mean_px=m)
    if best is None:
        return FitResult(None, np.arange(n), np.full(n, np.nan + 0j), np.full(n, np.nan), np.nan,
                         message="anchor landmarks outside Schira's valid range")
    if scored.sum() == 0:
        best.message = "only the two anchors: nothing left to score the fit"
    return best


def fit_session(x_deg, y_deg, cam, pair_ids, a, alpha, k, convention="repo", mode="similarity",
                mirror="auto", loo=True, r=None, anchors=None) -> FitResult:
    """Inner loop: best registration (and bar-end pairing, and mirror if 'auto') for fixed Schira parameters.
    Visual-field coordinates are permuted within bar pairs; camera points stay put. r = horizontal stretch
    (default STRETCH_R). mode 'anchors' needs anchors = (row A, row B): exact similarity through them."""
    x = np.asarray(x_deg, float); y = np.asarray(y_deg, float); cam = np.asarray(cam, complex)
    w_all = schira_w(x, y, a, alpha, k, convention, r)
    if mode == "anchors":
        return _fit_anchors(x, cam, w_all, pair_ids, anchors, mirror)
    ok = np.isfinite(w_all)
    need = 2 if mode == "similarity" else 1
    if ok.sum() < need:
        return FitResult(None, np.arange(len(x)), np.full(len(x), np.nan + 0j), np.full(len(x), np.nan), np.nan,
                         message="not enough landmarks inside Schira's valid range")
    mirrors = (False, True) if mirror == "auto" else (bool(mirror),)
    best = None
    for order in _pairings(list(pair_ids)):
        w = w_all[order]
        okk = np.isfinite(w)
        for mr in mirrors:
            reg = fit_registration(w[okk], cam[okk], mode, mr)
            pred = np.full(len(x), np.nan + 0j); pred[okk] = reg.apply(w[okk])
            r = np.abs(pred - cam)
            rm = float(np.sqrt(np.nanmean(r ** 2)))
            if best is None or rm < best.rmse_px:
                best = FitResult(reg, order, pred, r, rm, n_used=int(okk.sum()))
    if loo and best.n_used > need:
        w = w_all[best.order]; okk = np.isfinite(w); lo = np.full(len(x), np.nan)
        for i in np.where(okk)[0]:
            m = okk.copy(); m[i] = False
            if m.sum() < need:
                continue
            rg = fit_registration(w[m], cam[m], mode, best.reg.mirror)
            lo[i] = abs(rg.apply(w[i:i + 1])[0] - cam[i])
        best.loo_px = lo
    dof = {"similarity": 4, "rigid": 3}[mode]
    if best.n_used <= dof:
        best.message = f"only {best.n_used} landmarks for {dof} registration parameters: the RMSE is not informative"
    return best


# ---------------------------------------------------------------- fit to target lines (thesis Fig. 12)
def lines_cost(pts: np.ndarray, target: np.ndarray, tree_t, centre: complex, radius: float, cap: float = 6.0) -> float:
    """Symmetric capped chamfer distance (camera px) between mapped model points and target line pixels, model points
    restricted to the circle (centre, radius). Line identity is not enforced."""
    from scipy.spatial import cKDTree
    p = pts[np.isfinite(pts)]; p = p[np.abs(p - centre) < radius]
    if len(p) < 30:
        return 2 * cap
    d1 = np.minimum(cKDTree(np.c_[p.real, p.imag]).query(np.c_[target.real, target.imag])[0], cap).mean()
    d2 = np.minimum(tree_t.query(np.c_[p.real, p.imag])[0], cap).mean()
    return float(0.5 * (d1 + d2))


def fit_camera_to_lines(ix, iy, target, centre, radius, a, alpha, k, convention, reg0: Registration,
                        cap: float = 6.0) -> tuple[Registration, float]:
    """For fixed Schira parameters: the similarity camera (shift, rotation, scale; mirror as reg0) that brings the
    mapped ink onto the target lines. Starts from reg0 (e.g. the landmark registration). Returns (camera, cost px)."""
    from scipy.spatial import cKDTree
    tree_t = cKDTree(np.c_[target.real, target.imag])
    w = schira_w(ix, iy, a, alpha, k, convention)
    w = np.conj(w) if reg0.mirror else w
    ok = np.isfinite(w); w = w[ok]
    cost = lambda p: lines_cost((p[0] + 1j * p[1]) * w + (p[2] + 1j * p[3]), target, tree_t, centre, radius, cap)
    p0 = [reg0.c1.real, reg0.c1.imag, reg0.c0.real, reg0.c0.imag]
    best = minimize(cost, p0, method="Nelder-Mead", options=dict(maxiter=800, xatol=1e-4, fatol=1e-4))
    for _ in range(2):   # restart from the optimum (Nelder-Mead on a capped chamfer can stall)
        r = minimize(cost, best.x, method="Nelder-Mead", options=dict(maxiter=800, xatol=1e-4, fatol=1e-4))
        if r.fun < best.fun - 1e-4:
            best = r
        else:
            break
    p = best.x
    return Registration(p[0] + 1j * p[1], p[2] + 1j * p[3], reg0.mirror, "thesis"), float(best.fun)


def fit_schira_to_lines(ix, iy, target, centre, radius, landmarks, starts, k=1.0, *, camera="session",
                        mode="similarity", mirror="auto", fit_k=False, cap=6.0, maxiter=500, anchors=None) -> dict:
    """Find a, alpha (repo convention) [and k] that bring the mapped stimulus ink (ix, iy deg) onto target line pixels.
    camera 'session': registration = the landmark fit of the session (as the dashboard draws it); 'free': a similarity
    camera fitted to the target lines too (starting from the landmark fit). landmarks = (x, y, cam, pair_ids)."""
    from scipy.spatial import cKDTree
    tree_t = cKDTree(np.c_[target.real, target.imag])
    lx, ly, lcam, lpair = landmarks

    def session_reg(a, al, kk):
        r = fit_session(lx, ly, lcam, lpair, a, al, kk, "repo", mode, mirror, loo=False, anchors=anchors)
        return r.reg

    best = None
    for a0, al0 in starts:
        reg0 = session_reg(a0, al0, k)
        if reg0 is None:
            continue
        if camera == "session":
            def unpack(p):
                return np.exp(p[0]), np.exp(p[1]), (np.exp(p[2]) if fit_k else k), None
            p0 = [np.log(a0), np.log(al0)] + ([np.log(k)] if fit_k else [])
        else:
            def unpack(p, mr=reg0.mirror):
                return np.exp(p[0]), np.exp(p[1]), k, Registration(p[2] + 1j * p[3], p[4] + 1j * p[5], mr, "similarity")
            p0 = [np.log(a0), np.log(al0), reg0.c1.real, reg0.c1.imag, reg0.c0.real, reg0.c0.imag]

        def cost(p):
            a, al, kk, reg = unpack(p)
            if not (0.01 < a < 50 and 0.05 < al < 20):
                return 3 * cap
            reg = reg or session_reg(a, al, kk)
            if reg is None:
                return 3 * cap
            return lines_cost(reg.apply(schira_w(ix, iy, a, al, kk, "repo")), target, tree_t, centre, radius, cap)
        r = minimize(cost, p0, method="Nelder-Mead", options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-4))
        if best is None or r.fun < best[0]:
            best = (r.fun, unpack(r.x))
    if best is None:
        return dict(ok=False)
    c, (a, al, kk, reg) = best
    return dict(ok=True, cost=c, a=float(a), alpha_repo=float(al), k=float(kk), reg=reg or session_reg(a, al, kk))


# ---------------------------------------------------------------- outer loop
def _session_cost(res: FitResult, n: int) -> float:
    """Score of one session's inner fit: mean non-anchor distance ('anchors') or RMSE; every landmark must stay inside
    Schira's valid range, otherwise dropping points would 'improve' the score."""
    v = res.mean_px if res.reg is not None and res.reg.mode == "anchors" else res.rmse_px
    return v if (res.reg is not None and np.isfinite(v) and res.n_used == n) else np.nan


def optimize_schira(x_deg, y_deg, cam, pair_ids, convention="repo", mode="similarity", mirror="auto",
                    start=(0.7, 1.0, 1.0), fit_k=None, bounds=((0.02, 20), (0.05, 20), (0.1, 100)),
                    r=None, fit_r=False, anchors=None, r_bounds=(0.15, 3.0)):
    """Nested optimisation: outer loop over (a, alpha[, k][, r]) on a log scale, inner loop = fit_session.
    k is only searched in 'rigid' mode (in 'similarity'/'anchors' mode it has no effect). Multi-start Nelder-Mead.
    Score: RMSE ('similarity', 'rigid') or Joanna's mean non-anchor distance ('anchors')."""
    fit_k = (mode == "rigid") if fit_k is None else fit_k
    r0 = STRETCH_R if r is None else r
    lb = list(np.log([b[0] for b in bounds[:2 + fit_k]])) + ([np.log(r_bounds[0])] if fit_r else [])
    ub = list(np.log([b[1] for b in bounds[:2 + fit_k]])) + ([np.log(r_bounds[1])] if fit_r else [])
    lb, ub = np.array(lb), np.array(ub)

    def unpack(p):
        p = np.clip(p, lb, ub)
        a, al = np.exp(p[:2]); k = np.exp(p[2]) if fit_k else start[2]
        rr = np.exp(p[-1]) if fit_r else r0
        return a, al, k, rr

    def cost(p):
        a, al, k, rr = unpack(p)
        v = _session_cost(fit_session(x_deg, y_deg, cam, pair_ids, a, al, k, convention, mode, mirror, loo=False,
                                      r=rr, anchors=anchors), len(x_deg))
        return v if np.isfinite(v) else 1e6

    starts = [[np.log(a0), np.log(al0)] + ([np.log(k0)] if fit_k else []) + ([np.log(rs)] if fit_r else [])
              for a0 in (0.1, 0.5, 2.0) for al0 in (0.5, 1.0, 2.0) for k0 in ((2.0, 8.0) if fit_k else (1,))
              for rs in ((0.4, 0.7, 1.0) if fit_r else (1,))]
    starts.append(list(np.log(start[:3 if fit_k else 2])) + ([np.log(r0)] if fit_r else []))
    best = min((minimize(cost, s_, method="Nelder-Mead", options=dict(maxiter=800, xatol=1e-3, fatol=1e-4)) for s_ in starts),
               key=lambda o: o.fun)
    a, al, k, rr = unpack(best.x)
    return dict(a=float(a), alpha=float(al), k=float(k), r=float(rr), cost_px=float(best.fun), rmse_px=float(best.fun),
                n=len(x_deg))


def optimize_joint(days, convention="thesis", mirror="auto", fit_r=True, r=1.0, mode="anchors",
                   bounds=((0.05, 5.0), (0.3, 4.0), (0.15, 3.0))):
    """Our suggested fit (registration_revised/fit_stretch.py): ONE a, alpha and r shared by several days, each day
    with its own camera ('anchors': exact through that day's 2 anchors; 'similarity': least squares).
    days = list of dicts(x, y, cam, pair_ids, anchors). Score = mean distance over all scored landmarks of all days
    ('anchors') or the pooled RMSE ('similarity'). Bounds as in fit_stretch.py (alpha in the given convention)."""
    lb = np.log([b[0] for b in bounds[:2 + fit_r]]); ub = np.log([b[1] for b in bounds[:2 + fit_r]])

    def run(a, al, rr):
        return [fit_session(d["x"], d["y"], d["cam"], d["pair_ids"], a, al, 1.0, convention, mode, mirror, loo=False,
                            r=rr, anchors=d.get("anchors")) for d in days]

    def pooled(res):
        if mode == "anchors":
            v = [x.loo_px[np.isfinite(x.loo_px)] for x in res if x.reg is not None]
            return float(np.concatenate(v).mean()) if len(v) == len(res) else np.nan
        v = [x.resid_px[np.isfinite(x.resid_px)] for x in res if x.reg is not None]
        return float(np.sqrt(np.mean(np.concatenate(v) ** 2))) if len(v) == len(res) else np.nan

    def cost(p):
        if np.any(p < lb) or np.any(p > ub):
            return 1e3
        a, al = np.exp(p[:2]); rr = np.exp(p[2]) if fit_r else r
        res = run(a, al, rr)
        if any(x.reg is None or x.n_used < len(d["x"]) for x, d in zip(res, days)):
            return 1e3
        v = pooled(res)
        return v if np.isfinite(v) else 1e3

    starts = [[np.log(a0), np.log(al0)] + ([np.log(r0)] if fit_r else [])
              for a0 in (0.3, 0.7, 1.2) for al0 in (0.7, 1.0, 1.5) for r0 in ((0.4, 0.7, 1.0) if fit_r else (1,))]
    best = min((minimize(cost, s_, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-5, maxiter=3000)) for s_ in starts),
               key=lambda o: o.fun)
    a, al = np.exp(best.x[:2]); rr = float(np.exp(best.x[2])) if fit_r else r
    res = run(a, al, rr)
    per_day = [(x.mean_px if mode == "anchors" else x.rmse_px) for x in res]
    return dict(a=float(a), alpha=float(al), r=rr, cost_px=float(best.fun), per_day=per_day,
                n_scored=[int(np.isfinite(x.loo_px if mode == "anchors" else x.resid_px).sum()) for x in res])


# ---------------------------------------------------------------- stimuli
def ink_to_deg(img: np.ndarray, bg: float | None = None, thresh: float = 20.0, step: float = 0.5):
    """Visual-field coordinates (deg) of the stimulus ink, sampled every `step` canvas pixels.
    Canvas convention (repo renderer): 35 px/deg, fixation at the top-left corner."""
    g = img.mean(-1) if img.ndim == 3 else img.astype(float)
    bg = np.median(g) if bg is None else bg
    ink = np.abs(g - bg) > thresh
    rr, cc = np.nonzero(ink)
    if step < 1 and rr.size:
        off = np.arange(0, 1, step)
        dr, dc = np.meshgrid(off, off, indexing="ij")
        rr = (rr[:, None] + dr.ravel()[None]).ravel(); cc = (cc[:, None] + dc.ravel()[None]).ravel()
    else:
        rr = rr + 0.5; cc = cc + 0.5
    return cc / PPD, -rr / PPD


def ink_mask_on_camera(cam_pts: np.ndarray, shape=(100, 100), sigma: float = 0.0) -> np.ndarray:
    """Rasterise mapped ink points (complex col + i*row) into a camera-grid mask."""
    from scipy.ndimage import gaussian_filter
    m = np.zeros(shape)
    p = cam_pts[np.isfinite(cam_pts)]
    r = np.round(p.imag).astype(int); c = np.round(p.real).astype(int)
    keep = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    np.add.at(m, (r[keep], c[keep]), 1.0)
    return gaussian_filter(m, sigma) if sigma > 0 else m


def overlap_r(resp: np.ndarray, mask: np.ndarray, sigma: float = 2.0) -> float:
    """Pearson r between the response map and the blurred mapped-ink mask (secondary, landmark-free score)."""
    from scipy.ndimage import gaussian_filter
    b = gaussian_filter(mask, sigma)
    ok = np.isfinite(resp)
    if b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(resp[ok], b[ok])[0, 1])
