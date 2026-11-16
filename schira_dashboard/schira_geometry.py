"""Geometry for the Schira dashboard: visual field -> Schira w -> camera, with a fitted registration (inner loop)
and an optional search over Schira's a, alpha, k (outer loop).

Conventions
  * Schira is the repo implementation (src/retinotopy/schira.py, double-sech, S1 0.76, S2 0.1821):
        w = k * log(E * exp(i P fa) + a)
    'repo'   : P = alpha * theta   (Ayzenshtat 2012, the repo)
    'thesis' : P = theta / alpha   (legacy MATLAB / thesis values, e.g. alpha 1.5 == repo 0.667)
  * Visual field: x right, y up, degrees from fixation (lower field y < 0).
  * Camera points are complex u = col + i*row (0-based pixels of the 100 x 100 VSD map); 1 px = 0.17 mm.
  * Registration (inner loop), u = c1 * w' + c0 with w' = w or conj(w) (mirror):
      'similarity' : c1 free (rotation + scale) -> k has NO effect (absorbed by the scale)
      'rigid'      : |c1| = 1 / 0.17 px per mm, only rotation free -> w is read in mm, so k is the mm scale
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


# ---------------------------------------------------------------- Schira
def schira_w(x_deg, y_deg, a: float, alpha: float, k: float, convention: str = "repo") -> np.ndarray:
    alpha_repo = alpha if convention == "repo" else 1.0 / alpha
    E, th = cartesian_to_polar(np.asarray(x_deg, float), np.asarray(y_deg, float))
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


def fit_session(x_deg, y_deg, cam, pair_ids, a, alpha, k, convention="repo", mode="similarity",
                mirror="auto", loo=True) -> FitResult:
    """Inner loop: best registration (and bar-end pairing, and mirror if 'auto') for fixed Schira parameters.
    Visual-field coordinates are permuted within bar pairs; camera points stay put."""
    x = np.asarray(x_deg, float); y = np.asarray(y_deg, float); cam = np.asarray(cam, complex)
    w_all = schira_w(x, y, a, alpha, k, convention)
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


# ---------------------------------------------------------------- outer loop
def optimize_schira(x_deg, y_deg, cam, pair_ids, convention="repo", mode="similarity", mirror="auto",
                    start=(0.7, 1.0, 1.0), fit_k=None, bounds=((0.02, 20), (0.05, 20), (0.1, 100))):
    """Nested optimisation: outer loop over (a, alpha[, k]) on a log scale, inner loop = fit_session.
    k is only searched in 'rigid' mode (in 'similarity' mode it has no effect). Multi-start Nelder-Mead."""
    fit_k = (mode == "rigid") if fit_k is None else fit_k
    lb = np.log([b[0] for b in bounds]); ub = np.log([b[1] for b in bounds])

    def unpack(p):
        p = np.clip(p, lb[:len(p)], ub[:len(p)])
        a, al = np.exp(p[:2]); k = np.exp(p[2]) if fit_k else start[2]
        return a, al, k

    def cost(p):
        r = fit_session(x_deg, y_deg, cam, pair_ids, *unpack(p), convention, mode, mirror, loo=False)
        # every landmark must stay inside Schira's valid range, otherwise dropping points would 'improve' the RMSE
        return r.rmse_px if (np.isfinite(r.rmse_px) and r.n_used == len(x_deg)) else 1e6

    starts = [[np.log(a0), np.log(al0)] + ([np.log(k0)] if fit_k else [])
              for a0 in (0.1, 0.5, 2.0) for al0 in (0.5, 1.0, 2.0) for k0 in ((2.0, 8.0) if fit_k else (1,))]
    starts.append(list(np.log(start[:3 if fit_k else 2])))
    best = min((minimize(cost, s, method="Nelder-Mead", options=dict(maxiter=800, xatol=1e-3, fatol=1e-4)) for s in starts),
               key=lambda r: r.fun)
    a, al, k = unpack(best.x)
    return dict(a=float(a), alpha=float(al), k=float(k), rmse_px=float(best.fun), n=len(x_deg))


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
