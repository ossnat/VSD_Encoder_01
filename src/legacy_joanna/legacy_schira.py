"""Faithful port of Amit ``trans_cortex.m`` monopole Double-Sech (model=2, b>=200).

Live lab convention (not the incomplete ``retinotopic_model_joanna.m`` draft)::

    [T, R] = cart2pol(x, y)   # after abs VF sampling
    T = T / alpha
    c = sech(log(R/a) * 0.7609) * 0.1821
    corrPolar = T * sech(T)^c
    w = k * log(R * exp(1i * corrPolar) + a)

Also exposes the paper/thesis multiply convention for A/B comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

AlphaMode = Literal["divide", "multiply"]

ISO_POLAR_GRAD = 0.1821
ECC_WIDTH_LEGACY = 0.7609  # trans_cortex.m
ECC_WIDTH_PAPER = 0.76  # schira.py / retinotopic_model_joanna.m


@dataclass(frozen=True)
class LegacySchiraParams:
    a: float
    alpha: float
    k: float = 1.0
    alpha_mode: AlphaMode = "divide"
    ecc_width: float = ECC_WIDTH_LEGACY
    iso_polar_grad: float = ISO_POLAR_GRAD
    vf_abs: bool = True  # trans_cortex linspace of abs(retino_mat)


def _sech(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(np.clip(np.asarray(x, dtype=np.float64), -20.0, 20.0))


def prepare_vf_xy(
    x_deg: np.ndarray,
    y_deg: np.ndarray,
    *,
    vf_abs: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """VF cartesian degrees as fed into ``cart2pol`` in ``trans_cortex``.

    Legacy path takes ``abs`` of both axes (first quadrant). Current repo uses
    signed lower-right (``y < 0``).
    """
    x = np.asarray(x_deg, dtype=np.float64)
    y = np.asarray(y_deg, dtype=np.float64)
    if vf_abs:
        return np.abs(x), np.abs(y)
    return x, y


def compressed_polar_legacy(
    theta_rad: np.ndarray,
    params: LegacySchiraParams,
) -> np.ndarray:
    """Apply α to polar angle under divide or multiply convention."""
    t = np.asarray(theta_rad, dtype=np.float64)
    if params.alpha_mode == "divide":
        return t / params.alpha
    if params.alpha_mode == "multiply":
        return params.alpha * t
    raise ValueError(f"unknown alpha_mode {params.alpha_mode!r}")


def forward_legacy_schira(
    x_deg: np.ndarray | float,
    y_deg: np.ndarray | float,
    params: LegacySchiraParams,
) -> np.ndarray:
    """Cartesian VF deg → complex cortical ``w`` (continuous, pre-round).

    Matches ``trans_cortex`` monopole algebra before ``abs(angle)`` / ``*1000``.
    """
    x, y = prepare_vf_xy(
        np.asarray(x_deg, dtype=np.float64),
        np.asarray(y_deg, dtype=np.float64),
        vf_abs=params.vf_abs,
    )
    x, y = np.broadcast_arrays(x, y)
    shape = x.shape
    xf = np.ravel(x)
    yf = np.ravel(y)
    out = np.full(xf.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    ok = np.isfinite(xf) & np.isfinite(yf)
    if not np.any(ok):
        return out.reshape(shape)

    # MATLAB cart2pol(x,y) → (theta, rho); numpy angle/hypot match.
    r = np.hypot(xf[ok], yf[ok])
    theta = np.arctan2(yf[ok], xf[ok])
    t = compressed_polar_legacy(theta, params)
    c = _sech(np.log(np.maximum(r, 1e-12) / params.a) * params.ecc_width)
    c = c * params.iso_polar_grad
    corr_polar = t * np.power(_sech(t), c)
    shifted = r * np.exp(1j * corr_polar) + params.a
    w = np.full(shifted.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    valid = np.real(shifted) > 1e-12
    w[valid] = params.k * np.log(shifted[valid])
    out[ok] = w
    return out.reshape(shape)


def cortical_pixel_indices_from_w(
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``trans_cortex`` integer grid: abs(angle), abs(w) → pol2cart → round*1000.

    Returns ``(x1, y1)`` 1-based indices after min-shift (MATLAB style).
    """
    w_arr = np.asarray(w, dtype=np.complex128)
    log_angle = np.abs(np.angle(w_arr))
    log_rad = np.abs(w_arr)
    x = log_rad * np.cos(log_angle)
    y = log_rad * np.sin(log_angle)
    xr = np.round(x * 1000.0)
    yr = np.round(y * 1000.0)
    # Only finite entries participate in min-shift (match MATLAB on valid set).
    finite = np.isfinite(xr) & np.isfinite(yr)
    if not np.any(finite):
        return xr, yr
    x1 = np.full_like(xr, np.nan)
    y1 = np.full_like(yr, np.nan)
    x1[finite] = xr[finite] - np.min(xr[finite]) + 1.0
    y1[finite] = yr[finite] - np.min(yr[finite]) + 1.0
    return x1, y1


def joanna_params() -> LegacySchiraParams:
    """Joanna 201118 Opt winner (25-07-2022-17)."""
    return LegacySchiraParams(a=0.72, alpha=1.5, k=1.0, alpha_mode="divide")


def amit_params() -> LegacySchiraParams:
    """Amit winner201118.mat."""
    return LegacySchiraParams(a=0.39, alpha=1.5, k=1.0, alpha_mode="divide")


def thesis_like_params(a: float = 0.72, alpha: float = 1.5, k: float = 1.0) -> LegacySchiraParams:
    """Paper/thesis multiply + signed VF + ecc_width=0.76 (current-repo-like)."""
    return LegacySchiraParams(
        a=a,
        alpha=alpha,
        k=k,
        alpha_mode="multiply",
        ecc_width=ECC_WIDTH_PAPER,
        vf_abs=False,
    )
