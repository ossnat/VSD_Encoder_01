"""Thesis monopole Double-Sech (Schwartz log-polar + Schira shear).

Lab / thesis form::

    θ = atan2(y_deg, x_deg)          # radians, HM = 0, below HM negative
    P = α * θ                        # angular compression
    s_P = sech(P)
    s_E = sech(log(E / a) * sech_ecc_k)
    f_a = s_P ** (s_E * sech_amp)      # Schira 2007 eq.5 (power / superscript)
    w(E, P) = k * log(E * exp(i * P * f_a) + a)

Defaults ``sech_ecc_k = 0.76``, ``sech_amp = 0.1821``. Both YAML-overridable.

Schira 2007 (J Neurophysiol eq. 5): PDF typography has the second sech
*and* ``* S2`` superscripted on the first sech — i.e.::

    fa = sech(P) ** ( sech{log(E/a)*S1} * S2 )

(with S1=0.76, S2≈0.18). Flat-text “product / mult” readings
(``fa = sech(P) * sech(E) * S2``) are wrong and are not implemented.

``θ`` and ``P`` are always radians. ``E`` is eccentricity in degrees.
``shear="constant"`` (``f_a = α``) is ablation only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHEAR_DOUBLE_SECH = "double_sech"
SHEAR_SCHIRA2007 = "schira2007"
SHEAR_AYZENSHTAT = "ayzenshtat"
SHEAR_CONSTANT = "constant"
_DOUBLE_SECH_MODES = frozenset(
    {SHEAR_DOUBLE_SECH, SHEAR_SCHIRA2007, SHEAR_AYZENSHTAT}
)
_SHEAR_MODES = _DOUBLE_SECH_MODES | {SHEAR_CONSTANT}

# YAML ``fa_combine`` is power only (Schira 2007 eq.5 superscript).
FA_COMBINE_POWER = "power"
_FA_COMBINE_REMOVED = frozenset({"product", "mult", "multiply", "mul"})
_FA_COMBINE_MODES = frozenset({FA_COMBINE_POWER})

# Schira 2010 algebraic defaults when YAML omits the keys.
DEFAULT_SECH_ECC_K = 0.76
DEFAULT_SECH_AMP = 0.1821
DEFAULT_FA_COMBINE = FA_COMBINE_POWER

_E_MIN = 1e-12
_BRANCH_RE_MIN = 1e-12
# Peak of P sech(P): P tanh(P) = 1. Used as a Newton start for the power inverse.
_P_SECH_PEAK = 1.199678640257858


def uses_double_sech(shear: str) -> bool:
    return shear in _DOUBLE_SECH_MODES


@dataclass(frozen=True)
class SchiraParams:
    """Monopole parameters for one imaging day (from YAML)."""

    a: float
    alpha: float
    k: float
    shear: str = SHEAR_DOUBLE_SECH
    sech_ecc_k: float = DEFAULT_SECH_ECC_K
    sech_amp: float = DEFAULT_SECH_AMP
    fa_combine: str = DEFAULT_FA_COMBINE

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError(f"Schira a must be > 0, got {self.a}")
        if self.k == 0:
            raise ValueError("Schira k must be non-zero")
        if self.shear not in _SHEAR_MODES:
            raise ValueError(
                f"shear must be one of {sorted(_SHEAR_MODES)}, got {self.shear!r}"
            )
        mode = str(self.fa_combine).lower()
        if mode in _FA_COMBINE_REMOVED:
            raise ValueError(
                "fa_combine product/mult is not implemented. "
                "Schira 2007 eq.5 is power only: "
                "fa = sech(P) ** (sech(log(E/a)*S1) * S2)."
            )
        if mode not in _FA_COMBINE_MODES:
            raise ValueError(
                f"fa_combine must be {FA_COMBINE_POWER!r}, got {self.fa_combine!r}"
            )


def cartesian_to_polar(
    x_deg: np.ndarray | float, y_deg: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Eccentricity ``E`` (deg) and polar angle ``θ`` (rad) from Cartesian deg.

    ``θ = atan2(y, x)``: horizontal meridian is 0, lower field (``y < 0``)
    is negative. Not degrees.
    """
    x = np.asarray(x_deg, dtype=np.float64)
    y = np.asarray(y_deg, dtype=np.float64)
    ecc = np.hypot(x, y)
    polar = np.arctan2(y, x)
    return ecc, polar


def polar_to_cartesian(
    ecc_deg: np.ndarray | float, polar_rad: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian visual degrees from ``E`` (deg) and ``θ`` (rad)."""
    ecc = np.asarray(ecc_deg, dtype=np.float64)
    polar = np.asarray(polar_rad, dtype=np.float64)
    return ecc * np.cos(polar), ecc * np.sin(polar)


def compressed_polar(theta_rad: np.ndarray, params: SchiraParams) -> np.ndarray:
    """``P = α θ`` (radians)."""
    return params.alpha * np.asarray(theta_rad, dtype=np.float64)


def _sech(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(np.clip(np.asarray(x, dtype=np.float64), -20.0, 20.0))


def shear_fa(
    ecc_deg: np.ndarray,
    compressed_polar_rad: np.ndarray,
    params: SchiraParams,
) -> np.ndarray:
    """``f_a(E, P)`` with ``P`` already ``α θ`` (radians).

    Double-sech::

        s_P = sech(P)
        s_E = sech(log(E / a) * sech_ecc_k)
        f_a = s_P ** (s_E * sech_amp)
        # Schira 2007 eq.5: whole (sech{...}*S2) is the exponent

    Constant shear: ``f_a = α`` (ignores ``P``; ablation only).
    """
    if params.shear == SHEAR_CONSTANT:
        ecc_a, _p_a = np.broadcast_arrays(
            np.asarray(ecc_deg, dtype=np.float64),
            np.asarray(compressed_polar_rad, dtype=np.float64),
        )
        return np.full(ecc_a.shape, float(params.alpha))
    ecc = np.maximum(np.asarray(ecc_deg, dtype=np.float64), _E_MIN)
    p = np.asarray(compressed_polar_rad, dtype=np.float64)
    s_p = _sech(p)
    s_e = _sech(np.log(ecc / params.a) * params.sech_ecc_k)
    # sech(P) in (0, 1]; exponent s_E*S2 is positive.
    return np.power(s_p, s_e * params.sech_amp)


def _sheared_z(
    ecc_deg: np.ndarray,
    theta_rad: np.ndarray,
    params: SchiraParams,
) -> np.ndarray:
    """``z = E exp(i P f_a)`` with ``P = α θ`` unless constant shear."""
    ecc = np.asarray(ecc_deg, dtype=np.float64)
    theta = np.asarray(theta_rad, dtype=np.float64)
    if uses_double_sech(params.shear):
        p = compressed_polar(theta, params)
    else:
        p = theta
    fa = shear_fa(ecc, p, params)
    return ecc * np.exp(1j * p * fa)


def forward_schira(
    ecc_deg: np.ndarray | float,
    theta_rad: np.ndarray | float,
    params: SchiraParams,
) -> np.ndarray:
    """Visual field ``(E, θ)`` → complex cortical ``w``.

    ``theta_rad`` is uncompressed polar angle in radians (not ``P``).
    """
    ecc = np.asarray(ecc_deg, dtype=np.float64)
    polar = np.asarray(theta_rad, dtype=np.float64)
    ecc, polar = np.broadcast_arrays(ecc, polar)
    shape = ecc.shape
    ecc_f = np.ravel(ecc)
    polar_f = np.ravel(polar)
    out_f = np.full(ecc_f.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    ok = np.isfinite(ecc_f) & np.isfinite(polar_f) & (ecc_f >= 0.0)
    if np.any(ok):
        z = _sheared_z(ecc_f[ok], polar_f[ok], params)
        shifted = z + params.a
        valid = np.real(shifted) > _BRANCH_RE_MIN
        w = np.full(shifted.shape, np.nan + 1j * np.nan, dtype=np.complex128)
        w[valid] = params.k * np.log(shifted[valid])
        out_f[ok] = w
    return out_f.reshape(shape)


def _inverse_theta_constant(
    z: np.ndarray, params: SchiraParams
) -> tuple[np.ndarray, np.ndarray]:
    ecc = np.abs(z)
    theta = np.angle(z) / params.alpha
    return ecc, theta


def _inverse_theta_double_sech(
    z: np.ndarray, params: SchiraParams
) -> tuple[np.ndarray, np.ndarray]:
    """Invert ``angle(z) = P f_a(E,P)`` on the injective ``|P|`` branch.

    Power: ``angle = P sech(P)^{s_E · S2}`` with ``s_E = sech(log(E/a)·S1)``.
    Samples outside the injective range return NaN (do **not** clip onto
    the peak ridge).
    """
    ecc = np.abs(z)
    target = np.angle(z)
    log_term = np.log(np.maximum(ecc, _E_MIN) / params.a) * params.sech_ecc_k
    s_e = _sech(log_term)
    amp = max(float(params.sech_amp), 1e-12)

    # angle = P * sech(P)^(s_E * S2)
    exponent = s_e * amp
    rhs = target
    p = np.sign(rhs) * np.minimum(np.abs(rhs), _P_SECH_PEAK)
    # Peak of P sech(P)^exp where exp = s_E*S2 ≪ 1 is farther out.
    p_lim = np.maximum(_P_SECH_PEAK, 1.0 / np.maximum(exponent, 0.05))
    p_lim = np.minimum(p_lim, 12.0)
    p = np.clip(p, -p_lim, p_lim)
    for _ in range(30):
        sech_p = _sech(p)
        g = p * np.power(sech_p, exponent) - rhs
        gp = np.power(sech_p, exponent) * (1.0 - exponent * p * np.tanh(p))
        gp = np.where(np.abs(gp) < 1e-10, 1e-10, gp)
        p = np.clip(p - g / gp, -p_lim, p_lim)
    resid = np.abs(p * np.power(_sech(p), exponent) - rhs)
    reachable = np.isfinite(resid) & (resid < 1e-6)
    theta = p / params.alpha
    ecc_out = np.where(reachable, ecc, np.nan)
    theta_out = np.where(reachable, theta, np.nan)
    return ecc_out, theta_out


def inverse_schira(
    w: np.ndarray | complex,
    params: SchiraParams,
    *,
    min_real_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cortical ``w`` → visual-field ``(E, θ)`` with ``θ`` in radians.

    Inverse: ``z = exp(w / k) - a``, then undo ``P = α θ`` and ``f_a``.
    NaN where ``w`` is non-finite or ``Re(z) < min_real_z`` (left hemifield).
    """
    w_arr = np.asarray(w, dtype=np.complex128)
    shape = w_arr.shape
    w_f = np.ravel(w_arr)
    ecc_f = np.full(w_f.shape, np.nan, dtype=np.float64)
    theta_f = np.full(w_f.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(w_f.real) & np.isfinite(w_f.imag)
    if np.any(ok):
        z = np.exp(w_f[ok] / params.k) - params.a
        valid = np.real(z) >= min_real_z
        if uses_double_sech(params.shear):
            e, th = _inverse_theta_double_sech(z, params)
        else:
            e, th = _inverse_theta_constant(z, params)
        ecc_f[ok] = np.where(valid, e, np.nan)
        theta_f[ok] = np.where(valid, th, np.nan)
    return ecc_f.reshape(shape), theta_f.reshape(shape)


def fovea_w(params: SchiraParams) -> complex:
    """Cortical coordinate of the fovea (``E = 0``)."""
    return complex(params.k * np.log(params.a), 0.0)


def cortical_uv_to_visual_deg(
    u: np.ndarray | float,
    v: np.ndarray | float,
    params: SchiraParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Invert cortical ``(u, v)`` to visual-field degrees.

    Full inverse of Double-Sech: ``z = exp(w/k) - a``, then undo ``P f_a``.
    On the horizontal meridian (``v = 0``) this is the monopole radial inverse::

        E = exp(u / k) - a
          = a * (exp((u - k log a) / k) - 1)

    The second line is the Ayzenshtat / Schwartz form with the origin shifted
    so the fovea is at ``u' = 0``. Polar angle is **not** ``P = v/k`` near the
    fovea (``a`` is comparable to ``E``); shear is inverted numerically.

    Returns
    -------
    ecc_deg, theta_deg, x_deg, y_deg
        ``theta_deg`` from ``atan2(y, x)``: HM = 0, lower field negative.
    """
    u_a = np.asarray(u, dtype=np.float64)
    v_a = np.asarray(v, dtype=np.float64)
    u_a, v_a = np.broadcast_arrays(u_a, v_a)
    ecc, theta = inverse_schira(u_a + 1j * v_a, params)
    x_deg, y_deg = polar_to_cartesian(ecc, theta)
    return ecc, np.rad2deg(theta), x_deg, y_deg
