"""Differentiable Schira pull-warp used only while fitting camera parameters.

The accepted numbers are written back to :class:`~src.retinotopy.schira.SchiraParams`
and :class:`~src.retinotopy.affine.CorticalAffine`. Lookup tables for the ridge
are still built by the numpy path (``build_schira_lut``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.retinotopy.schira import SchiraParams

# Same Newton constants as ``src.retinotopy.schira``.
_E_MIN = 1e-12
_P_SECH_PEAK = 1.199678640257858
_NEWTON_STEPS = 30


def _sech(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.cosh(x.clamp(-20.0, 20.0))


def _index_to_align_corners(coord: torch.Tensor, size: int) -> torch.Tensor:
    """Pixel-center index → ``grid_sample(..., align_corners=True)`` coordinate."""
    if size <= 1:
        return torch.zeros_like(coord)
    return 2.0 * coord / float(size - 1) - 1.0


def stimulus_sample_coords(
    *,
    origin_x: torch.Tensor,
    origin_y: torch.Tensor,
    rotation_deg: torch.Tensor,
    pixels_per_unit: torch.Tensor,
    flip_u: bool,
    flip_v: bool,
    rows: torch.Tensor,
    cols: torch.Tensor,
    params: SchiraParams,
    pixels_per_deg: float,
    a: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Camera pixels → stimulus ``(row, col)`` and a soft in-canvas weight.

    Mirrors ``CorticalAffine.pixel_to_w`` and ``inverse_schira`` (double-sech).
    ``rows`` / ``cols`` are ``(H, W)`` pixel-center indices.
    """
    if params.shear == "constant":
        raise ValueError("geometry fit supports double_sech only")
    dx = cols - origin_x
    dy = origin_y - rows
    if flip_u:
        dx = -dx
    if flip_v:
        dy = -dy
    theta = torch.deg2rad(rotation_deg)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    u_off = (dx * cos_t - dy * sin_t) / pixels_per_unit
    v_off = (dx * sin_t + dy * cos_t) / pixels_per_unit
    a_t = (
        torch.as_tensor(params.a, dtype=rows.dtype, device=rows.device)
        if a is None
        else a
    )
    alpha_t = (
        torch.as_tensor(params.alpha, dtype=rows.dtype, device=rows.device)
        if alpha is None
        else alpha
    )
    k_t = (
        torch.as_tensor(params.k, dtype=rows.dtype, device=rows.device)
        if k is None
        else k
    )
    u = k_t * torch.log(a_t) + u_off
    v = v_off

    scale = torch.exp(u / k_t)
    zr = scale * torch.cos(v / k_t) - a_t
    zi = scale * torch.sin(v / k_t)
    # hypot/atan2 gradients are undefined at the fovea (z = 0).
    rad2 = zr * zr + zi * zi
    ecc = torch.sqrt(rad2 + 1e-16)
    small = rad2 < 1e-12
    ang_safe = torch.atan2(zi, torch.where(small, torch.ones_like(zr), zr))
    ang = torch.where(small, torch.zeros_like(ang_safe), ang_safe)

    log_term = torch.log(ecc.clamp_min(_E_MIN) / a_t) * float(params.sech_ecc_k)
    exponent = _sech(log_term) * float(params.sech_amp)
    exponent = exponent.clamp_min(1e-12)
    p_lim = torch.minimum(
        torch.maximum(
            torch.full_like(exponent, _P_SECH_PEAK),
            1.0 / exponent.clamp_min(0.05),
        ),
        torch.full_like(exponent, 12.0),
    )
    p = ang.sign() * torch.minimum(ang.abs(), torch.full_like(ang, _P_SECH_PEAK))
    p = torch.minimum(torch.maximum(p, -p_lim), p_lim)
    for _ in range(_NEWTON_STEPS):
        sech_p = _sech(p)
        powered = torch.exp(exponent * torch.log(sech_p.clamp_min(1e-12)))
        g = p * powered - ang
        gp = powered * (1.0 - exponent * p * torch.tanh(p))
        gp = torch.sign(gp) * gp.abs().clamp_min(1e-10)
        gp = torch.where(gp == 0, torch.full_like(gp, 1e-10), gp)
        p = torch.minimum(torch.maximum(p - g / gp, -p_lim), p_lim)
    sech_p = _sech(p)
    powered = torch.exp(exponent * torch.log(sech_p.clamp_min(1e-12)))
    resid = (p * powered - ang).abs()
    theta = p / alpha_t
    x_deg = ecc * torch.cos(theta)
    y_deg = ecc * torch.sin(theta)
    col_px = x_deg * float(pixels_per_deg)
    row_px = -y_deg * float(pixels_per_deg)

    # Soft stand-in for the numpy NaN mask, so boundary pixels keep a gradient.
    reach = torch.sigmoid((1e-4 - resid) / 2e-5)
    right = torch.sigmoid(zr / 1e-2)
    return row_px, col_px, reach * right


def _canvas_gate(
    row_px: torch.Tensor, col_px: torch.Tensor, canvas: int
) -> torch.Tensor:
    edge = 0.75
    row_ok = torch.sigmoid((row_px + 0.5) / edge) * torch.sigmoid(
        (float(canvas) - 0.5 - row_px) / edge
    )
    col_ok = torch.sigmoid((col_px + 0.5) / edge) * torch.sigmoid(
        (float(canvas) - 0.5 - col_px) / edge
    )
    return row_ok * col_ok


def _gaussian_kernel1d(sigma: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    if sigma <= 0:
        return None
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return kernel / kernel.sum()


def blur_maps(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur. ``maps`` is ``(N, 1, H, W)``."""
    kernel = _gaussian_kernel1d(sigma, dtype=maps.dtype, device=maps.device)
    if kernel is None:
        return maps
    radius = (kernel.numel() - 1) // 2
    kx = kernel.view(1, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1)
    blurred = F.conv2d(maps, kx, padding=(0, radius))
    return F.conv2d(blurred, ky, padding=(radius, 0))


def sample_contrast_maps(
    contrast: torch.Tensor,
    *,
    origin_x: torch.Tensor,
    origin_y: torch.Tensor,
    rotation_deg: torch.Tensor,
    pixels_per_unit: torch.Tensor,
    flip_u: bool,
    flip_v: bool,
    rows: torch.Tensor,
    cols: torch.Tensor,
    params: SchiraParams,
    pixels_per_deg: float,
    blur_sigma_px: float,
    a: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Warp ``contrast`` ``(N, 1, canvas, canvas)`` onto the VSD grid ``(N, H, W)``."""
    if contrast.ndim != 4 or contrast.shape[1] != 1:
        raise ValueError(f"Expected contrast (N, 1, H, W), got {tuple(contrast.shape)}")
    canvas = int(contrast.shape[-1])
    if int(contrast.shape[-2]) != canvas:
        raise ValueError(f"Contrast canvas must be square, got {tuple(contrast.shape)}")
    row_px, col_px, field_gate = stimulus_sample_coords(
        origin_x=origin_x,
        origin_y=origin_y,
        rotation_deg=rotation_deg,
        pixels_per_unit=pixels_per_unit,
        flip_u=flip_u,
        flip_v=flip_v,
        rows=rows,
        cols=cols,
        params=params,
        pixels_per_deg=pixels_per_deg,
        a=a,
        alpha=alpha,
        k=k,
    )
    gate = field_gate * _canvas_gate(row_px, col_px, canvas)
    grid = torch.stack(
        [
            _index_to_align_corners(col_px, canvas),
            _index_to_align_corners(row_px, canvas),
        ],
        dim=-1,
    )
    grid = grid.unsqueeze(0).expand(contrast.shape[0], -1, -1, -1)
    sampled = F.grid_sample(
        contrast,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled * gate.unsqueeze(0).unsqueeze(0)
    return blur_maps(sampled, blur_sigma_px).squeeze(1)
