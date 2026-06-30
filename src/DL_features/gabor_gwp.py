"""Gabor wavelet pyramid feature extractor (Serre et al. style C1 energy)."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gabor_kernel(
    *,
    kernel_size: int,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float,
    psi: float,
) -> np.ndarray:
    """Real-valued Gabor kernel (OpenCV-style parameterization)."""
    half = kernel_size // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-0.5 * (x_theta**2 + (gamma**2) * (y_theta**2)) / sigma**2)
    gb *= np.cos(2 * np.pi * x_theta / lambd + psi)
    gb -= gb.mean()
    norm = np.sqrt((gb**2).sum())
    if norm > 1e-12:
        gb /= norm
    return gb.astype(np.float32)


def _build_gabor_bank(
    *,
    n_scales: int,
    n_orientations: int,
    kernel_size: int,
    min_wavelength: float,
    wavelength_factor: float,
    gamma: float = 0.5,
) -> np.ndarray:
    """Quadrature pairs: even/odd phase per scale and orientation -> (C, 1, K, K)."""
    kernels: list[np.ndarray] = []
    thetas = np.linspace(0.0, np.pi, n_orientations, endpoint=False)
    for scale_idx in range(n_scales):
        lambd = min_wavelength * (wavelength_factor**scale_idx)
        sigma = 0.56 * lambd
        for theta in thetas:
            for psi in (0.0, math.pi / 2.0):
                kernels.append(
                    _gabor_kernel(
                        kernel_size=kernel_size,
                        sigma=sigma,
                        theta=float(theta),
                        lambd=float(lambd),
                        gamma=gamma,
                        psi=psi,
                    )
                )
    bank = np.stack(kernels, axis=0)[:, np.newaxis, :, :]
    return bank.astype(np.float32)


class GaborWaveletPyramid(nn.Module):
    """
  Fixed Gabor filter bank + local energy (sum of squared quadrature responses).

  Output channels = n_scales * n_orientations (energy per scale/orientation).
  """

    def __init__(
        self,
        *,
        n_scales: int = 5,
        n_orientations: int = 8,
        kernel_size: int = 31,
        min_wavelength: float = 3.0,
        wavelength_factor: float = math.sqrt(2.0),
        gamma: float = 0.5,
        padding: str = "same",
        energy_mode: str = "sum_squares",
    ) -> None:
        super().__init__()
        if padding not in {"same", "valid"}:
            raise ValueError(f"Unsupported padding: {padding!r}")
        if energy_mode not in {"sum_squares", "sqrt_contrast"}:
            raise ValueError(f"Unsupported energy_mode: {energy_mode!r}")
        self.n_scales = int(n_scales)
        self.n_orientations = int(n_orientations)
        self.padding_mode = padding
        self.energy_mode = energy_mode
        bank = _build_gabor_bank(
            n_scales=self.n_scales,
            n_orientations=self.n_orientations,
            kernel_size=kernel_size,
            min_wavelength=min_wavelength,
            wavelength_factor=wavelength_factor,
            gamma=gamma,
        )
        n_pairs = bank.shape[0]
        if n_pairs != 2 * self.n_scales * self.n_orientations:
            raise RuntimeError("Expected two quadrature kernels per scale/orientation")
        weight = torch.from_numpy(bank)
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer(
            "_pair_count",
            torch.tensor(n_pairs, dtype=torch.int64),
            persistent=False,
        )
        pad = kernel_size // 2
        self._pad = (pad, pad, pad, pad)

    @property
    def out_channels(self) -> int:
        return self.n_scales * self.n_orientations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"Gabor extractor expects 1-channel input, got {x.shape[1]}")
        if self.padding_mode == "same":
            x = F.pad(x, self._pad, mode="reflect")
        responses = F.conv2d(x, self.weight, bias=None)
        n_pairs = int(self._pair_count.item())
        n_energy = n_pairs // 2
        even = responses[:, 0::2]
        odd = responses[:, 1::2]
        quad = even.square() + odd.square()
        if self.energy_mode == "sqrt_contrast":
            energy = torch.sqrt(quad + 1e-12)
        else:
            energy = quad
        return energy


class GaborGWPFextractor(nn.Module):
    """Gabor energy pyramid with optional spatial average pooling."""

    def __init__(
        self,
        pyramid: GaborWaveletPyramid,
        *,
        pool: str | None = None,
        pool_size: int | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.pyramid = pyramid
        self.pool = pool
        if pool is not None and pool != "avg":
            raise ValueError(f"Unsupported pool mode: {pool!r}")
        if pool == "avg":
            if pool_size is None:
                raise ValueError("pool_size is required when pool='avg'")
            if isinstance(pool_size, int):
                self.pool_size = (int(pool_size), int(pool_size))
            else:
                self.pool_size = (int(pool_size[0]), int(pool_size[1]))
        else:
            self.pool_size = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = self.pyramid(x)
        if self.pool_size is not None:
            energy = F.adaptive_avg_pool2d(energy, self.pool_size)
        return energy


def build_gabor_gwp_extractor(model_cfg: dict) -> nn.Module:
    gwp_cfg = model_cfg.get("gwp", {}) or {}
    pyramid = GaborWaveletPyramid(
        n_scales=int(gwp_cfg.get("number_of_scales", 5)),
        n_orientations=int(gwp_cfg.get("number_of_directions", 8)),
        kernel_size=int(gwp_cfg.get("kernel_size", 31)),
        min_wavelength=float(gwp_cfg.get("min_wavelength", 3.0)),
        wavelength_factor=float(gwp_cfg.get("wavelength_factor", 2**0.5)),
        gamma=float(gwp_cfg.get("gamma", 0.5)),
        padding=str(gwp_cfg.get("padding", "same")),
        energy_mode=str(gwp_cfg.get("energy_mode", "sum_squares")),
    )
    pool = model_cfg.get("pool") or gwp_cfg.get("pool")
    pool_size = model_cfg.get("pool_size") or gwp_cfg.get("spatial_pool_size")
    if pool or pool_size:
        return GaborGWPFextractor(pyramid, pool=pool or "avg", pool_size=pool_size)
    return pyramid
