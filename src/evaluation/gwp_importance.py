"""Ridge coefficient importance for Gabor wavelet pyramid features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.encoding.ridge import RidgeEncodeResult


@dataclass
class GwpImportanceMaps:
    """Aggregated |Ridge coefficient| mass for interpretable GWP components."""

    n_scales: int
    n_orientations: int
    wavelengths_px: list[float]
    orientations_deg: list[float]
    # (n_scales, n_orientations) — total coupling per scale/orientation
    scale_orientation: np.ndarray
    # (n_scales,) — marginal over orientation and space
    scale_total: np.ndarray
    # (n_scales, Hf, Wf) — where in pooled stimulus grid matters per scale
    scale_spatial: np.ndarray
    # (Hf, Wf) — marginal feature-space location (all scales/orientations)
    feature_location: np.ndarray
    # (H_out, W_out) — which VSD pixels depend most on GWP features
    vsd_output: np.ndarray
    # (n_orientations,) — marginal over scales
    orientation_total: np.ndarray


def _wavelengths_px(
    *,
    n_scales: int,
    min_wavelength: float,
    wavelength_factor: float,
) -> list[float]:
    return [float(min_wavelength * (wavelength_factor**s)) for s in range(n_scales)]


def compute_gwp_importance(
    result: RidgeEncodeResult,
    *,
    feature_shape: tuple[int, int, int],
    n_scales: int,
    n_orientations: int,
    min_wavelength: float = 3.0,
    wavelength_factor: float = 2**0.5,
) -> GwpImportanceMaps:
    """
    Summarize Ridge weights over flattened GWP maps (C, Hf, Wf).

    Features are stored channel-major (C-order): index = c·Hf·Wf + h·Wf + w.
    """
    c, hf, wf = (int(feature_shape[0]), int(feature_shape[1]), int(feature_shape[2]))
    expected_c = n_scales * n_orientations
    if c != expected_c:
        raise ValueError(f"Expected {expected_c} channels, got {c}")

    out_h, out_w = result.spatial_size
    coef = np.asarray(result.model.coef_, dtype=np.float64)
    if coef.shape != (out_h * out_w, c * hf * wf):
        raise ValueError(f"Unexpected coef shape {coef.shape}")

    abs_coef = np.abs(coef).reshape(out_h, out_w, c, hf, wf)
    soc = abs_coef.reshape(out_h, out_w, n_scales, n_orientations, hf, wf)

    scale_orientation = abs_coef.sum(axis=(0, 1, 3, 4)).reshape(
        n_scales, n_orientations
    )
    scale_total = scale_orientation.sum(axis=1)
    orientation_total = scale_orientation.sum(axis=0)
    scale_spatial = soc.sum(axis=(0, 1, 3))
    feature_location = abs_coef.sum(axis=(0, 1, 2))
    vsd_output = abs_coef.sum(axis=(2, 3, 4))

    orientations_deg = [
        float(deg) for deg in np.linspace(0.0, 157.5, n_orientations, endpoint=True)
    ]
    wavelengths = _wavelengths_px(
        n_scales=n_scales,
        min_wavelength=min_wavelength,
        wavelength_factor=wavelength_factor,
    )

    return GwpImportanceMaps(
        n_scales=n_scales,
        n_orientations=n_orientations,
        wavelengths_px=wavelengths,
        orientations_deg=orientations_deg,
        scale_orientation=scale_orientation.astype(np.float32),
        scale_total=scale_total.astype(np.float32),
        scale_spatial=scale_spatial.astype(np.float32),
        feature_location=feature_location.astype(np.float32),
        vsd_output=vsd_output.astype(np.float32),
        orientation_total=orientation_total.astype(np.float32),
    )


def importance_summary_table(maps: GwpImportanceMaps) -> list[dict[str, float | int | str]]:
    """One row per scale with rank-friendly totals (for optional CSV export)."""
    order = np.argsort(-maps.scale_total)
    rows: list[dict[str, float | int | str]] = []
    for rank, s in enumerate(order, start=1):
        s = int(s)
        best_ori = int(np.argmax(maps.scale_orientation[s]))
        rows.append(
            {
                "rank": rank,
                "scale": s,
                "wavelength_px": maps.wavelengths_px[s],
                "total_importance": float(maps.scale_total[s]),
                "best_orientation_deg": maps.orientations_deg[best_ori],
                "best_orientation_importance": float(
                    maps.scale_orientation[s, best_ori]
                ),
            }
        )
    return rows
