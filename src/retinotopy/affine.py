"""Similarity map from Schira complex ``w`` onto the VSDI camera grid.

``k=1`` in the fitted Schira params means millimetre scaling is **not**
inside ``k``. Ayzenshtat et al. (2012) fitted ``k`` in mm *and then*
registered the warped stimulus with translation, rotation, and scaling
onto the exposed cortex using retinotopic landmarks. No landmark origin
was found in this repo or parent ``VSD_FM`` tree, so the default affine
is an explicit, unconfirmed convention — not a silent fit to VSD blobs.

Pixel convention
----------------
VSD maps are ``reshape(H, W)`` with ``imshow`` default: row 0 is the **top**
of the figure. ROI review notes V2 toward the top of gandalf maps
(``white_point_0.1``). Default rotation is 0: ``Re(w)`` increases to the
right (eccentricity along the HM), ``Im(w)`` increases **up** in the figure
(decreasing row). Lower-field polar angles (``P < 0``) therefore fall
below the foveal origin.

What still needs confirmation
-----------------------------
``origin_xy`` (foveal pixel), ``pixels_per_unit``, ``rotation_deg``, and
flips. Defaults below are labeled ``needs_confirmation`` and must not be
treated as a measured chamber alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.retinotopy.schira import SchiraParams, fovea_w


@dataclass(frozen=True)
class CorticalAffine:
    """Similarity transform: VSDI pixel (row, col) ↔ Schira ``w = u + iv``.

    The fovea ``w = k log(a)`` is placed at ``(origin_x, origin_y)`` in
    ``(col, row)`` pixel coordinates.
    """

    origin_x: float
    origin_y: float
    pixels_per_unit: float
    rotation_deg: float = 0.0
    flip_u: bool = False
    flip_v: bool = False
    needs_confirmation: bool = True

    def __post_init__(self) -> None:
        if self.pixels_per_unit == 0:
            raise ValueError("pixels_per_unit must be non-zero")

    @classmethod
    def from_mapping(cls, data: dict) -> CorticalAffine:
        if "origin_xy" not in data or data["origin_xy"] is None:
            raise ValueError("affine.origin_xy is required (col, row of fovea)")
        if "pixels_per_unit" not in data or data["pixels_per_unit"] is None:
            raise ValueError("affine.pixels_per_unit is required")
        origin = data["origin_xy"]
        return cls(
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            pixels_per_unit=float(data["pixels_per_unit"]),
            rotation_deg=float(data.get("rotation_deg", 0.0)),
            flip_u=bool(data.get("flip_u", False)),
            flip_v=bool(data.get("flip_v", False)),
            needs_confirmation=bool(data.get("needs_confirmation", True)),
        )

    def pixel_to_w(
        self,
        row: np.ndarray | float,
        col: np.ndarray | float,
        params: SchiraParams,
    ) -> np.ndarray:
        """Map camera pixels to complex cortical ``w``."""
        r = np.asarray(row, dtype=np.float64)
        c = np.asarray(col, dtype=np.float64)
        dx = c - self.origin_x
        dy = self.origin_y - r
        if self.flip_u:
            dx = -dx
        if self.flip_v:
            dy = -dy
        theta = np.deg2rad(self.rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        u_off = (dx * cos_t - dy * sin_t) / self.pixels_per_unit
        v_off = (dx * sin_t + dy * cos_t) / self.pixels_per_unit
        return fovea_w(params) + (u_off + 1j * v_off)

    def w_to_pixel(
        self,
        w: np.ndarray | complex,
        params: SchiraParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map complex cortical ``w`` to ``(row, col)``."""
        delta = np.asarray(w, dtype=np.complex128) - fovea_w(params)
        u = np.real(delta) * self.pixels_per_unit
        v = np.imag(delta) * self.pixels_per_unit
        theta = np.deg2rad(self.rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Inverse of the rotation used in pixel_to_w.
        dx = u * cos_t + v * sin_t
        dy = -u * sin_t + v * cos_t
        if self.flip_u:
            dx = -dx
        if self.flip_v:
            dy = -dy
        col = self.origin_x + dx
        row = self.origin_y - dy
        return row, col
