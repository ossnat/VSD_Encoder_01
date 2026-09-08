"""Rigid / similarity 2D transform: moving camera pixels → fixed (anchor) pixels.

Pixel convention matches ``imshow`` / ``CorticalAffine``: ``row`` increases
down, ``col`` increases right. ``translation_xy`` is ``[dx_col, dy_row]``.

Positive ``rotation_deg`` is counter-clockwise on the displayed image
(scipy.ndimage.rotate convention). Rotation is about ``center_xy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.ndimage import affine_transform


def default_center_xy(shape: tuple[int, int]) -> tuple[float, float]:
    """Geometric pixel center ``(col, row)`` for an ``(H, W)`` array."""
    height, width = int(shape[0]), int(shape[1])
    return ((width - 1) / 2.0, (height - 1) / 2.0)


def _rotation_matrix_yx(theta_rad: float) -> np.ndarray:
    """2×2 CCW-on-screen rotation acting on ``(row, col)`` offsets."""
    cos_t = float(np.cos(theta_rad))
    sin_t = float(np.sin(theta_rad))
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)


@dataclass(frozen=True)
class SessionTransform:
    """Maps a pixel in the **moving** image into the **fixed** image.

    ``p_fixed = scale * R(rotation_deg) @ (p_moving - c) + c + t``
    with ``p = (row, col)``, ``c = (center_row, center_col)``,
    ``t = (dy_row, dx_col)``.
    """

    rotation_deg: float
    translation_xy: tuple[float, float]
    transform_type: str = "rigid"
    scale: float = 1.0
    center_xy: tuple[float, float] = (49.5, 49.5)
    spatial_size: tuple[int, int] = (100, 100)

    def __post_init__(self) -> None:
        if self.scale == 0:
            raise ValueError("scale must be non-zero")
        t = str(self.transform_type).strip().lower()
        if t not in ("translation", "rigid", "similarity"):
            raise ValueError(f"Unknown transform_type {self.transform_type!r}")

    @property
    def dx_col(self) -> float:
        return float(self.translation_xy[0])

    @property
    def dy_row(self) -> float:
        return float(self.translation_xy[1])

    @property
    def center_col(self) -> float:
        return float(self.center_xy[0])

    @property
    def center_row(self) -> float:
        return float(self.center_xy[1])

    @property
    def translation_magnitude_px(self) -> float:
        return float(np.hypot(self.dx_col, self.dy_row))

    def invert(self) -> SessionTransform:
        """Closed-form inverse: fixed → moving."""
        theta = np.deg2rad(self.rotation_deg)
        rot_inv = _rotation_matrix_yx(-theta)
        t_yx = np.array([self.dy_row, self.dx_col], dtype=np.float64)
        t_inv_yx = -rot_inv @ t_yx / float(self.scale)
        inv_type = self.transform_type
        if abs(float(self.scale) - 1.0) > 1e-12:
            inv_type = "similarity"
        return SessionTransform(
            rotation_deg=-float(self.rotation_deg),
            translation_xy=(float(t_inv_yx[1]), float(t_inv_yx[0])),
            transform_type=inv_type,
            scale=1.0 / float(self.scale),
            center_xy=self.center_xy,
            spatial_size=self.spatial_size,
        )

    def apply_to_points(
        self,
        rows: np.ndarray | float,
        cols: np.ndarray | float,
        *,
        inverse: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map point coordinates. ``inverse=True`` is fixed → moving."""
        if inverse:
            return self.invert().apply_to_points(rows, cols, inverse=False)
        y = np.asarray(rows, dtype=np.float64)
        x = np.asarray(cols, dtype=np.float64)
        cy, cx = self.center_row, self.center_col
        theta = np.deg2rad(self.rotation_deg)
        rot = _rotation_matrix_yx(theta)
        y0 = y - cy
        x0 = x - cx
        y1 = float(self.scale) * (rot[0, 0] * y0 + rot[0, 1] * x0)
        x1 = float(self.scale) * (rot[1, 0] * y0 + rot[1, 1] * x0)
        return y1 + cy + self.dy_row, x1 + cx + self.dx_col

    def apply_to_map(
        self,
        image: np.ndarray,
        *,
        inverse: bool = False,
        cval: float = 0.0,
        order: int = 1,
    ) -> np.ndarray:
        """Warp an image with bilinear sampling (``order=1``).

        ``inverse=False``: moving image → fixed grid (``T``).
        ``inverse=True``: fixed image → moving grid (``T^{-1}``).
        """
        image = np.asarray(image, dtype=np.float64)
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image, got shape {image.shape}")
        sampler = self if inverse else self.invert()
        matrix, offset = sampler._affine_forward_matrix()
        warped = affine_transform(
            image,
            matrix,
            offset=offset,
            output_shape=image.shape,
            order=int(order),
            mode="constant",
            cval=float(cval),
            prefilter=order > 1,
        )
        return warped.astype(np.float32)

    def _affine_forward_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """``p_out_in_input = matrix @ p_output + offset`` for ``affine_transform``.

        This matrix implements the *forward* map of ``self`` (output coords
        are the domain, input coords are ``self(output)``).
        """
        theta = np.deg2rad(self.rotation_deg)
        rot = _rotation_matrix_yx(theta)
        s = float(self.scale)
        matrix = s * rot
        cy, cx = self.center_row, self.center_col
        dy, dx = self.dy_row, self.dx_col
        # p_in = s R (p_out - c) + c + t  → offset = c + t - s R c
        c = np.array([cy, cx], dtype=np.float64)
        t = np.array([dy, dx], dtype=np.float64)
        offset = c + t - matrix @ c
        return matrix, offset

    def to_mapping(self) -> dict[str, Any]:
        inv = self.invert()
        return {
            "transform_type": str(self.transform_type),
            "rotation_deg": float(self.rotation_deg),
            "translation_xy": [float(self.dx_col), float(self.dy_row)],
            "scale": float(self.scale),
            "center_xy": [float(self.center_col), float(self.center_row)],
            "spatial_size": [int(self.spatial_size[0]), int(self.spatial_size[1])],
            "inverse": {
                "rotation_deg": float(inv.rotation_deg),
                "translation_xy": [float(inv.dx_col), float(inv.dy_row)],
                "scale": float(inv.scale),
            },
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> SessionTransform:
        trans = data.get("translation_xy", (0.0, 0.0))
        center = data.get("center_xy", (49.5, 49.5))
        spatial = data.get("spatial_size", (100, 100))
        return cls(
            rotation_deg=float(data.get("rotation_deg", 0.0)),
            translation_xy=(float(trans[0]), float(trans[1])),
            transform_type=str(data.get("transform_type", "rigid")),
            scale=float(data.get("scale", 1.0)),
            center_xy=(float(center[0]), float(center[1])),
            spatial_size=(int(spatial[0]), int(spatial[1])),
        )

    def save_yaml(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        payload = dict(extra or {})
        payload.update(self.to_mapping())
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False))

    @classmethod
    def load_yaml(cls, path: str | Path) -> tuple[SessionTransform, dict[str, Any]]:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a mapping")
        return cls.from_mapping(raw), raw
