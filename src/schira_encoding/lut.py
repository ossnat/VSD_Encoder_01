"""Static VSD→visual-field lookup table on the anchor camera grid.

For each anchor VSD pixel ``(row, col)``::

    (c, r) --CorticalAffine.pixel_to_w--> w
         --inverse_schira--> (E, θ) --> (x_deg, y_deg)
         --cartesian_deg_to_image_px--> stimulus (x_px, y_px)

CNN feature sampling scales those stimulus coordinates into the feature
grid at warp time (``input_size`` / layer spatial size), so one LUT serves
every backbone layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.retinotopy.affine import CorticalAffine
from src.retinotopy.schira import SchiraParams, inverse_schira, polar_to_cartesian
from src.retinotopy.visual_field import cartesian_deg_to_image_px
from src.retinotopy.warp import cortical_w_grid
from src.stimuli.render import RenderConfig


@dataclass
class SchiraLUT:
    """Anchor-grid geometry for Schira encoding.

    Arrays are ``(H, W)`` matching ``spatial_size``. Float sample coords use
    the same convention as ``scipy.ndimage.map_coordinates`` (row, col).
    """

    spatial_size: tuple[int, int]
    canvas_size: int
    input_size: int
    pixels_per_deg: float
    quadrant_extent_deg: float
    schira_set: str
    anchor_session: str
    valid: np.ndarray
    u: np.ndarray
    v: np.ndarray
    x_deg: np.ndarray
    y_deg: np.ndarray
    stim_row: np.ndarray
    stim_col: np.ndarray
    max_ecc_deg: float | None = None

    def __post_init__(self) -> None:
        h, w = self.spatial_size
        for name in (
            "valid",
            "u",
            "v",
            "x_deg",
            "y_deg",
            "stim_row",
            "stim_col",
        ):
            arr = getattr(self, name)
            if arr.shape != (h, w):
                raise ValueError(
                    f"{name} shape {arr.shape} != spatial_size {(h, w)}"
                )

    @property
    def height(self) -> int:
        return int(self.spatial_size[0])

    @property
    def width(self) -> int:
        return int(self.spatial_size[1])

    def feature_sample_coords(
        self, feature_hw: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map stimulus-canvas coords → feature-map sample coords.

        Matches preprocess: stimulus ``canvas_size`` is resized to
        ``input_size``, then the CNN downsamples to ``feature_hw``.
        """
        hf, wf = int(feature_hw[0]), int(feature_hw[1])
        scale_to_input = float(self.input_size) / float(self.canvas_size)
        row_in = self.stim_row * scale_to_input
        col_in = self.stim_col * scale_to_input
        feat_row = row_in * (hf / float(self.input_size))
        feat_col = col_in * (wf / float(self.input_size))
        return feat_row, feat_col

    def to_npz(self) -> dict[str, Any]:
        meta = np.array(
            [
                self.spatial_size[0],
                self.spatial_size[1],
                self.canvas_size,
                self.input_size,
                self.pixels_per_deg,
                self.quadrant_extent_deg,
                -1.0 if self.max_ecc_deg is None else float(self.max_ecc_deg),
            ],
            dtype=np.float64,
        )
        return {
            "valid": self.valid.astype(bool),
            "u": self.u.astype(np.float64),
            "v": self.v.astype(np.float64),
            "x_deg": self.x_deg.astype(np.float64),
            "y_deg": self.y_deg.astype(np.float64),
            "stim_row": self.stim_row.astype(np.float64),
            "stim_col": self.stim_col.astype(np.float64),
            "meta": meta,
            "schira_set": np.asarray(self.schira_set),
            "anchor_session": np.asarray(self.anchor_session),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_npz())

    @classmethod
    def load(cls, path: Path) -> SchiraLUT:
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            meta = z["meta"].astype(np.float64)
            max_ecc = float(meta[6])
            return cls(
                spatial_size=(int(meta[0]), int(meta[1])),
                canvas_size=int(meta[2]),
                input_size=int(meta[3]),
                pixels_per_deg=float(meta[4]),
                quadrant_extent_deg=float(meta[5]),
                schira_set=str(z["schira_set"]),
                anchor_session=str(z["anchor_session"]),
                valid=np.asarray(z["valid"], dtype=bool),
                u=np.asarray(z["u"], dtype=np.float64),
                v=np.asarray(z["v"], dtype=np.float64),
                x_deg=np.asarray(z["x_deg"], dtype=np.float64),
                y_deg=np.asarray(z["y_deg"], dtype=np.float64),
                stim_row=np.asarray(z["stim_row"], dtype=np.float64),
                stim_col=np.asarray(z["stim_col"], dtype=np.float64),
                max_ecc_deg=None if max_ecc < 0 else max_ecc,
            )


def build_schira_lut(
    *,
    params: SchiraParams,
    affine: CorticalAffine,
    schira_set: str,
    anchor_session: str,
    spatial_size: tuple[int, int] = (100, 100),
    render_cfg: RenderConfig | None = None,
    input_size: int = 224,
    max_ecc_deg: float | None = None,
) -> SchiraLUT:
    """Build an anchor-grid LUT from Schira params + camera affine."""
    cfg = render_cfg or RenderConfig()
    height, width = int(spatial_size[0]), int(spatial_size[1])
    w = cortical_w_grid((height, width), affine, params)
    ecc, polar = inverse_schira(w, params, min_real_z=0.0)
    x_deg, y_deg = polar_to_cartesian(ecc, polar)
    stim_col, stim_row = cartesian_deg_to_image_px(x_deg, y_deg, cfg)
    # cartesian_deg_to_image_px returns (x_px, y_px) = (col, row).

    canvas = int(cfg.canvas_size)
    ecc_ok = np.isfinite(ecc) & np.isfinite(polar)
    if max_ecc_deg is not None:
        ecc_ok &= ecc <= float(max_ecc_deg)
    valid = (
        ecc_ok
        & (stim_col >= -0.5)
        & (stim_row >= -0.5)
        & (stim_col < canvas - 0.5)
        & (stim_row < canvas - 0.5)
    )

    u = np.real(w).astype(np.float64)
    v = np.imag(w).astype(np.float64)
    u = np.where(valid, u, np.nan)
    v = np.where(valid, v, np.nan)
    x_deg = np.where(valid, x_deg, np.nan)
    y_deg = np.where(valid, y_deg, np.nan)
    stim_row = np.where(valid, stim_row, np.nan)
    stim_col = np.where(valid, stim_col, np.nan)

    return SchiraLUT(
        spatial_size=(height, width),
        canvas_size=canvas,
        input_size=int(input_size),
        pixels_per_deg=float(cfg.pixels_per_deg),
        quadrant_extent_deg=float(cfg.quadrant_extent_deg),
        schira_set=str(schira_set),
        anchor_session=str(anchor_session),
        valid=valid,
        u=u,
        v=v,
        x_deg=x_deg.astype(np.float64),
        y_deg=y_deg.astype(np.float64),
        stim_row=stim_row.astype(np.float64),
        stim_col=stim_col.astype(np.float64),
        max_ecc_deg=None if max_ecc_deg is None else float(max_ecc_deg),
    )
