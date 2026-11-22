"""Read the thesis Fig. 12 material made by make_thesis_fig12.py (schira_dashboard/stimuli/thesis_fig12/).

Camera points are complex col + i*row (0-based, 100 x 100 frame), as in schira_geometry.py.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

DIR = Path(__file__).resolve().parent / "stimuli/thesis_fig12"
DATES = ("201118", "100718")
PLATE, PLATE_R = 214, 100.0          # plate size and chamber-circle radius (thesis px)
PANEL = {("polar", "201118"): "A(ii)", ("polar", "100718"): "A(iii)",
         ("cart", "201118"): "B(ii)", ("cart", "100718"): "B(iii)"}


def available() -> bool:
    return (DIR / "grids.csv").is_file() and (DIR / "thesis_registration.json").is_file()


def grids() -> pd.DataFrame:
    g = pd.read_csv(DIR / "grids.csv")
    g["image_path"] = [str(DIR / f) for f in g.image]
    return g


@lru_cache(None)
def _all() -> dict:
    return json.loads((DIR / "thesis_registration.json").read_text())


def _reg(kind: str, date: str) -> dict:
    return _all()[f"{kind}_{date}"]


def joanna_params(date: str) -> dict:
    """Joanna's session winner (a, alpha in the thesis convention P = theta/alpha, k, 2-point anchors)."""
    return _all()[f"joanna_{date}"]


def bv(date: str) -> np.ndarray:
    return np.asarray(Image.open(DIR / f"bv_{date}.png").convert("L")).astype(float)


def plate_to_cam(kind: str, date: str, x, y) -> np.ndarray:
    """Thesis plate pixel (x right, y down) -> camera point (complex col + i*row)."""
    r = _reg(kind, date); s = r["thesis_px_per_cam_px"]; th = np.deg2rad(r["rotation_deg"])
    c, si = np.cos(th), np.sin(th)
    row = r["row0"] + (c * y - si * x) / s
    col = r["col0"] + (si * y + c * x) / s
    return col + 1j * row


def chamber_circle(kind: str, date: str) -> tuple[complex, float]:
    """Centre (camera) and radius (camera px) of the chamber circle shown in the thesis plate."""
    return complex(plate_to_cam(kind, date, 106.5, 106.5)), PLATE_R / _reg(kind, date)["thesis_px_per_cam_px"]


def thesis_image_cam(kind: str, date: str, up: int = 4) -> np.ndarray:
    """The thesis plate resampled into the camera frame (100*up px square), NaN outside the chamber circle."""
    r = _reg(kind, date); s = r["thesis_px_per_cam_px"]; th = np.deg2rad(r["rotation_deg"])
    plate = np.asarray(Image.open(DIR / f"thesis_{kind}_{date}.png").convert("L")).astype(float)
    rr, cc = np.mgrid[0:100:1 / up, 0:100:1 / up]
    dr, dc = rr - r["row0"], cc - r["col0"]; c, si = np.cos(th), np.sin(th)
    ty, tx = s * (c * dr + si * dc), s * (-si * dr + c * dc)
    out = map_coordinates(plate, [ty, tx], order=1, cval=np.nan)
    out[np.hypot(tx - 106.5, ty - 106.5) > PLATE_R] = np.nan
    return out


def thesis_lines_cam(kind: str, date: str) -> np.ndarray:
    """White grid-line pixels of the thesis plate, in camera coordinates."""
    plate = np.asarray(Image.open(DIR / f"thesis_{kind}_{date}.png").convert("RGB")).astype(float)
    g, sat = plate.mean(-1), plate.max(-1) - plate.min(-1)
    yy, xx = np.mgrid[:PLATE, :PLATE]
    m = (g >= 205) & (sat < 25) & (np.hypot(xx - 106.5, yy - 106.5) < PLATE_R - 3)
    y, x = np.nonzero(m)
    return plate_to_cam(kind, date, x.astype(float), y.astype(float))


def joanna_lines_cam(kind: str, date: str) -> np.ndarray:
    """Grid-line pixels of Joanna's prediction (her pipeline, re-run), in camera coordinates."""
    im = np.asarray(Image.open(DIR / f"joanna_{kind}_{date}.png").convert("L"))
    r, c = np.nonzero(im >= 200)
    return c.astype(float) + 1j * r.astype(float)


def inside(pts: np.ndarray, centre: complex, radius: float) -> np.ndarray:
    pts = pts[np.isfinite(pts)]
    return pts[np.abs(pts - centre) < radius]


def chamfer(a: np.ndarray, b: np.ndarray, cap: float = 6.0) -> tuple[float, float]:
    """Mean nearest-neighbour distance a->b and b->a (camera px), each distance capped at `cap` px (thesis lines are
    dashed at this resolution). Line identity is not enforced."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan, np.nan
    ta, tb = cKDTree(np.c_[a.real, a.imag]), cKDTree(np.c_[b.real, b.imag])
    return (float(np.minimum(tb.query(np.c_[a.real, a.imag])[0], cap).mean()),
            float(np.minimum(ta.query(np.c_[b.real, b.imag])[0], cap).mean()))
