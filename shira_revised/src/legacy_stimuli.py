"""Legacy Gandalf stimuli (301×512 @ 35 ppd), ports of Joanna MATLAB."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

PPD = 35
LEGACY_ROWS = 301
LEGACY_COLS = 512
BG = 130


def _gray_rgb(stim: np.ndarray) -> np.ndarray:
    g = np.clip(stim, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def create_cart_stim(*, extent_deg: float | None = None, ppd: int = PPD) -> np.ndarray:
    """Port of ``create_cart_stim.m`` (0.5° lines on gray).

    If ``extent_deg`` is set, return an ``N×N`` square crop (``N = round(extent_deg*ppd)``)
    with white borders on all four edges — matching thesis Fig. 12 A/B(i) at 3°.
    """
    stimulus = np.full((LEGACY_ROWS, LEGACY_COLS), BG, dtype=np.float64)
    for rad in range(0, LEGACY_ROWS, ppd):
        r = 0 if rad == 0 else rad
        stimulus[r, :] = 255
        stimulus[:, r] = 255
    # MATLAB create_cart_stim.m hardcodes 17 for ppd=35 (not round(ppd/2)=18).
    half = 17 if ppd == 35 else max(int(ppd // 2), 1)
    for hrad in range(half, LEGACY_ROWS, ppd):
        stimulus[hrad, :] = 255
        stimulus[:, hrad] = 255
    rgb = _gray_rgb(stimulus)
    if extent_deg is None:
        return rgb
    n = int(round(float(extent_deg) * ppd))
    out = rgb[:n, :n].copy()
    # Thesis Fig. 12 squares include the far borders (6×6 cells → 7 lines).
    out[n - 1, :] = 255
    out[:, n - 1] = 255
    out[0, :] = 255
    out[:, 0] = 255
    return out


def _draw_line(stim: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    n = int(max(abs(x1 - x0), abs(y1 - y0), 1))
    xs = np.linspace(x0, x1, n + 1)
    ys = np.linspace(y0, y1, n + 1)
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < stim.shape[0] and 0 <= xi < stim.shape[1]:
            stim[yi, xi] = 255


def create_polar_stim(
    ppd: int = PPD,
    *,
    is_left_hem: bool = True,
    extent_deg: float | None = None,
) -> np.ndarray:
    """Port of ``create_polar_stim.m`` (rings + radial rays from top-left).

    MATLAB draws rings out to ``3*ppd`` (3°). Pass ``extent_deg`` to return the
    thesis Fig. 12 square crop (typically 3°) instead of the full 301×512 canvas.
    """
    stimulus = np.full((LEGACY_ROWS, LEGACY_COLS), BG, dtype=np.float64)
    seed = np.zeros((LEGACY_ROWS, LEGACY_COLS), dtype=bool)
    seed[0, 0] = True
    d = distance_transform_edt(~seed)
    # Legacy: every 0.5° from 0.5° through 3°.
    max_ring = int(round(3 * ppd))
    if extent_deg is not None:
        max_ring = max(max_ring, int(round(float(extent_deg) * ppd)))
    # MATLAB: round(ppd/2) and ppd:ppd:ppd*3 — for ppd=35 → 18,35,...,105
    half = int(round(ppd / 2))
    radii = sorted(set(list(range(ppd, max_ring + 1, ppd)) + list(range(half, max_ring + 1, ppd))))
    for rad in radii:
        mask = (d >= rad - 1) & (d < rad)
        stimulus[mask] = 255
    angles = [0, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    ray_len = 200
    if extent_deg is not None:
        ray_len = max(ray_len, int(round(float(extent_deg) * ppd * 1.5)))
    for ang in angles:
        a = np.deg2rad(90.0 - float(ang))
        x1 = int(round(ray_len * np.cos(a)))
        y1 = int(round(ray_len * np.sin(a)))
        _draw_line(stimulus, 0, 0, x1, y1)
    if not is_left_hem:
        stimulus = np.fliplr(stimulus)
    rgb = _gray_rgb(stimulus)
    if extent_deg is None:
        return rgb
    n = int(round(float(extent_deg) * ppd))
    return rgb[:n, :n].copy()


def vertical_line(
    center: tuple[float, float],
    input_size_deg: float = 1.0,
    ppd: int = PPD,
    *,
    vert: bool = True,
) -> np.ndarray:
    """Port of ``vertical_line.m`` (``center`` = ``(x_deg, y_deg)``)."""
    x_deg, y_deg = center
    stim = np.full((LEGACY_ROWS, LEGACY_COLS), BG, dtype=np.float64)
    xs = int(round(x_deg * ppd))
    ys = int(abs(round(y_deg * ppd)))
    half = int(input_size_deg * ppd / 2)
    if vert:
        r0 = max(0, ys - half)
        r1 = min(LEGACY_ROWS, ys + half + 1)
        stim[r0:r1, xs] = 255
    else:
        c0 = max(0, xs - half)
        c1 = min(LEGACY_COLS, xs + half + 1)
        stim[ys, c0:c1] = 255
    return _gray_rgb(stim)


def create_letter_stim(
    letter: str,
    stim_center: tuple[float, float],
    *,
    bmp_path: str | Path | None = None,
    ppd: int = PPD,
) -> np.ndarray:
    """Port of ``create_letter_stim.m`` (35×35 letter patch on 301×512).

    ``stim_center`` is ``(x_deg, y_deg)`` as in Joanna (session a: ``(1, -0.9)``).
    """
    letter = letter.upper()
    list_letters = ["G", "A", "N", "D", "L", "F"]
    if letter not in list_letters:
        raise ValueError(f"letter must be one of {list_letters}")
    if bmp_path is None:
        bmp_path = (
            Path("/Users/ossnat/GondaResearch/VSD_FM/Data/EncoderData/letters_stimuli")
            / f"{letter}.bmp"
        )
    bmp = np.asarray(Image.open(bmp_path))
    if bmp.ndim == 3:
        bmp = bmp[:, :, 0]
    cdata = bmp.astype(np.float64)
    # MATLAB: 130 → letter (1), 129 → background (0)
    patch = cdata[281:316, 381:416].copy()
    patch = np.where(patch == 130, 1.0, np.where(patch == 129, 0.0, patch))

    coord_x = abs(float(stim_center[0]))
    coord_y = abs(float(stim_center[1]))
    cx = int(round(ppd * coord_x))
    cy = int(round(ppd * coord_y))
    x0, x1 = cx - 17, cx - 17 + 35
    y0, y1 = cy - 17, cy - 17 + 35

    stimulus = np.zeros((LEGACY_ROWS, LEGACY_COLS), dtype=np.float64)
    stimulus[y0:y1, x0:x1] = patch
    stimulus[stimulus == 0] = BG / 255.0
    stim_u8 = np.clip(np.round(stimulus * 255.0), 0, 255).astype(np.uint8)
    return _gray_rgb(stim_u8)
