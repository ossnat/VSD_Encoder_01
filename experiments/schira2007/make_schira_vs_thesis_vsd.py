#!/usr/bin/env python3
"""Schira-only reconstruction vs thesis Fig. 13 VSD overlay (column 2).

Renders a catalog stimulus at an overridden visual-field position (e.g. letter L
at thesis (1.4, -0.7) instead of catalog (1, -0.9)), forward-maps ink through
Schira, and compares to the cropped thesis panel (black dashed model on VSD).

Manual trace (recommended for L): click two overlay arms on the thesis crop,
then align with **proportional** similarity — scale thesis to match Schira arm
lengths, rotate so vertical arm is left and horizontal arm is bottom.

Usage:
  scripts/py experiments/schira2007/make_schira_vs_thesis_vsd.py \\
      --pick-thesis-overlay --pick-only \\
      --stimulus letter_L_white_1 --pos-x 1.4 --pos-y -0.7 --thesis-panel e

  scripts/py experiments/schira2007/make_schira_vs_thesis_vsd.py \\
      --thesis-trace experiments/schira2007/phase_0_register/...yaml \\
      --manual-only --align-mode proportional \\
      --stimulus letter_L_white_1 --pos-x 1.4 --pos-y -0.7 --thesis-panel e
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from src.paths import project_root, resolve_data_path
from src.retinotopy.params import load_schira_set
from src.retinotopy.plotting import plot_schira_only
from src.retinotopy.warp import forward_ink_cloud_w
from src.stimuli.catalog import (
    load_full_encoder_catalog,
    stimulus_spec_from_mapping,
)
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.render import RenderConfig, render_stimulus

HERE = Path(__file__).resolve().parent
DEFAULT_SCHIRA = project_root() / "configs/schira/sets.yaml"
DEFAULT_STIMULI = project_root() / "configs/stimuli/default.yaml"
THESIS_PAGE = HERE / "_throwaway_thesis_fig10_pdf_pages" / "page_031.png"
OUT_DIR = HERE / "phase_0_schira"

# Fig. 13 panel → vertical center on page_031 (20/11/18 left VSD column).
THESIS_PANEL_Y: dict[str, float] = {
    "c": 0.30,
    "d": 0.33,
    "e": 0.40,
    "f": 0.35,
    "g": 0.55,
    "h": 0.50,
}


@dataclass(frozen=True)
class LArmGeometry:
    """L junction + tips of vertical (left) and horizontal (bottom) arms."""

    junction: tuple[float, float]
    vert_tip: tuple[float, float]
    horiz_tip: tuple[float, float]


@dataclass(frozen=True)
class Similarity2D:
    """Map thesis crop ``(col, row)`` → our ``(u, v)``."""

    origin_u: float
    origin_v: float
    scale: float
    rotation_deg: float
    flip_u: bool = False
    flip_v: bool = False

    def apply(self, col: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(col, dtype=np.float64)
        r = np.asarray(row, dtype=np.float64)
        sx = c.copy()
        sy = r.copy()
        if self.flip_u:
            sx = -sx
        if self.flip_v:
            sy = -sy
        theta = np.deg2rad(self.rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        du = self.scale * (cos_t * sx - sin_t * sy)
        dv = self.scale * (sin_t * sx + cos_t * sy)
        return self.origin_u + du, self.origin_v + dv


def _path_relative_to_repo(path: Path, repo: Path) -> str:
    resolved = path.resolve() if not path.is_absolute() else path
    try:
        return str(resolved.relative_to(repo.resolve()))
    except ValueError:
        return str(resolved)


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _render_config(stimuli_cfg: dict, *, canvas_size: int = 672) -> RenderConfig:
    base_canvas = int(stimuli_cfg.get("canvas_size", 224))
    quadrant_extent_deg = float(stimuli_cfg.get("quadrant_extent_deg", 6.0))
    base_ppd = stimuli_cfg.get("pixels_per_deg")
    if base_ppd is None:
        base_ppd = base_canvas / quadrant_extent_deg
    pixels_per_deg = float(base_ppd) * (canvas_size / base_canvas)
    return RenderConfig(
        canvas_size=canvas_size,
        pixels_per_deg=pixels_per_deg,
        quadrant_extent_deg=quadrant_extent_deg,
        background_gray=int(stimuli_cfg.get("background_gray", 128)),
        bar_length_deg=float(stimuli_cfg.get("bar_length_deg", 1.0)),
        bar_width_px=int(stimuli_cfg.get("bar_width_px", 1)),
        contour_width_px=int(stimuli_cfg.get("contour_width_px", 1)),
        assume_size_is_diameter=bool(stimuli_cfg.get("assume_size_is_diameter", True)),
        draw_fixation=bool(stimuli_cfg.get("draw_fixation", False)),
    )


def _crop_thesis_vsd(page: Image.Image, y_frac: float) -> np.ndarray:
    w, h = page.size
    y = int(y_frac * h)
    half = 80
    box = (int(0.38 * w), y - half, int(0.54 * w), y + half)
    arr = np.asarray(page.crop(box))
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float64) / 255.0
    return arr


def _extract_thesis_overlay_pixels(
    rgb: np.ndarray,
    *,
    dilate_iters: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Model overlay strokes from a thesis VSD crop (col, row in image px).

    Fig. 13 uses **dashed** and **solid** dark lines on the heatmap. We keep
    near-black neutral pixels and drop large blobs (activation patches).
    """
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.max() > 1.5:
        arr = arr / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    dark = np.min(arr[..., :3], axis=-1)
    bright = np.max(arr[..., :3], axis=-1)
    neutral = (
        (np.abs(r - g) < 0.10)
        & (np.abs(g - b) < 0.14)
        & (np.abs(r - b) < 0.14)
    )
    not_blue = b <= np.maximum(r, g) + 0.04
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    from scipy.ndimage import binary_dilation, binary_erosion, label, uniform_filter

    local_dark = uniform_filter(dark, size=11)

    # Dashed / dark strokes on green/blue field.
    mask = (bright <= 0.10) & (dark <= 0.12) & neutral & not_blue
    # Dashed strokes on yellow/orange bridge (not bright enough for mask_dark).
    stroke_mid = (
        (lum > 0.12)
        & (lum < 0.55)
        & (dark < 0.24)
        & neutral
        & not_blue
        & ((local_dark - dark) > 0.035)
    )
    mask = mask | stroke_mid
    # Solid strokes on bright activation blobs (anti-aliased dark on white/pink).
    # Magenta hotspots are not "neutral" gray — use top-hat on the dark channel.
    stroke_on_bright = (
        (lum > 0.40)
        & ((local_dark - dark) > 0.045)
        & (dark < 0.70)
    )
    mask = mask | stroke_on_bright

    h, w = mask.shape
    cy, cx = 0.47 * h, 0.50 * w
    radius = 0.44 * min(h, w)
    row_g = np.arange(h, dtype=np.float64)
    col_g = np.arange(w, dtype=np.float64)
    chamber = (row_g[:, None] - cy) ** 2 + (col_g[None, :] - cx) ** 2 <= radius ** 2
    margin = col_g[None, :] >= 0.10 * w
    mask &= chamber & margin

    if dilate_iters > 0:
        mask = binary_dilation(mask, iterations=dilate_iters)
    mask = binary_erosion(mask, iterations=1)
    if dilate_iters > 0:
        mask = binary_dilation(mask, iterations=max(1, dilate_iters - 1))

    # Drop large connected components (VSD blobs); keep thin overlay strokes.
    labeled, n_comp = label(mask)
    if n_comp > 0:
        keep = np.zeros_like(mask, dtype=bool)
        for cid in range(1, n_comp + 1):
            comp = labeled == cid
            area = int(comp.sum())
            if area > 600:
                continue
            rows_c, cols_c = np.where(comp)
            h = int(rows_c.max() - rows_c.min()) + 1
            w = int(cols_c.max() - cols_c.min()) + 1
            elong = max(h, w) / max(1, min(h, w))
            if area <= 400 or elong >= 2.0:
                keep |= comp
        mask = keep

    rows, cols = np.where(mask)
    return cols.astype(np.float64), rows.astype(np.float64)


def _use_interactive_backend() -> str:
    import matplotlib

    candidates = ("MacOSX", "QtAgg", "Qt5Agg", "TkAgg")
    errors: list[str] = []
    for name in candidates:
        try:
            matplotlib.use(name, force=True)
            matplotlib.backends.backend_registry.load_backend_module(name)
            print(f"Using matplotlib backend: {name}", flush=True)
            return name
        except Exception as exc:  # noqa: BLE001 — probe only
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    detail = "; ".join(errors) if errors else "no candidates tried"
    raise RuntimeError(
        "No interactive matplotlib backend for thesis overlay picking. "
        f"Tried: {detail}"
    )


def _rasterize_polylines(
    polylines: list[list[tuple[float, float]]],
    *,
    step_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(col, row)`` samples along piecewise-linear traces."""
    cols: list[float] = []
    rows: list[float] = []
    step = max(float(step_px), 0.25)
    for poly in polylines:
        if len(poly) < 2:
            if poly:
                cols.append(poly[0][0])
                rows.append(poly[0][1])
            continue
        for (c0, r0), (c1, r1) in zip(poly, poly[1:], strict=False):
            dx, dy = c1 - c0, r1 - r0
            dist = float(np.hypot(dx, dy))
            n = max(1, int(np.ceil(dist / step)))
            for t in np.linspace(0.0, 1.0, n + 1):
                cols.append(c0 + t * dx)
                rows.append(r0 + t * dy)
    if not cols:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return np.asarray(cols, dtype=np.float64), np.asarray(rows, dtype=np.float64)


def _split_l_polyline_at_corner(
    poly: list[tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    """Split one continuous L trace into vertical + horizontal arms."""
    if len(poly) < 3:
        return [poly]
    best_i = 1
    best_turn = -1.0
    for i in range(1, len(poly) - 1):
        v1 = (poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1])
        v2 = (poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
        a1 = float(np.arctan2(v1[1], v1[0]))
        a2 = float(np.arctan2(v2[1], v2[0]))
        da = abs((a2 - a1 + np.pi) % (2 * np.pi) - np.pi)
        if da > best_turn:
            best_turn = da
            best_i = i
    if best_turn < np.deg2rad(25):
        return [poly]
    return [poly[: best_i + 1], poly[best_i:]]


def pick_thesis_overlay_trace(
    thesis_rgb: np.ndarray,
    *,
    title: str,
    output_yaml: Path,
    expected_arms: int = 2,
) -> list[list[tuple[float, float]]]:
    """Click along thesis overlay strokes on the VSD crop.

    For letter L: trace the **left vertical arm** first, press ``n``, then the
    **bottom horizontal arm** (include the shared corner on both arms).

    Keys: ``n`` = new arm, ``z`` = undo last point, close window to save.
    """
    _use_interactive_backend()
    import matplotlib.pyplot as plt

    arm_colors = ("#ff7f0e", "#2ca02c", "#9467bd", "#8c564b")
    polylines: list[list[tuple[float, float]]] = [[]]
    artists: list = []

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    fig.canvas.manager.set_window_title(
        "Trace thesis overlay: arm1 vertical · n · arm2 horizontal"
    )
    ax.imshow(thesis_rgb)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.text(
        0.5,
        0.01,
        (
            "Letter L: click along LEFT vertical arm (top → corner), "
            "press n, click BOTTOM horizontal arm (corner → right). "
            "z = undo. Close window to save."
        ),
        ha="center",
        fontsize=9,
        color="0.25",
    )

    def _arm_color(i: int) -> str:
        return arm_colors[i % len(arm_colors)]

    def _redraw() -> None:
        while artists:
            artists.pop().remove()
        for i, poly in enumerate(polylines):
            if not poly:
                continue
            color = _arm_color(i)
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            artists.append(
                ax.scatter(xs, ys, s=28, c=color, zorder=5, edgecolors="white", linewidths=0.4)
            )
            if len(poly) >= 2:
                artists.append(
                    ax.plot(xs, ys, color=color, lw=2.2, alpha=0.95, zorder=4)[0]
                )
            artists.append(
                ax.text(
                    xs[0],
                    ys[0] - 4,
                    f"arm {i + 1}",
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                    zorder=6,
                )
            )
        fig.canvas.draw_idle()

    def onclick(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        polylines[-1].append((float(event.xdata), float(event.ydata)))
        _redraw()

    def onkey(event) -> None:
        if event.key == "n":
            if len(polylines) >= expected_arms:
                print(f"Already have {expected_arms} arms; close window to save.", flush=True)
                return
            if polylines[-1]:
                polylines.append([])
            _redraw()
        elif event.key == "z":
            if polylines[-1]:
                polylines[-1].pop()
            elif len(polylines) > 1:
                polylines.pop()
            _redraw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    polylines = [p for p in polylines if len(p) >= 2]
    if len(polylines) == 1:
        split = _split_l_polyline_at_corner(polylines[0])
        if len(split) == 2:
            polylines = split
            print("Auto-split single L trace into 2 arms at corner", flush=True)
    if len(polylines) != expected_arms:
        raise RuntimeError(
            f"Need exactly {expected_arms} arms: trace vertical arm, press n, "
            f"trace horizontal arm (or one continuous L). Got {len(polylines)}."
        )

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "polylines_col_row": [[list(pt) for pt in poly] for poly in polylines],
        "note": (
            "Manual trace of thesis Fig. 13 overlay on VSD crop (col, row). "
            "Arm 1 = left vertical; arm 2 = bottom horizontal."
        ),
    }
    output_yaml.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Saved thesis overlay trace: {output_yaml} ({len(polylines)} arms)")
    return polylines


def _load_thesis_trace(path: Path) -> list[list[tuple[float, float]]]:
    data = _load_yaml(path)
    raw = data.get("polylines_col_row") or data.get("polylines") or []
    polylines: list[list[tuple[float, float]]] = []
    for poly in raw:
        pts = [(float(p[0]), float(p[1])) for p in poly]
        if pts:
            polylines.append(pts)
    return polylines


def _merge_thesis_overlay_points(
    auto_col: np.ndarray,
    auto_row: np.ndarray,
    manual_col: np.ndarray,
    manual_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if manual_col.size == 0:
        return auto_col, auto_row
    if auto_col.size == 0:
        return manual_col, manual_row
    return (
        np.concatenate([auto_col, manual_col]),
        np.concatenate([auto_row, manual_row]),
    )


def _angle_deg(dx: float, dy: float) -> float:
    return float(np.rad2deg(np.arctan2(dy, dx)))


def _dist2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _geometry_from_polylines(
    polylines: list[list[tuple[float, float]]],
) -> LArmGeometry:
    """Junction + vertical-left / horizontal-bottom tips from two traced arms."""
    if len(polylines) < 2:
        raise ValueError(f"Need 2 polylines for L geometry, got {len(polylines)}")

    p0, p1 = polylines[0], polylines[1]
    end_pairs = [
        (p0[0], p1[0], p0[-1], p1[-1]),
        (p0[0], p1[-1], p0[-1], p1[0]),
        (p0[-1], p1[0], p0[0], p1[-1]),
        (p0[-1], p1[-1], p0[0], p1[0]),
    ]
    best = min(end_pairs, key=lambda t: _dist2d(t[0], t[1]))
    j0, j1, tip0, tip1 = best
    junction = (
        (j0[0] + j1[0]) / 2.0,
        (j0[1] + j1[1]) / 2.0,
    )

    def _arm_span(tip: tuple[float, float], junc: tuple[float, float]) -> tuple[float, float]:
        dc = abs(tip[0] - junc[0])
        dr = abs(tip[1] - junc[1])
        return dr, dc

    s0 = _arm_span(tip0, junction)
    s1 = _arm_span(tip1, junction)
    # Image row increases downward → vertical arm has larger |Δrow|.
    if s0[0] >= s1[0]:
        vert_tip, horiz_tip = tip0, tip1
    else:
        vert_tip, horiz_tip = tip1, tip0

    # Horizontal arm should extend right from the junction.
    if horiz_tip[0] < junction[0]:
        vert_tip, horiz_tip = horiz_tip, vert_tip
    if vert_tip[1] > junction[1]:
        # vertical tip should be above junction (smaller row index).
        vert_tip = (vert_tip[0], junction[1] - abs(vert_tip[1] - junction[1]))

    return LArmGeometry(junction=junction, vert_tip=vert_tip, horiz_tip=horiz_tip)


def _schira_l_geometry_from_points(
    u: np.ndarray,
    v: np.ndarray,
) -> LArmGeometry:
    """Infer L junction and arm tips from Schira ink cloud."""
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    if u.size < 8:
        raise ValueError("Need at least 8 Schira points for L geometry")

    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    u_span = max(u_max - u_min, 1e-9)
    v_span = max(v_max - v_min, 1e-9)

    # Bottom-left corner of the L in (u, v): left + bottom (most negative v).
    corner = np.array([u_min, v_min])
    ji = int(np.argmin((u - corner[0]) ** 2 + (v - corner[1]) ** 2))
    ju, jv = float(u[ji]), float(v[ji])

    dist = np.hypot(u - ju, v - jv)
    active = dist > 1e-9 * max(u_span, v_span)

    left_mask = active & (u <= ju + 0.08 * u_span)
    if left_mask.any():
        vi = int(np.nanargmax(np.where(left_mask, v, np.nan)))
        vert_tip = (float(u[vi]), float(v[vi]))
    else:
        vi = int(np.argmax(np.where(active, v, -np.inf)))
        vert_tip = (float(u[vi]), float(v[vi]))

    bottom_mask = active & (v <= jv + 0.08 * v_span) & (u >= ju - 0.02 * u_span)
    if bottom_mask.any():
        hi = int(np.nanargmax(np.where(bottom_mask, u, np.nan)))
        horiz_tip = (float(u[hi]), float(v[hi]))
    else:
        hi = int(np.argmax(np.where(active, u, -np.inf)))
        horiz_tip = (float(u[hi]), float(v[hi]))

    return LArmGeometry(junction=(ju, jv), vert_tip=vert_tip, horiz_tip=horiz_tip)


def _arm_masks_from_geometry(
    x: np.ndarray,
    y: np.ndarray,
    geom: LArmGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Left-vertical and bottom-horizontal ink masks for L-like shapes."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ju, jv = geom.junction
    x_span = max(float(x.max() - x.min()), 1e-9)
    y_span = max(float(y.max() - y.min()), 1e-9)
    dist = np.hypot(x - ju, y - jv)
    active = dist > 1e-9 * max(x_span, y_span)
    vert_mask = active & (x <= ju + 0.12 * x_span)
    horiz_mask = active & (y <= jv + 0.12 * y_span) & (x >= ju - 0.05 * x_span)
    return vert_mask, horiz_mask


def _ordered_arm_polyline(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    junction: tuple[float, float],
    tip: tuple[float, float],
    *,
    n_bins: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Median-sampled polyline along an arm (keeps curvature in the ink cloud)."""
    pts = np.column_stack(
        [np.asarray(x, dtype=np.float64)[mask], np.asarray(y, dtype=np.float64)[mask]]
    )
    j = np.array(junction, dtype=np.float64)
    tip_arr = np.array(tip, dtype=np.float64)
    direction = tip_arr - j
    dn = float(np.linalg.norm(direction))
    if pts.shape[0] < 2 or dn < 1e-12:
        return np.array([junction[0]]), np.array([junction[1]])
    direction = direction / dn
    proj = (pts - j) @ direction
    valid = proj >= -0.04 * dn
    pts = pts[valid]
    proj = proj[valid]
    if pts.shape[0] < 2:
        return np.array([junction[0]]), np.array([junction[1]])
    nb = int(min(n_bins, max(8, pts.shape[0] // 4)))
    edges = np.linspace(0.0, float(proj.max()), nb + 1)
    xs: list[float] = [float(junction[0])]
    ys: list[float] = [float(junction[1])]
    for i in range(nb):
        in_bin = (proj >= edges[i]) & (proj < edges[i + 1])
        if i == nb - 1:
            in_bin = proj >= edges[i]
        if not np.any(in_bin):
            continue
        xs.append(float(np.median(pts[in_bin, 0])))
        ys.append(float(np.median(pts[in_bin, 1])))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _arm_polylines_from_cloud(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], LArmGeometry]:
    geom = _schira_l_geometry_from_points(x, y)
    vert_mask, horiz_mask = _arm_masks_from_geometry(x, y, geom)
    polylines = [
        _ordered_arm_polyline(x, y, vert_mask, geom.junction, geom.vert_tip),
        _ordered_arm_polyline(x, y, horiz_mask, geom.junction, geom.horiz_tip),
    ]
    return polylines, geom


def _flip_xy(
    x: float,
    y: float,
    *,
    flip_u: bool,
    flip_v: bool,
) -> tuple[float, float]:
    return (-x if flip_u else x, -y if flip_v else y)


def _fit_similarity_proportional_l(
    thesis: LArmGeometry,
    schira: LArmGeometry,
    *,
    scale_mode: str = "mean_arms",
) -> tuple[Similarity2D, dict]:
    """Uniform scale + rotation so thesis L matches Schira L size and pose."""
    t_j = np.array(thesis.junction, dtype=np.float64)
    t_v = np.array(thesis.vert_tip, dtype=np.float64) - t_j
    t_h = np.array(thesis.horiz_tip, dtype=np.float64) - t_j
    s_j = np.array(schira.junction, dtype=np.float64)
    s_v = np.array(schira.vert_tip, dtype=np.float64) - s_j
    s_h = np.array(schira.horiz_tip, dtype=np.float64) - s_j

    t_v_len = float(np.linalg.norm(t_v))
    t_h_len = float(np.linalg.norm(t_h))
    s_v_len = float(np.linalg.norm(s_v))
    s_h_len = float(np.linalg.norm(s_h))
    if t_v_len < 1e-9 or t_h_len < 1e-9:
        raise ValueError("Thesis arm lengths are too small for proportional fit")

    flip_opts = [(False, False), (True, False), (False, True), (True, True)]
    best_sim: Similarity2D | None = None
    best_meta: dict | None = None
    best_score = float("inf")

    for flip_u, flip_v in flip_opts:
        fj = np.array(_flip_xy(t_j[0], t_j[1], flip_u=flip_u, flip_v=flip_v))
        fv = np.array(_flip_xy(t_v[0], t_v[1], flip_u=flip_u, flip_v=flip_v))
        fh = np.array(_flip_xy(t_h[0], t_h[1], flip_u=flip_u, flip_v=flip_v))
        fv_len = float(np.linalg.norm(fv))
        fh_len = float(np.linalg.norm(fh))
        if fv_len < 1e-9:
            continue

        scale_v = s_v_len / fv_len
        scale_h = s_h_len / fh_len if fh_len > 1e-9 else scale_v
        if scale_mode == "vertical_arm":
            scale = scale_v
        elif scale_mode == "horizontal_arm":
            scale = scale_h
        else:
            scale = (scale_v + scale_h) / 2.0

        rot_deg = _angle_deg(s_v[0], s_v[1]) - _angle_deg(fv[0], fv[1])
        theta = np.deg2rad(rot_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_fj = np.array(
            [cos_t * fj[0] - sin_t * fj[1], sin_t * fj[0] + cos_t * fj[1]]
        )
        origin = s_j - scale * rot_fj

        sim = Similarity2D(
            origin_u=float(origin[0]),
            origin_v=float(origin[1]),
            scale=float(scale),
            rotation_deg=float(rot_deg),
            flip_u=flip_u,
            flip_v=flip_v,
        )

        pred_u, pred_v = sim.apply(
            np.array([thesis.horiz_tip[0]]),
            np.array([thesis.horiz_tip[1]]),
        )
        pred_h_vec = np.array([float(pred_u[0]) - s_j[0], float(pred_v[0]) - s_j[1]])
        ang_h_thesis = _angle_deg(float(pred_h_vec[0]), float(pred_h_vec[1]))
        ang_h_schira = _angle_deg(s_h[0], s_h[1])
        horiz_ang_diff = abs((ang_h_thesis - ang_h_schira + 180) % 360 - 180)
        score = horiz_ang_diff

        if score < best_score:
            best_score = score
            best_sim = sim
            best_meta = {
                "scale_vertical_arm": scale_v,
                "scale_horizontal_arm": scale_h,
                "scale_applied": scale,
                "thesis_vert_len_px": t_v_len,
                "thesis_horiz_len_px": t_h_len,
                "schira_vert_len": s_v_len,
                "schira_horiz_len": s_h_len,
                "schira_vert_angle_deg": _angle_deg(s_v[0], s_v[1]),
                "schira_horiz_angle_deg": ang_h_schira,
                "horiz_angle_diff_deg": horiz_ang_diff,
                "flip_u": flip_u,
                "flip_v": flip_v,
            }

    if best_sim is None or best_meta is None:
        raise RuntimeError("Proportional L fit failed for all flip combinations")
    return best_sim, best_meta


def _apply_sim_point(
    sim: Similarity2D, col: float, row: float,
) -> tuple[float, float]:
    u, v = sim.apply(np.array([col]), np.array([row]))
    return float(u[0]), float(v[0])


def _tip_align_rmsd(
    sim: Similarity2D,
    thesis_geom: LArmGeometry,
    schira_geom: LArmGeometry,
) -> float:
    dists: list[float] = []
    for t_pt, s_pt in (
        (thesis_geom.junction, schira_geom.junction),
        (thesis_geom.vert_tip, schira_geom.vert_tip),
        (thesis_geom.horiz_tip, schira_geom.horiz_tip),
    ):
        pu, pv = _apply_sim_point(sim, t_pt[0], t_pt[1])
        dists.append(np.hypot(pu - s_pt[0], pv - s_pt[1]))
    return float(np.sqrt(np.mean(np.square(dists))))


def _transform_polylines(
    sim: Similarity2D,
    polylines: list[list[tuple[float, float]]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for poly in polylines:
        cols = np.asarray([p[0] for p in poly], dtype=np.float64)
        rows = np.asarray([p[1] for p in poly], dtype=np.float64)
        u, v = sim.apply(cols, rows)
        out.append((u, v))
    return out


def _subsample_xy(
    x: np.ndarray, y: np.ndarray, *, max_pts: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.size)
    if n <= max_pts:
        return x, y
    idx = rng.choice(n, size=max_pts, replace=False)
    return x[idx], y[idx]


def _fit_similarity_thesis_to_schira(
    thesis_col: np.ndarray,
    thesis_row: np.ndarray,
    schira_u: np.ndarray,
    schira_v: np.ndarray,
    *,
    try_flips: bool = True,
    icp_iters: int = 6,
    rng: np.random.Generator,
) -> tuple[Similarity2D, float]:
    """Register thesis overlay pixels onto Schira ``(u,v)`` via similarity + ICP."""
    src_c = np.asarray(thesis_col, dtype=np.float64).ravel()
    src_r = np.asarray(thesis_row, dtype=np.float64).ravel()
    dst_u = np.asarray(schira_u, dtype=np.float64).ravel()
    dst_v = np.asarray(schira_v, dtype=np.float64).ravel()
    if src_c.size < 8 or dst_u.size < 8:
        raise ValueError("Need at least 8 points in each cloud for similarity fit")

    src_c, src_r = _subsample_xy(src_c, src_r, max_pts=2500, rng=rng)
    dst_u, dst_v = _subsample_xy(dst_u, dst_v, max_pts=2500, rng=rng)

    flip_opts = (
        [(False, False), (True, False), (False, True), (True, True)]
        if try_flips
        else [(False, False)]
    )

    best_sim: Similarity2D | None = None
    best_rmsd = float("inf")

    for flip_u, flip_v in flip_opts:
        s_u = -1.0 if flip_u else 1.0
        s_v = -1.0 if flip_v else 1.0
        sx = s_u * src_c
        sy = s_v * src_r

        # Initial similarity from centered clouds (scale + rotation via SVD).
        sx0 = sx - float(np.mean(sx))
        sy0 = sy - float(np.mean(sy))
        du0 = dst_u - float(np.mean(dst_u))
        dv0 = dst_v - float(np.mean(dst_v))
        rms_s = float(np.sqrt(np.mean(sx0 * sx0 + sy0 * sy0)))
        rms_d = float(np.sqrt(np.mean(du0 * du0 + dv0 * dv0)))
        if rms_s < 1e-9:
            continue
        sx_n = sx0 / rms_s
        sy_n = sy0 / rms_s
        du_n = du0 / rms_d
        dv_n = dv0 / rms_d
        n_pts = min(sx_n.size, du_n.size, 800)
        idx_s = rng.choice(sx_n.size, size=n_pts, replace=False)
        idx_d = rng.choice(du_n.size, size=n_pts, replace=False)
        xs = np.column_stack([sx_n[idx_s], sy_n[idx_s]])
        xd = np.column_stack([du_n[idx_d], dv_n[idx_d]])
        cov = xs.T @ xd / n_pts
        rot_m, _ = np.linalg.qr(cov)
        if np.linalg.det(rot_m) < 0:
            rot_m[:, 1] *= -1.0
        rot_deg = float(np.rad2deg(np.arctan2(rot_m[1, 0], rot_m[0, 0])))
        scale = rms_d / rms_s
        cu_s, cv_s = float(np.mean(sx)), float(np.mean(sy))
        cu_d, cv_d = float(np.mean(dst_u)), float(np.mean(dst_v))
        theta = np.deg2rad(rot_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        ox = cu_d - scale * (cos_t * cu_s - sin_t * cv_s)
        oy = cv_d - scale * (sin_t * cu_s + cos_t * cv_s)
        sim = Similarity2D(
            origin_u=ox,
            origin_v=oy,
            scale=scale,
            rotation_deg=rot_deg,
            flip_u=flip_u,
            flip_v=flip_v,
        )

        for _ in range(icp_iters):
            pred_u, pred_v = sim.apply(src_c, src_r)
            nn_idx = np.argmin(
                (pred_u[:, None] - dst_u[None, :]) ** 2
                + (pred_v[:, None] - dst_v[None, :]) ** 2,
                axis=1,
            )
            tgt_u = dst_u[nn_idx]
            tgt_v = dst_v[nn_idx]

            n = sx.size
            M = np.zeros((2 * n, 4), dtype=np.float64)
            y = np.zeros(2 * n, dtype=np.float64)
            for i in range(n):
                # u = ox + A*sx - B*sy; v = oy + B*sx + A*sy  (A=scale*cos, B=scale*sin)
                M[2 * i] = [1.0, 0.0, sx[i], -sy[i]]
                y[2 * i] = tgt_u[i]
                M[2 * i + 1] = [0.0, 1.0, sy[i], sx[i]]
                y[2 * i + 1] = tgt_v[i]
            sol, _, rank, _ = np.linalg.lstsq(M, y, rcond=None)
            if rank < 4:
                break
            ox, oy, a_cos, b_sin = (float(x) for x in sol)
            ppu = float(np.hypot(a_cos, b_sin))
            if ppu < 1e-12:
                break
            rot = float(np.rad2deg(np.arctan2(b_sin, a_cos)))
            sim = Similarity2D(
                origin_u=ox,
                origin_v=oy,
                scale=ppu,
                rotation_deg=rot,
                flip_u=flip_u,
                flip_v=flip_v,
            )

        pred_u, pred_v = sim.apply(src_c, src_r)
        nn_idx = np.argmin(
            (pred_u[:, None] - dst_u[None, :]) ** 2
            + (pred_v[:, None] - dst_v[None, :]) ** 2,
            axis=1,
        )
        err = np.hypot(pred_u - dst_u[nn_idx], pred_v - dst_v[nn_idx])
        rmsd = float(np.sqrt(np.mean(err**2)))
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_sim = sim

    if best_sim is None:
        raise RuntimeError("Similarity fit failed for all flip combinations")
    return best_sim, best_rmsd


def _plot_overlay_on_schira(
    *,
    schira_u: np.ndarray,
    schira_v: np.ndarray,
    thesis_u: np.ndarray,
    thesis_v: np.ndarray,
    output_path: Path,
    title: str,
    sim: Similarity2D,
    align_rmsd: float,
    schira_arm_polylines: list[tuple[np.ndarray, np.ndarray]] | None = None,
    thesis_arm_polylines: list[tuple[np.ndarray, np.ndarray]] | None = None,
    align_mode: str = "icp",
    proportional_meta: dict | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.8), layout="constrained")
    ax.set_facecolor("white")
    ax.scatter(
        schira_u,
        schira_v,
        s=3,
        c="#1f77b4",
        marker="s",
        linewidths=0,
        alpha=0.22,
        label="Ours · Schira ink",
        zorder=1,
    )
    if schira_arm_polylines:
        for i, (pu, pv) in enumerate(schira_arm_polylines):
            ax.plot(
                pu,
                pv,
                color="#1f77b4",
                lw=2.4,
                ls="-",
                alpha=0.95,
                label="Ours · Schira stroke" if i == 0 else None,
                zorder=3,
            )

    if thesis_arm_polylines:
        for i, (pu, pv) in enumerate(thesis_arm_polylines):
            ax.plot(
                pu,
                pv,
                color="#d62728",
                lw=2.4,
                ls="-",
                alpha=0.95,
                label="Thesis overlay (aligned)" if i == 0 else None,
                zorder=4,
            )
    elif thesis_u.size:
        ax.scatter(
            thesis_u,
            thesis_v,
            s=4,
            c="#d62728",
            marker="o",
            linewidths=0,
            alpha=0.65,
            label="Thesis overlay (aligned)",
            zorder=4,
        )

    pad = 0.12 * max(
        float(np.max(schira_u) - np.min(schira_u)),
        float(np.max(schira_v) - np.min(schira_v)),
        0.1,
    )
    ax.set_xlim(float(np.min(schira_u)) - pad, float(np.max(schira_u)) + pad)
    ax.set_ylim(float(np.min(schira_v)) - pad, float(np.max(schira_v)) + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("u (Cartesian cortical)")
    ax.set_ylabel("v (Cartesian cortical)")
    ax.axhline(0.0, color="0.75", lw=0.5, ls="--")
    ax.axvline(0.0, color="0.75", lw=0.5, ls="--")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=10)
    if align_mode == "proportional" and proportional_meta is not None:
        footer = (
            f"Proportional L align · scale={sim.scale:.4g} "
            f"rot={sim.rotation_deg:.1f}° · "
            f"vert/horiz scale={proportional_meta['scale_vertical_arm']:.3g}/"
            f"{proportional_meta['scale_horizontal_arm']:.3g} · "
            f"horiz Δangle={proportional_meta['horiz_angle_diff_deg']:.2f}° · "
            f"tip RMSD={align_rmsd:.4g}"
        )
    else:
        footer = (
            f"Similarity align thesis→Schira: scale={sim.scale:.4g} "
            f"rot={sim.rotation_deg:.1f}° flip_u={sim.flip_u} flip_v={sim.flip_v} "
            f"NN-RMSD={align_rmsd:.4g} (Schira units)"
        )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=7, color="0.35")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", dest="set_name", default="201118")
    p.add_argument("--schira-config", type=Path, default=DEFAULT_SCHIRA)
    p.add_argument("--stimulus", default="letter_L_white_1")
    p.add_argument("--pos-x", type=float, default=1.4)
    p.add_argument("--pos-y", type=float, default=-0.7)
    p.add_argument(
        "--thesis-panel",
        default="e",
        help="Fig. 13 panel letter (c=D … h=horizontal bar)",
    )
    p.add_argument("--thesis-page", type=Path, default=THESIS_PAGE)
    p.add_argument("--canvas-size", type=int, default=672)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--pick-thesis-overlay",
        action="store_true",
        help="GUI: click along thesis overlay arms on the VSD crop; saves trace YAML",
    )
    p.add_argument(
        "--thesis-trace",
        type=Path,
        default=None,
        help="YAML with manual polylines_col_row on thesis crop",
    )
    p.add_argument(
        "--pick-only",
        action="store_true",
        help="With --pick-thesis-overlay: save trace YAML and exit",
    )
    p.add_argument(
        "--manual-only",
        action="store_true",
        help="Use manual trace only (skip auto pixel extraction)",
    )
    p.add_argument(
        "--align-mode",
        choices=("proportional", "icp"),
        default="icp",
        help="icp = pixel-cloud similarity (default); proportional = manual L arm lengths",
    )
    args = p.parse_args(argv)

    panel = str(args.thesis_panel).strip().lower()
    if panel not in THESIS_PANEL_Y:
        raise SystemExit(f"Unknown panel {panel!r}; choose from {sorted(THESIS_PANEL_Y)}")

    repo = project_root()
    set_id, params, _affine, session_raw = load_schira_set(
        args.schira_config, set_name=args.set_name
    )
    date_prefix = str(session_raw.get("date_prefix") or set_id)
    tag = f"{date_prefix}__{args.stimulus}__thesis_pos_{args.pos_x:g}_{args.pos_y:g}"

    thesis_path = args.thesis_page if args.thesis_page.is_absolute() else repo / args.thesis_page
    if not thesis_path.exists():
        raise FileNotFoundError(f"Missing thesis page: {thesis_path}")
    thesis_vsd = _crop_thesis_vsd(Image.open(thesis_path), THESIS_PANEL_Y[panel])

    register_dir = HERE / "phase_0_register"
    register_dir.mkdir(parents=True, exist_ok=True)
    trace_path = register_dir / f"thesis_overlay_trace__{tag}__panel_{panel}.yaml"

    if args.pick_thesis_overlay:
        pick_thesis_overlay_trace(
            thesis_vsd,
            title=f"Thesis panel {panel} · trace black overlay on VSD crop",
            output_yaml=trace_path,
            expected_arms=2,
        )
        args.thesis_trace = trace_path
        if args.pick_only:
            print(f"Pick-only: saved {trace_path}")
            return 0

    stimuli_cfg = _load_yaml(DEFAULT_STIMULI)
    render_cfg = _render_config(stimuli_cfg, canvas_size=args.canvas_size)

    cfg = _load_yaml(project_root() / "configs/default.yaml")
    encoder_root = resolve_data_path(cfg["paths"]["encoder_data_root"], repo)
    catalog = attach_stimulus_ids(
        load_full_encoder_catalog(
            encoder_root,
            monkey=str(cfg["monkey"]),
            bar_length_deg=render_cfg.bar_length_deg,
        )
    )
    hit = catalog[catalog["stimulus_id"].astype(str) == args.stimulus]
    if hit.empty:
        raise SystemExit(f"No catalog row for {args.stimulus!r}")
    base_spec = stimulus_spec_from_mapping(hit.iloc[0].to_dict())
    spec = replace(
        base_spec,
        pos_x_deg=float(args.pos_x),
        pos_y_deg=float(args.pos_y),
    )

    stimulus_rgb = render_stimulus(spec, render_cfg)
    fu, fv, _w = forward_ink_cloud_w(
        stimulus_rgb, params=params, render_cfg=render_cfg
    )
    if fu.size == 0:
        raise RuntimeError("No ink pixels after Schira forward map")

    out_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    register_dir = HERE / "phase_0_register"
    register_dir.mkdir(parents=True, exist_ok=True)

    schira_path = out_dir / f"{tag}__schira_only.png"
    param_note = (
        f"set={set_id} · pos=({spec.pos_x_deg:g}, {spec.pos_y_deg:g}) · "
        f"a={params.a:g} α={params.alpha:g} k={params.k:g} "
        f"sech_amp={params.sech_amp:g}"
    )
    plot_schira_only(
        fu,
        fv,
        output_path=schira_path,
        title=f"{args.stimulus} · Schira (ink only) → (u, v)",
        param_note=param_note,
    )

    manual_col = manual_row = np.array([], dtype=np.float64)
    manual_polylines: list[list[tuple[float, float]]] = []
    if args.thesis_trace is not None:
        trace_file = args.thesis_trace
        if not trace_file.is_absolute():
            trace_file = (repo / trace_file).resolve()
        manual_polylines = _load_thesis_trace(trace_file)
        manual_col, manual_row = _rasterize_polylines(manual_polylines, step_px=1.0)

    compare_path = register_dir / f"schira_vs_thesis_vsd__{tag}__panel_{panel}.png"
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), layout="constrained")

    axes[0].imshow(thesis_vsd)
    arm_colors = ("#ff7f0e", "#2ca02c")
    for i, poly in enumerate(manual_polylines):
        if len(poly) >= 2:
            axes[0].plot(
                [p[0] for p in poly],
                [p[1] for p in poly],
                color=arm_colors[i % len(arm_colors)],
                lw=2.0,
                alpha=0.95,
            )
    axes[0].set_title(
        f"Thesis Fig. 13 panel {panel} · VSD + model overlay\n"
        "(black dashed reconstruction on VSDI)",
        fontsize=10,
    )
    axes[0].axis("off")

    axes[1].scatter(fu, fv, s=4, c="black", marker="s", linewidths=0)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("u (Cartesian cortical)")
    axes[1].set_ylabel("v (Cartesian cortical)")
    axes[1].set_title(
        f"Ours · Schira only · {args.stimulus}\n"
        f"pos=({spec.pos_x_deg:g}, {spec.pos_y_deg:g}) deg",
        fontsize=10,
    )
    axes[1].axhline(0.0, color="0.75", lw=0.5, ls="--")
    axes[1].axvline(0.0, color="0.75", lw=0.5, ls="--")
    pad = 0.12 * max(float(fu.max() - fu.min()), float(fv.max() - fv.min()), 0.1)
    axes[1].set_xlim(float(fu.min()) - pad, float(fu.max()) + pad)
    axes[1].set_ylim(float(fv.min()) - pad, float(fv.max()) + pad)

    fig.suptitle(
        f"Schira reconstruction vs thesis VSD overlay · set {set_id} · panel {panel}",
        fontsize=11,
    )
    fig.savefig(compare_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(42)
    auto_col = auto_row = np.array([], dtype=np.float64)
    if not args.manual_only:
        auto_col, auto_row = _extract_thesis_overlay_pixels(thesis_vsd)
    th_col, th_row = _merge_thesis_overlay_points(
        auto_col, auto_row, manual_col, manual_row
    )
    if th_col.size < 8:
        raise RuntimeError(
            f"Too few thesis overlay pixels ({th_col.size}); "
            "use --pick-thesis-overlay or --thesis-trace"
        )

    schira_arm_polylines, schira_geom = _arm_polylines_from_cloud(fu, fv)
    align_mode = str(args.align_mode)
    proportional_meta: dict | None = None
    thesis_arm_polylines: list[tuple[np.ndarray, np.ndarray]] | None = None

    if align_mode == "proportional" and len(manual_polylines) >= 2:
        thesis_geom = _geometry_from_polylines(manual_polylines[:2])
        sim, proportional_meta = _fit_similarity_proportional_l(
            thesis_geom, schira_geom, scale_mode="mean_arms"
        )
        thesis_arm_polylines = _transform_polylines(sim, manual_polylines[:2])
        align_rmsd = _tip_align_rmsd(sim, thesis_geom, schira_geom)
    elif align_mode == "proportional":
        print(
            "Warning: proportional align needs 2 manual arms; falling back to ICP",
            flush=True,
        )
        align_mode = "icp"
        sim, align_rmsd = _fit_similarity_thesis_to_schira(
            th_col, th_row, fu, fv, rng=rng
        )
    else:
        sim, align_rmsd = _fit_similarity_thesis_to_schira(
            th_col, th_row, fu, fv, rng=rng
        )

    if thesis_arm_polylines is None and auto_col.size >= 8:
        try:
            t_polys, _ = _arm_polylines_from_cloud(auto_col, auto_row)
            thesis_arm_polylines = [
                (np.asarray(u), np.asarray(v))
                for u, v in (sim.apply(pu, pv) for pu, pv in t_polys)
            ]
        except (ValueError, RuntimeError):
            thesis_arm_polylines = None

    th_u, th_v = sim.apply(th_col, th_row)

    overlay_path = register_dir / (
        f"schira_vs_thesis_overlay_aligned__{tag}__panel_{panel}.png"
    )
    extract_debug_path = register_dir / (
        f"thesis_overlay_extract__{tag}__panel_{panel}.png"
    )
    fig_ex, ax_ex = plt.subplots(figsize=(4.2, 4.2), layout="constrained")
    ax_ex.imshow(thesis_vsd)
    if auto_col.size:
        ax_ex.scatter(auto_col, auto_row, s=3, c="lime", linewidths=0, label="auto")
    if manual_col.size:
        ax_ex.scatter(
            manual_col, manual_row, s=8, c="orange", linewidths=0, label="manual"
        )
    for poly in manual_polylines:
        if len(poly) >= 2:
            ax_ex.plot(
                [p[0] for p in poly],
                [p[1] for p in poly],
                color="orange",
                lw=1.0,
                alpha=0.9,
            )
    ax_ex.legend(loc="upper right", fontsize=7)
    ax_ex.set_title(
        f"Thesis overlay extract ({th_col.size} px total)\n"
        f"auto={auto_col.size} manual={manual_col.size}",
        fontsize=9,
    )
    ax_ex.axis("off")
    fig_ex.savefig(extract_debug_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ex)

    _plot_overlay_on_schira(
        schira_u=fu,
        schira_v=fv,
        thesis_u=th_u,
        thesis_v=th_v,
        output_path=overlay_path,
        title=(
            f"Schira (blue) vs thesis overlay (red) · panel {panel}\n"
            f"{args.stimulus} pos=({spec.pos_x_deg:g}, {spec.pos_y_deg:g})"
        ),
        sim=sim,
        align_rmsd=align_rmsd,
        schira_arm_polylines=schira_arm_polylines,
        thesis_arm_polylines=thesis_arm_polylines,
        align_mode=align_mode,
        proportional_meta=proportional_meta,
    )

    meta = {
        "set_name": set_id,
        "stimulus_id": args.stimulus,
        "pos_x_deg": spec.pos_x_deg,
        "pos_y_deg": spec.pos_y_deg,
        "catalog_pos": [base_spec.pos_x_deg, base_spec.pos_y_deg],
        "thesis_panel": panel,
        "schira_only": str(schira_path.relative_to(repo)),
        "comparison": str(compare_path.relative_to(repo)),
        "overlay_aligned": str(overlay_path.relative_to(repo)),
        "thesis_extract_debug": str(extract_debug_path.relative_to(repo)),
        "thesis_overlay_pixels": int(th_col.size),
        "thesis_overlay_pixels_auto": int(auto_col.size),
        "thesis_overlay_pixels_manual": int(manual_col.size),
        "thesis_trace": (
            _path_relative_to_repo(args.thesis_trace, repo)
            if args.thesis_trace is not None
            else None
        ),
        "align_mode": align_mode,
        "proportional_l_align": proportional_meta,
        "schira_l_geometry": {
            "junction": list(schira_geom.junction),
            "vert_tip": list(schira_geom.vert_tip),
            "horiz_tip": list(schira_geom.horiz_tip),
        },
        "similarity_align": {
            "scale": sim.scale,
            "rotation_deg": sim.rotation_deg,
            "flip_u": sim.flip_u,
            "flip_v": sim.flip_v,
            "origin_uv": [sim.origin_u, sim.origin_v],
            "nn_rmsd_schira_units": align_rmsd,
        },
        "thesis_page": str(thesis_path.relative_to(repo)),
        "n_ink": int(fu.size),
        "u_span": [float(fu.min()), float(fu.max())],
        "v_span": [float(fv.min()), float(fv.max())],
    }
    meta_path = compare_path.with_suffix(".yaml")
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    print(f"Wrote {schira_path}")
    print(f"Wrote {compare_path}")
    print(f"Wrote {overlay_path}")
    print(f"Wrote {extract_debug_path}")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
