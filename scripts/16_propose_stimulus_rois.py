#!/usr/bin/env python3
"""Propose stimulus-level VSD ROIs for interactive review (LOO prep).

Builds a stimulus inventory, then for **every** stimulus identity:

1. Average VSD maps across trials over frames ``[start_frame, end_frame)``
   (default ``[35, 42)``) — trial-weighted across all sessions.
2. Form residual ``stim_mean − global_common`` where ``global_common`` is the
   **grand mean of all trials in all sessions** (all non-blank conditions).
   Residual is used only for **automatic box placement**.
3. Shape-aware ROI on residual (VSD domain rules):
   - **point / circle**: one hotspot (upper-third prior), thin margin, size-capped.
   - **bar**: two poles + weaker bridge between them, thin margin, hard area/side caps.
   - **letter / triangle**: corner hotspots + weak edges, central prior for letters,
     thin margin, hard area/side caps.
4. Write **single-panel** review figures (stimulus mean + box), one global
   common-mean figure, ROI YAML, optional ``.npy`` masks, and
   ``rois/all_rois.yaml``.

Stimulus identity keys (stable across sessions):

- shapes: ``{color}_{shape_type}_{size}`` e.g. ``white_point_0.1``
- letters: ``letter_{L}_{color}_{size}`` e.g. ``letter_A_white_1``

Does **not** train encoders — ROI review only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

from src.data.averaging import average_frames
from src.data.h5_io import read_trial_by_global_id
from src.data.splits import load_trial_table
from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import VSD_CMAP

REVIEW_DIR = project_root() / "experiments" / "loo_encoding" / "roi_review"

DEFAULT_BOX_H = 20
DEFAULT_BOX_W = 25
DEFAULT_MARGIN = 4  # thin margin around shape response
DEFAULT_POINT_MARGIN = 3

METHOD_ID = "stim_minus_global_common"

# Hard caps: never cover almost the whole frame.
MAX_SIDE_PX = 70
MAX_AREA_FRAC = 0.35  # of frame area
MAX_LETTER_SIDE_PX = 75  # letter_A accepted box is 75 wide

# Forced user-accepted boxes (always used). Re-runs must not wipe these.
PRESERVED_OVERRIDE: dict[str, tuple[int, int, int, int]] = {
    # Kept good (user-accepted; coords unchanged)
    "black_bar_horizontal_0.3": (28, 5, 40, 53),  # stretch top to y≈5
    "black_bar_horizontal_1": (16, 5, 50, 55),
    "black_bar_vertical_0.3": (14, 26, 70, 30),
    "black_bar_vertical_1": (12, 27, 68, 32),
    "black_filled_circle_0.3": (21, 26, 48, 28),  # accepted/great
    "black_triangle_contour_0.4": (9, 22, 58, 30),
    "letter_A_white_1": (21, 24, 75, 43),
    "letter_D_white_1": (20, 28, 68, 32),
    "letter_F_white_1": (14, 34, 58, 40),
    "letter_N_white_1": (12, 28, 76, 40),  # stretch left ~6px
    "white_circle_contour_0.3": (24, 28, 48, 30),
    "white_filled_circle_0.3": (18, 17, 50, 35),  # accepted/great
    "white_filled_circle_0.8": (8, 12, 66, 42),
    "white_point_0.1": (32, 32, 30, 25),
    # Fixed 2026-07-23 from user review feedback
    "black_circle_contour_0.3": (14, 20, 60, 40),  # accepted/great
    "black_circle_contour_0.95": (14, 5, 58, 68),
    "black_point_0.05": (30, 28, 28, 24),
    "black_point_0.1": (30, 28, 32, 26),
    "letter_G_white_1": (10, 25, 85, 60),  # stretch up + right
    "letter_L_white_1": (12, 18, 86, 50),
}

# Previously accepted — keep status=accepted if new auto is similar.
PREVIOUS_ACCEPTED: dict[str, tuple[int, int, int, int]] = dict(PRESERVED_OVERRIDE)

POINT_SHAPE_TYPES = frozenset({"point"})
CIRCLE_SHAPE_TYPES = frozenset(
    {"circle_contour", "filled_circle", "circle"}
)
BAR_SHAPE_TYPES = frozenset({"bar_vertical", "bar_horizontal"})
COMPLEX_SHAPE_TYPES = frozenset({"letter", "triangle_contour", "triangle"})


@dataclass(frozen=True)
class RoiBox:
    """Axis-aligned ROI in image coordinates (row = y, col = x)."""

    x0: int  # leftmost column (inclusive)
    y0: int  # topmost row (inclusive)
    width: int
    height: int

    @property
    def x1(self) -> int:
        return self.x0 + self.width

    @property
    def y1(self) -> int:
        return self.y0 + self.height

    def as_dict(self) -> dict:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "width": self.width,
            "height": self.height,
            "row_start": self.y0,
            "row_end": self.y1,
            "col_start": self.x0,
            "col_end": self.x1,
        }

    def clamp(self, shape: tuple[int, int]) -> RoiBox:
        h, w = shape
        x0 = int(np.clip(self.x0, 0, max(0, w - 1)))
        y0 = int(np.clip(self.y0, 0, max(0, h - 1)))
        width = int(np.clip(self.width, 1, w - x0))
        height = int(np.clip(self.height, 1, h - y0))
        return RoiBox(x0=x0, y0=y0, width=width, height=height)

    def to_mask(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[self.y0 : self.y1, self.x0 : self.x1] = 1
        return mask


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _fmt_size(size_deg: float | None) -> str:
    if size_deg is None or (isinstance(size_deg, float) and np.isnan(size_deg)):
        return "na"
    v = float(size_deg)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def stimulus_id_from_row(row: pd.Series) -> str | None:
    """Stable stimulus identity key from catalog / manifest fields."""
    shape = str(row.get("shape_type", "") or "")
    if not shape or shape == "blank" or bool(row.get("is_blank", False)):
        return None
    color = str(row.get("color", "unknown") or "unknown")
    size = _fmt_size(row.get("size_deg"))
    if shape == "letter":
        letter = row.get("letter")
        if letter is None or (isinstance(letter, float) and np.isnan(letter)):
            text = str(row.get("stimulus_text", ""))
            parts = text.strip().split()
            letter = parts[-1] if parts else "?"
        return f"letter_{str(letter).upper()}_{color}_{size}"
    return f"{color}_{shape}_{size}"


def _load_stimulus_manifest(repo: Path, cfg: dict) -> pd.DataFrame:
    stim_root = resolve_data_path(cfg["paths"]["stimuli_root"], repo)
    monkey = cfg["monkey"]
    man_path = stim_root / monkey / "manifest.parquet"
    if not man_path.exists():
        raise FileNotFoundError(
            f"Missing stimulus manifest at {man_path}. "
            "Run scripts/01b_build_stimulus_images.py first."
        )
    return pd.read_parquet(man_path)


def _attach_stimulus_ids(
    trials: pd.DataFrame, stim_man: pd.DataFrame
) -> pd.DataFrame:
    """Join trials to stimulus metadata on (h5_session/date, condition)."""
    meta = stim_man.copy()
    meta["stimulus_id"] = meta.apply(stimulus_id_from_row, axis=1)
    meta = meta.rename(columns={"h5_session": "date"})
    keep = [
        "date",
        "condition",
        "stimulus_id",
        "shape_type",
        "stimulus_text",
        "color",
        "size_deg",
        "letter",
        "pos_x_deg",
        "pos_y_deg",
        "is_blank",
    ]
    meta = meta[keep].drop_duplicates(subset=["date", "condition"])
    return trials.merge(meta, on=["date", "condition"], how="left")


def build_inventory(joined: pd.DataFrame) -> pd.DataFrame:
    """One row per stimulus_id with sessions, splits, and trial counts."""
    rows: list[dict] = []
    for sid, g in joined.dropna(subset=["stimulus_id"]).groupby("stimulus_id"):
        sessions = (
            g[["date", "condition"]]
            .drop_duplicates()
            .sort_values(["date", "condition"])
        )
        session_list = [
            f"{r.date}/{r.condition}" for r in sessions.itertuples(index=False)
        ]
        split_counts = g["split"].value_counts().to_dict()
        shape = str(g["shape_type"].iloc[0])
        letter = g["letter"].iloc[0]
        if pd.isna(letter):
            letter = None
        else:
            letter = str(letter)
        rows.append(
            {
                "stimulus_id": sid,
                "shape_type": shape,
                "stimulus_text_example": str(g["stimulus_text"].iloc[0]),
                "color": str(g["color"].iloc[0]),
                "size_deg": float(g["size_deg"].iloc[0])
                if pd.notna(g["size_deg"].iloc[0])
                else None,
                "letter": letter,
                "n_sessions": int(len(sessions)),
                "n_trials": int(len(g)),
                "n_train": int(split_counts.get("train", 0)),
                "n_val": int(split_counts.get("val", 0)),
                "n_test": int(split_counts.get("test", 0)),
                "test_only": bool(
                    split_counts.get("train", 0) == 0
                    and split_counts.get("val", 0) == 0
                    and split_counts.get("test", 0) > 0
                ),
                "dates_conditions": ";".join(session_list),
                "pos_x_deg_unique": ",".join(
                    f"{v:g}" for v in sorted(g["pos_x_deg"].dropna().unique())
                ),
                "pos_y_deg_unique": ",".join(
                    f"{v:g}" for v in sorted(g["pos_y_deg"].dropna().unique())
                ),
                "candidate_held_out": _is_held_out_candidate(str(sid), shape),
            }
        )
    inv = pd.DataFrame(rows).sort_values(
        ["candidate_held_out", "shape_type", "stimulus_id"],
        ascending=[False, True, True],
    )
    return inv.reset_index(drop=True)


def _is_held_out_candidate(stimulus_id: str, shape_type: str) -> bool:
    if stimulus_id == "white_point_0.1":
        return True
    if shape_type == "triangle_contour":
        return True
    if shape_type == "bar_vertical":
        return True
    if shape_type == "letter":
        return True
    return False


class ConditionMeanCache:
    """Lazy cache of window-averaged maps keyed by (date, condition)."""

    def __init__(
        self,
        joined: pd.DataFrame,
        *,
        repo: Path,
        spatial_size: tuple[int, int],
        start_frame: int,
        end_frame: int,
    ) -> None:
        self.joined = joined
        self.repo = repo
        self.spatial_size = spatial_size
        self.start_frame = start_frame
        self.end_frame = end_frame
        self._cache: dict[tuple[str, str], tuple[np.ndarray, int]] = {}

    def get(self, date: str, condition: str) -> tuple[np.ndarray, int]:
        key = (str(date), str(condition))
        if key in self._cache:
            return self._cache[key]
        group = self.joined[
            (self.joined["date"] == date) & (self.joined["condition"] == condition)
        ].reset_index(drop=True)
        mean_map, n = _mean_map_from_group(
            group,
            repo=self.repo,
            spatial_size=self.spatial_size,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
        )
        self._cache[key] = (mean_map, n)
        return mean_map, n

    def mean_over(
        self, pairs: list[tuple[str, str]], *, trial_weighted: bool = True
    ) -> tuple[np.ndarray, int]:
        if not pairs:
            raise ValueError("No (date, condition) pairs to average")
        maps: list[np.ndarray] = []
        weights: list[int] = []
        for date, condition in pairs:
            m, n = self.get(date, condition)
            maps.append(m)
            weights.append(max(n, 1))
        stack = np.stack(maps, axis=0).astype(np.float64)
        if trial_weighted:
            w = np.asarray(weights, dtype=np.float64)
            mean_map = np.average(stack, axis=0, weights=w).astype(np.float32)
            n_total = int(sum(weights))
        else:
            mean_map = np.mean(stack, axis=0).astype(np.float32)
            n_total = int(sum(weights))
        return mean_map, n_total


def _mean_map_from_group(
    group: pd.DataFrame,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    start_frame: int,
    end_frame: int,
) -> tuple[np.ndarray, int]:
    height, width = spatial_size
    maps: list[np.ndarray] = []
    for row in group.itertuples(index=False):
        h5_path = resolve_data_path(str(row.target_file), repo)
        if not h5_path.exists():
            continue
        trial = read_trial_by_global_id(h5_path, int(row.trial_global_id))
        maps.append(
            average_frames(
                trial,
                start_frame=start_frame,
                end_frame=end_frame,
                spatial_size=spatial_size,
            )
        )
    if not maps:
        raise FileNotFoundError(
            f"No readable trials for group "
            f"(n_rows={len(group)}, date/conds="
            f"{sorted(set(zip(group['date'], group['condition'])))})"
        )
    mean_map = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
    assert mean_map.shape == (height, width)
    return mean_map, len(maps)


def _unique_date_condition(group: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = (
        group[["date", "condition"]]
        .drop_duplicates()
        .sort_values(["date", "condition"])
    )
    return [(str(r.date), str(r.condition)) for r in pairs.itertuples(index=False)]


def compute_stim_and_residual(
    joined: pd.DataFrame,
    *,
    stimulus_id: str,
    cache: ConditionMeanCache,
    global_common: np.ndarray,
    n_global_trials: int,
) -> dict:
    """Stimulus mean + residual vs global grand-mean of all trials.

    ``global_common`` = trial-weighted mean over **every** non-blank
    (date, condition) across the whole dataset (all sessions). Residual
    ``stim_mean − global_common`` is for automatic box placement only;
    review figures show the raw stimulus mean.
    """
    group = joined[joined["stimulus_id"] == stimulus_id].reset_index(drop=True)
    if group.empty:
        raise ValueError(f"No trials for {stimulus_id!r}")

    stim_pairs = _unique_date_condition(group)
    dates = sorted({d for d, _ in stim_pairs})
    stim_mean, n_stim = cache.mean_over(stim_pairs, trial_weighted=True)
    residual = (stim_mean - global_common).astype(np.float32)

    session_details = []
    for date in dates:
        stim_date_pairs = [(d, c) for d, c in stim_pairs if d == date]
        _, n_stim_s = cache.mean_over(stim_date_pairs, trial_weighted=True)
        session_details.append(
            {
                "date": date,
                "n_stim_trials": n_stim_s,
                "n_global_common_trials": n_global_trials,
                "common_scope": "global_all_trials_all_sessions",
            }
        )

    return {
        "stim_mean": stim_mean,
        "residual": residual,
        "n_stim_trials": n_stim,
        "n_common_trials": n_global_trials,
        "method": METHOD_ID,
        "dates": dates,
        "session_details": session_details,
        "stim_pairs": stim_pairs,
        "common_scope": "global_all_trials_all_sessions",
    }


def _estimate_v2_bottom(stim_mean: np.ndarray, *, border_ignore: int = 3) -> int:
    """Bottom of dominant upper high-response cluster on raw stim (≈ V2)."""
    h, w = stim_mean.shape
    stim_s = ndimage.gaussian_filter(stim_mean.astype(np.float64), sigma=1.5)
    valid = _interior_valid((h, w), border_ignore)
    finite = stim_s[valid]
    thr = float(np.percentile(finite, 92.0))
    hi = valid & (stim_s >= thr)
    labeled, n_lab = ndimage.label(hi)
    v2_bottom = max(15, int(0.2 * h))
    if n_lab > 0:
        best_n = -1
        for i in range(1, n_lab + 1):
            ys, xs = np.where(labeled == i)
            if ys.mean() >= 0.45 * h:
                continue
            if len(ys) > best_n:
                best_n = len(ys)
                v2_bottom = int(ys.max())
    return v2_bottom


def _interior_valid(shape: tuple[int, int], border_ignore: int) -> np.ndarray:
    h, w = shape
    valid = np.ones((h, w), dtype=bool)
    if border_ignore > 0:
        valid[:border_ignore, :] = False
        valid[-border_ignore:, :] = False
        valid[:, :border_ignore] = False
        valid[:, -border_ignore:] = False
    return valid


def _shape_class(shape_type: str) -> str:
    """Map catalog shape_type → placement class."""
    if shape_type in POINT_SHAPE_TYPES:
        return "point"
    if shape_type in CIRCLE_SHAPE_TYPES:
        return "circle"
    if shape_type == "bar_vertical":
        return "bar_vertical"
    if shape_type == "bar_horizontal":
        return "bar_horizontal"
    if shape_type in COMPLEX_SHAPE_TYPES or shape_type == "letter":
        if "triangle" in shape_type:
            return "triangle"
        return "letter"
    return "other"


def _box_from_pixels(
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    shape: tuple[int, int],
    margin: int,
    target_h: int = 1,
    target_w: int = 1,
) -> RoiBox:
    """Tight bbox around pixels + thin margin (no forced expansion beyond target)."""
    h, w = shape
    r0 = max(0, int(ys.min()) - margin)
    r1 = min(h, int(ys.max()) + 1 + margin)
    c0 = max(0, int(xs.min()) - margin)
    c1 = min(w, int(xs.max()) + 1 + margin)
    need_h = max(r1 - r0, target_h)
    need_w = max(c1 - c0, target_w)
    ch = (r0 + r1) / 2.0
    cw = (c0 + c1) / 2.0
    y0 = int(np.clip(round(ch - need_h / 2.0), 0, max(0, h - need_h)))
    x0 = int(np.clip(round(cw - need_w / 2.0), 0, max(0, w - need_w)))
    height = min(need_h, h - y0)
    width = min(need_w, w - x0)
    return RoiBox(x0=x0, y0=y0, width=int(width), height=int(height)).clamp((h, w))


def _apply_hard_caps(
    box: RoiBox,
    *,
    shape: tuple[int, int],
    max_side: int = MAX_SIDE_PX,
    max_area_frac: float = MAX_AREA_FRAC,
    center: tuple[float, float] | None = None,
) -> RoiBox:
    """Shrink oversized boxes toward their content center; never ~full-frame."""
    h, w = shape
    max_area = int(max_area_frac * h * w)
    width = min(box.width, max_side, w)
    height = min(box.height, max_side, h)
    if width * height > max_area:
        scale = (max_area / float(width * height)) ** 0.5
        width = max(8, int(round(width * scale)))
        height = max(8, int(round(height * scale)))
        width = min(width, max_side, w)
        height = min(height, max_side, h)
    if center is None:
        cx = box.x0 + box.width / 2.0
        cy = box.y0 + box.height / 2.0
    else:
        cx, cy = center
    x0 = int(np.clip(round(cx - width / 2.0), 0, max(0, w - width)))
    y0 = int(np.clip(round(cy - height / 2.0), 0, max(0, h - height)))
    return RoiBox(x0=x0, y0=y0, width=width, height=height).clamp((h, w))


def _boxes_similar(
    a: RoiBox,
    b: tuple[int, int, int, int] | RoiBox,
    *,
    iou_min: float = 0.55,
    center_max: float = 12.0,
    area_rel_max: float = 0.45,
) -> bool:
    """True if boxes are close enough to keep a prior accepted status."""
    if isinstance(b, tuple):
        b = RoiBox(x0=b[0], y0=b[1], width=b[2], height=b[3])
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = a.width * a.height + b.width * b.height - inter
    iou = inter / union if union > 0 else 0.0
    acx, acy = a.x0 + a.width / 2.0, a.y0 + a.height / 2.0
    bcx, bcy = b.x0 + b.width / 2.0, b.y0 + b.height / 2.0
    center_dist = float(np.hypot(acx - bcx, acy - bcy))
    area_a = float(a.width * a.height)
    area_b = float(b.width * b.height)
    area_rel = abs(area_a - area_b) / max(area_a, area_b)
    return iou >= iou_min and center_dist <= center_max and area_rel <= area_rel_max


def _score_maps(
    residual: np.ndarray,
    *,
    border_ignore: int = 5,
    smooth_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (smoothed residual, local-contrast score, interior valid mask)."""
    h, w = residual.shape
    resid_s = ndimage.gaussian_filter(residual.astype(np.float64), sigma=smooth_sigma)
    score = resid_s - ndimage.uniform_filter(resid_s, size=25)
    valid = _interior_valid((h, w), border_ignore)
    # Extra edge strip: residual artifacts dominate absolute peaks here.
    valid[:, :5] = False
    valid[:, w - 5 :] = False
    valid[h - 6 :, :] = False
    return resid_s, score, valid


def _components_from_mask(
    mask: np.ndarray,
    resid_s: np.ndarray,
    *,
    min_area: int,
) -> list[dict]:
    labeled, n_lab = ndimage.label(mask)
    comps: list[dict] = []
    for i in range(1, n_lab + 1):
        ys, xs = np.where(labeled == i)
        if ys.size < min_area:
            continue
        comps.append(
            {
                "ys": ys,
                "xs": xs,
                "intensity": float(resid_s[ys, xs].sum()),
                "peak": float(resid_s[ys, xs].max()),
                "cy": float(ys.mean()),
                "cx": float(xs.mean()),
                "area": int(ys.size),
                "x0": int(xs.min()),
                "y0": int(ys.min()),
                "x1": int(xs.max()),
                "y1": int(ys.max()),
            }
        )
    comps.sort(key=lambda c: c["intensity"], reverse=True)
    return comps


def _threshold_components(
    resid_s: np.ndarray,
    score: np.ndarray,
    valid: np.ndarray,
    *,
    hotspot_pct: float,
    min_area: int,
) -> list[dict]:
    finite = score[valid]
    if finite.size == 0:
        return []
    thr = float(np.percentile(finite, hotspot_pct))
    med = float(np.median(resid_s[valid]))
    hi = valid & (score >= thr) & (resid_s > med)
    return _components_from_mask(hi, resid_s, min_area=min_area)


def _peak_fallback_box(
    resid_s: np.ndarray,
    valid: np.ndarray,
    *,
    shape: tuple[int, int],
    margin: int,
    target_h: int,
    target_w: int,
    prefer_upper: bool = False,
    prefer_central: bool = False,
) -> RoiBox:
    h, w = shape
    search = resid_s.copy()
    search[~valid] = -np.inf
    if prefer_upper:
        # Prefer cortical upper band, not the noisy top rim.
        y = np.arange(h)[:, None]
        band = ((y >= 0.12 * h) & (y <= 0.48 * h)).astype(np.float64)
        search = search + 0.35 * band * np.nanstd(resid_s[valid])
        search[: int(0.08 * h), :] = -np.inf
    if prefer_central:
        y = np.arange(h)[:, None]
        x = np.arange(w)[None, :]
        dist = np.hypot((y - 0.45 * h) / h, (x - 0.5 * w) / w)
        search = search - 0.2 * dist * np.nanstd(resid_s[valid])
    cy, cx = np.unravel_index(int(np.argmax(search)), search.shape)
    return _box_from_pixels(
        np.array([cy]),
        np.array([cx]),
        shape=shape,
        margin=margin + 2,
        target_h=target_h,
        target_w=target_w,
    )


def _size_caps_for_point_circle(
    *,
    shape_class: str,
    size_deg: float | None,
) -> tuple[int, int]:
    """Return (max_w, max_h) for single-spot responses."""
    s = 0.3 if size_deg is None or not np.isfinite(size_deg) else float(size_deg)
    if shape_class == "point":
        # Points are small; white_point accepted is 30×25.
        return (32, 28) if s >= 0.1 else (28, 24)
    # Circles grow with diameter.
    if s <= 0.3:
        return (42, 36)
    if s <= 0.5:
        return (50, 42)
    if s <= 0.8:
        return (58, 48)
    return (62, 52)


def _bridge_pixels(
    resid_s: np.ndarray,
    valid: np.ndarray,
    poles: list[dict],
    *,
    bridge_pct: float = 82.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixels above a lower threshold inside the bbox spanning the poles."""
    if len(poles) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    x0 = min(c["x0"] for c in poles)
    x1 = max(c["x1"] for c in poles)
    y0 = min(c["y0"] for c in poles)
    y1 = max(c["y1"] for c in poles)
    # Pad the corridor slightly so the weaker bridge is included.
    h, w = resid_s.shape
    x0 = max(0, x0 - 2)
    x1 = min(w - 1, x1 + 2)
    y0 = max(0, y0 - 2)
    y1 = min(h - 1, y1 + 2)
    band = resid_s[y0 : y1 + 1, x0 : x1 + 1]
    band_valid = valid[y0 : y1 + 1, x0 : x1 + 1]
    if not np.any(band_valid):
        return np.array([], dtype=int), np.array([], dtype=int)
    thr = float(np.percentile(band[band_valid], bridge_pct))
    med = float(np.median(resid_s[valid]))
    thr = max(thr, med)
    by, bx = np.where(band_valid & (band >= thr))
    if by.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return by + y0, bx + x0


def _select_bar_poles(components: list[dict], *, shape_class: str) -> list[dict]:
    """Pick the two strongest poles that form a bar-like pair."""
    if not components:
        return []
    if len(components) == 1:
        return [components[0]]
    primary = components[0]
    best = None
    best_score = -1.0
    for cand in components[1:8]:
        if cand["intensity"] < 0.18 * primary["intensity"]:
            continue
        dx = abs(cand["cx"] - primary["cx"])
        dy = abs(cand["cy"] - primary["cy"])
        sep = float(np.hypot(dx, dy))
        if sep < 8:
            continue
        # Prefer axis-aligned separation matching the bar orientation.
        if shape_class == "bar_vertical":
            # Cortical map of a vertical bar is often a *horizontal* two-pole streak.
            axis_score = dx / max(dy, 1.0)
        else:  # bar_horizontal → often more vertical/compact streak in cortex
            axis_score = max(dx, dy) / max(min(dx, dy), 1.0)
        score = cand["intensity"] * (1.0 + 0.15 * min(sep, 50)) * (1.0 + 0.1 * axis_score)
        if score > best_score:
            best_score = score
            best = cand
    if best is None:
        # Fallback: second-strongest with any separation.
        for cand in components[1:]:
            if abs(cand["cx"] - primary["cx"]) + abs(cand["cy"] - primary["cy"]) >= 8:
                best = cand
                break
    return [primary, best] if best is not None else [primary]


def _select_complex_hotspots(
    components: list[dict],
    *,
    shape_class: str,
    h: int,
    w: int,
    max_n: int = 5,
) -> list[dict]:
    """Corner-like hotspots for letters/triangles (central prior for letters)."""
    if not components:
        return []
    pool = components
    if shape_class == "letter":
        mid = [
            c
            for c in components
            if 0.18 * h <= c["cy"] <= 0.78 * h and 0.08 * w <= c["cx"] <= 0.92 * w
        ]
        if mid:
            pool = mid
    kept: list[dict] = [pool[0]]
    kept_ids = {id(pool[0])}
    best_i = pool[0]["intensity"]
    # Greedy: next pick maximizes distance × intensity (spread to corners).
    while len(kept) < max_n:
        best_cand = None
        best_score = -1.0
        for cand in pool[1:]:
            if id(cand) in kept_ids:
                continue
            if cand["intensity"] < 0.12 * best_i:
                continue
            min_dist = min(
                float(np.hypot(cand["cx"] - k["cx"], cand["cy"] - k["cy"]))
                for k in kept
            )
            if min_dist < 8:
                continue
            score = cand["intensity"] * (1.0 + 0.25 * min(min_dist, 40))
            if score > best_score:
                best_score = score
                best_cand = cand
        if best_cand is None:
            break
        kept.append(best_cand)
        kept_ids.add(id(best_cand))
    return kept


def _prefer_region(
    components: list[dict],
    *,
    y_lo: float,
    y_hi: float,
    x_lo: float = 0.0,
    x_hi: float = 1e9,
    boost: float = 1.6,
) -> list[dict]:
    """Re-rank components with a soft spatial prior (does not drop outside)."""
    if not components:
        return components

    def key(c: dict) -> float:
        in_band = y_lo <= c["cy"] <= y_hi and x_lo <= c["cx"] <= x_hi
        return c["intensity"] * (boost if in_band else 1.0)

    return sorted(components, key=key, reverse=True)


def _local_contour_around_peak(
    resid_s: np.ndarray,
    valid: np.ndarray,
    *,
    cy: float,
    cx: float,
    half_win: int,
    contour_pct: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixels in a local window above a percentile of that window (shape growth)."""
    h, w = resid_s.shape
    y0 = max(0, int(round(cy)) - half_win)
    y1 = min(h, int(round(cy)) + half_win + 1)
    x0 = max(0, int(round(cx)) - half_win)
    x1 = min(w, int(round(cx)) + half_win + 1)
    band = resid_s[y0:y1, x0:x1]
    band_valid = valid[y0:y1, x0:x1]
    if not np.any(band_valid):
        return np.array([int(round(cy))]), np.array([int(round(cx))])
    thr = float(np.percentile(band[band_valid], contour_pct))
    med = float(np.median(resid_s[valid]))
    thr = max(thr, med)
    by, bx = np.where(band_valid & (band >= thr))
    if by.size == 0:
        return np.array([int(round(cy))]), np.array([int(round(cx))])
    return by + y0, bx + x0


def propose_roi_on_residual(
    residual: np.ndarray,
    stim_mean: np.ndarray,
    *,
    shape_type: str = "unknown",
    size_deg: float | None = None,
    target_h: int = DEFAULT_BOX_H,
    target_w: int = DEFAULT_BOX_W,
    margin: int = DEFAULT_MARGIN,
    border_ignore: int = 5,
    smooth_sigma: float = 1.5,
) -> RoiBox:
    """Shape-aware ROI from residual hotspots (box drawn on stim mean).

    VSD domain rules:
    - point/circle → one spot (upper-third prior), thin margin, size-capped
    - bar → two poles + weaker bridge, thin margin, hard side/area caps
    - letter/triangle → corner hotspots + weak edges, thin margin, caps
    """
    h, w = residual.shape
    cls = _shape_class(shape_type)
    use_margin = DEFAULT_POINT_MARGIN if cls == "point" else margin
    resid_s, score, valid = _score_maps(
        residual, border_ignore=border_ignore, smooth_sigma=smooth_sigma
    )

    # Hard spatial priors for search masks (avoid edge / bottom artifacts).
    search_valid = valid.copy()
    # Skip top ~20% for point/circle: V2 / vessel rim dominates residual there.
    # True spots sit in the cortical upper band below that rim.
    if cls in {"point", "circle"}:
        search_valid[: int(0.20 * h), :] = False
        if cls == "point":
            search_valid[int(0.55 * h) :, :] = False
        else:
            s = 0.3 if size_deg is None else float(size_deg)
            y_hi = 0.58 * h if s <= 0.5 else 0.68 * h
            search_valid[int(y_hi) :, :] = False
    else:
        search_valid[: int(0.08 * h), :] = False

    if cls == "point":
        hotspot_pct, min_area, max_n = 96.5, 5, 1
    elif cls == "circle":
        hotspot_pct, min_area, max_n = 95.5, 8, 1
    elif cls in {"bar_vertical", "bar_horizontal"}:
        hotspot_pct, min_area, max_n = 95.5, 8, 2
    else:
        hotspot_pct, min_area, max_n = 95.0, 8, 5

    components = _threshold_components(
        resid_s,
        score,
        search_valid if cls in {"point", "circle"} else valid,
        hotspot_pct=hotspot_pct,
        min_area=min_area,
    )

    # Letters/triangles: also seed from stim-mean hotspots so weak corners
    # (e.g. letter L left arm) are not missed when residual is right-skewed.
    if cls in {"letter", "triangle"}:
        stim_s = ndimage.gaussian_filter(stim_mean.astype(np.float64), sigma=smooth_sigma)
        stim_score = stim_s - ndimage.uniform_filter(stim_s, size=25)
        stim_comps = _threshold_components(
            stim_s, stim_score, valid, hotspot_pct=93.0, min_area=6
        )
        # Keep stim comps that add spatial coverage.
        if components:
            best_i = components[0]["intensity"]
            for sc in stim_comps[:8]:
                if all(
                    np.hypot(sc["cx"] - c["cx"], sc["cy"] - c["cy"]) > 12
                    for c in components
                ):
                    # Scale intensity into residual units roughly.
                    sc = dict(sc)
                    sc["intensity"] = 0.5 * best_i * (
                        sc["intensity"] / max(stim_comps[0]["intensity"], 1e-12)
                    )
                    components.append(sc)
            components.sort(key=lambda c: c["intensity"], reverse=True)
        elif stim_comps:
            components = stim_comps

    if cls in {"point", "circle"} and components:
        components = _prefer_region(
            components, y_lo=0.22 * h, y_hi=0.50 * h, boost=1.6
        )

    if not components:
        box = _peak_fallback_box(
            resid_s,
            search_valid if cls in {"point", "circle"} else valid,
            shape=(h, w),
            margin=use_margin,
            target_h=max(12, target_h // 2),
            target_w=max(12, target_w // 2),
            prefer_upper=cls in {"point", "circle"},
            prefer_central=cls == "letter",
        )
        max_side = MAX_LETTER_SIDE_PX if cls in {"letter", "triangle"} else MAX_SIDE_PX
        return _apply_hard_caps(box, shape=(h, w), max_side=max_side)

    if cls in {"point", "circle"}:
        kept = components[0]
        s = 0.3 if size_deg is None else float(size_deg)
        half_win = 14 if cls == "point" else (18 if s <= 0.3 else (24 if s <= 0.8 else 28))
        ys, xs = _local_contour_around_peak(
            resid_s,
            search_valid,
            cy=kept["cy"],
            cx=kept["cx"],
            half_win=half_win,
            contour_pct=88.0 if cls == "point" else 85.0,
        )
        max_w, max_h = _size_caps_for_point_circle(shape_class=cls, size_deg=size_deg)
        box = _box_from_pixels(
            ys, xs, shape=(h, w), margin=use_margin, target_h=1, target_w=1
        )
        return _apply_hard_caps(
            box,
            shape=(h, w),
            max_side=max(max_w, max_h),
            max_area_frac=min(MAX_AREA_FRAC, (max_w * max_h) / float(h * w)),
            center=(float(xs.mean()), float(ys.mean())),
        )

    if cls in {"bar_vertical", "bar_horizontal"}:
        kept = _select_bar_poles(components, shape_class=cls)
        ys_list = [c["ys"] for c in kept]
        xs_list = [c["xs"] for c in kept]
        by, bx = _bridge_pixels(resid_s, valid, kept, bridge_pct=80.0)
        if by.size:
            ys_list.append(by)
            xs_list.append(bx)
        ys = np.concatenate(ys_list)
        xs = np.concatenate(xs_list)
        box = _box_from_pixels(
            ys, xs, shape=(h, w), margin=use_margin, target_h=1, target_w=1
        )
        cx = float(np.mean([c["cx"] for c in kept]))
        cy = float(np.mean([c["cy"] for c in kept]))
        return _apply_hard_caps(
            box, shape=(h, w), max_side=MAX_SIDE_PX, center=(cx, cy)
        )

    # Letters / triangles / other complex shapes.
    if cls == "letter" and components:
        components = _prefer_region(
            components, y_lo=0.20 * h, y_hi=0.75 * h, boost=1.5
        )
    kept = _select_complex_hotspots(
        components, shape_class=cls, h=h, w=w, max_n=max_n
    )
    ys_list = [c["ys"] for c in kept]
    xs_list = [c["xs"] for c in kept]
    # Local contour around each hotspot + bridge between them.
    for c in kept:
        ly, lx = _local_contour_around_peak(
            resid_s, valid, cy=c["cy"], cx=c["cx"], half_win=16, contour_pct=84.0
        )
        ys_list.append(ly)
        xs_list.append(lx)
    by, bx = _bridge_pixels(resid_s, valid, kept, bridge_pct=78.0)
    if by.size:
        ys_list.append(by)
        xs_list.append(bx)
    ys = np.concatenate(ys_list)
    xs = np.concatenate(xs_list)
    box = _box_from_pixels(
        ys, xs, shape=(h, w), margin=use_margin, target_h=1, target_w=1
    )
    cx = float(np.mean([c["cx"] for c in kept]))
    cy = float(np.mean([c["cy"] for c in kept]))
    max_side = MAX_LETTER_SIDE_PX if cls in {"letter", "triangle"} else MAX_SIDE_PX
    return _apply_hard_caps(
        box, shape=(h, w), max_side=max_side, center=(cx, cy)
    )


# Back-compat aliases (older helpers removed; keep names if imported elsewhere).
def _bbox_gap(a: dict, b: dict) -> int:
    dx = max(0, max(a["x0"] - b["x1"], b["x0"] - a["x1"]))
    dy = max(0, max(a["y0"] - b["y1"], b["y0"] - a["y1"]))
    return int(max(dx, dy))


def _union_bbox(comps: list[dict]) -> dict:
    return {
        "x0": min(c["x0"] for c in comps),
        "y0": min(c["y0"] for c in comps),
        "x1": max(c["x1"] for c in comps),
        "y1": max(c["y1"] for c in comps),
        "cx": float(np.mean([c["cx"] for c in comps])),
        "cy": float(np.mean([c["cy"] for c in comps])),
    }


def _shared_limits(mean_map: np.ndarray) -> tuple[float, float]:
    finite = mean_map[np.isfinite(mean_map)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo != 0 else 1e-6
        return lo - pad, hi + pad
    return lo, hi


def _draw_roi(ax, roi: RoiBox) -> None:
    rect = mpatches.Rectangle(
        (roi.x0 - 0.5, roi.y0 - 0.5),
        roi.width,
        roi.height,
        linewidth=2.2,
        edgecolor="white",
        facecolor="none",
        linestyle="-",
    )
    rect_outline = mpatches.Rectangle(
        (roi.x0 - 0.5, roi.y0 - 0.5),
        roi.width,
        roi.height,
        linewidth=3.6,
        edgecolor="black",
        facecolor="none",
        linestyle="-",
    )
    ax.add_patch(rect_outline)
    ax.add_patch(rect)


def plot_stim_mean_with_roi(
    *,
    stim_mean: np.ndarray,
    roi: RoiBox,
    stimulus_id: str,
    n_trials: int,
    n_common_trials: int,
    window_label: str,
    sessions_label: str,
    status: str = "proposed",
    output_path: Path,
) -> None:
    """Single-panel review figure: stimulus mean map + ROI box."""
    fig, ax = plt.subplots(figsize=(6.4, 5.8), layout="constrained")
    vmin, vmax = _shared_limits(stim_mean)
    im = ax.imshow(stim_mean, cmap=VSD_CMAP, vmin=vmin, vmax=vmax, origin="upper")
    _draw_roi(ax, roi)
    ax.set_xlabel("column (x)")
    ax.set_ylabel("row (y)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="VSD signal")
    fig.suptitle(
        f"{stimulus_id}  |  stimulus mean  |  n_stim={n_trials}  |  "
        f"window {window_label}\n"
        f"ROI x0={roi.x0}, y0={roi.y0}, w={roi.width}, h={roi.height}  |  "
        f"status={status}  |  box from stim−global_common "
        f"(n_global={n_common_trials})",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        f"sessions: {sessions_label}",
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_common_mean(
    *,
    common_map: np.ndarray,
    title: str,
    subtitle: str,
    output_path: Path,
) -> None:
    vmin, vmax = _shared_limits(common_map)
    fig, ax = plt.subplots(figsize=(6.2, 5.6), layout="constrained")
    im = ax.imshow(common_map, cmap=VSD_CMAP, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_xlabel("column (x)")
    ax.set_ylabel("row (y)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="VSD signal")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_roi_override(spec: str) -> tuple[str, RoiBox]:
    """Parse ``stimulus_id=x0,y0,width,height``."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"ROI override must be stimulus_id=x0,y0,w,h (got {spec!r})"
        )
    sid, coords = spec.split("=", 1)
    parts = [p.strip() for p in coords.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"ROI override coords must be x0,y0,w,h (got {coords!r})"
        )
    x0, y0, width, height = (int(p) for p in parts)
    return sid.strip(), RoiBox(x0=x0, y0=y0, width=width, height=height)


def write_roi_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/default.yaml",
    )
    p.add_argument(
        "--window-config",
        type=Path,
        default=project_root() / "configs/windows/evoked_35_42.yaml",
        help="Half-open frame window (default [35, 42)).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REVIEW_DIR,
    )
    p.add_argument(
        "--stimulus-ids",
        nargs="+",
        default=None,
        help="Stimulus IDs to propose ROIs for (default: all in inventory).",
    )
    p.add_argument("--box-height", type=int, default=DEFAULT_BOX_H)
    p.add_argument("--box-width", type=int, default=DEFAULT_BOX_W)
    p.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    p.add_argument(
        "--roi-override",
        action="append",
        default=[],
        type=parse_roi_override,
        metavar="ID=x0,y0,w,h",
        help="Manual ROI box (repeatable). Skips auto-proposal for that ID.",
    )
    p.add_argument(
        "--status-override",
        action="append",
        default=[],
        metavar="ID=status",
        help="Force status accepted|proposed for an ID (repeatable).",
    )
    p.add_argument(
        "--inventory-only",
        action="store_true",
        help="Write inventory CSV/JSON only (skip mean maps / ROI figures).",
    )
    p.add_argument(
        "--no-masks",
        action="store_true",
        help="Skip writing binary ROI mask .npy files.",
    )
    return p.parse_args()


def _parse_status_overrides(specs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Bad --status-override {spec!r}; want ID=accepted|proposed")
        sid, status = spec.split("=", 1)
        status = status.strip()
        if status not in {"accepted", "proposed"}:
            raise SystemExit(f"Bad status {status!r} for {sid!r}")
        out[sid.strip()] = status
    return out


def _cleanup_stale_session_common_artifacts(figures_dir: Path, rois_dir: Path) -> None:
    """Remove old per-session common-mean figures/maps (no longer used)."""
    for path in figures_dir.glob("common_mean__session_*.png"):
        path.unlink(missing_ok=True)
    for path in rois_dir.glob("common_mean__session_*.npy"):
        path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    repo = project_root()
    cfg = _load_yaml(args.config)
    win = _load_yaml(args.window_config)
    start_frame = int(win["start_frame"])
    end_frame = int(win["end_frame"])
    window_id = str(win.get("window_id", f"win_{start_frame:04d}_{end_frame:04d}"))
    spatial_size = tuple(int(v) for v in cfg["spatial_size"])
    monkey = cfg["monkey"]
    overrides = dict(args.roi_override)
    status_overrides = _parse_status_overrides(args.status_override)

    # Seed white_point (and any CLI overrides) as forced boxes.
    for sid, coords in PRESERVED_OVERRIDE.items():
        if sid not in overrides:
            overrides[sid] = RoiBox(
                x0=coords[0], y0=coords[1], width=coords[2], height=coords[3]
            )
        status_overrides.setdefault(sid, "accepted")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    rois_dir = out_dir / "rois"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rois_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_session_common_artifacts(figures_dir, rois_dir)

    print("Loading trials + stimulus manifest...")
    trials = load_trial_table(
        cfg["split_csv"],
        monkey,
        trials_index_csv=cfg.get("trials_index_csv"),
        project_root_path=repo,
    )
    available = trials["target_file"].apply(
        lambda p: resolve_data_path(p, repo).exists()
    )
    n_missing = int((~available).sum())
    if n_missing:
        print(f"Skipping {n_missing} trials with missing session H5")
        trials = trials[available].reset_index(drop=True)

    stim_man = _load_stimulus_manifest(repo, cfg)
    joined = _attach_stimulus_ids(trials, stim_man)

    inventory = build_inventory(joined)
    inv_csv = out_dir / "stimulus_inventory.csv"
    inv_json = out_dir / "stimulus_inventory.json"
    inventory.to_csv(inv_csv, index=False)
    inv_json.write_text(inventory.to_json(orient="records", indent=2) + "\n")
    focus = inventory[inventory["candidate_held_out"]].copy()
    focus_csv = out_dir / "heldout_candidates.csv"
    focus.to_csv(focus_csv, index=False)
    print(f"Inventory: {inv_csv} ({len(inventory)} stimulus_ids)")
    print(f"Held-out candidates: {focus_csv} ({len(focus)} rows)")

    if args.inventory_only:
        return 0

    stimulus_ids = (
        list(args.stimulus_ids)
        if args.stimulus_ids is not None
        else inventory["stimulus_id"].tolist()
    )
    shape_by_id = {
        str(r.stimulus_id): str(r.shape_type) for r in inventory.itertuples(index=False)
    }

    window_label = f"[{start_frame}, {end_frame}) / {window_id}"
    print(
        f"Computing ROIs (stim − global all-trials common) over frames "
        f"{window_label} for {len(stimulus_ids)} stimuli..."
    )
    print(
        "Common mean = grand mean of ALL trials in ALL sessions "
        "(all non-blank conditions). Used for auto box placement only; "
        "figures show stimulus mean."
    )

    cache = ConditionMeanCache(
        joined,
        repo=repo,
        spatial_size=spatial_size,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    # Global common = grand mean of all trials across all sessions.
    all_pairs = _unique_date_condition(joined.dropna(subset=["stimulus_id"]))
    global_common, n_global = cache.mean_over(all_pairs, trial_weighted=True)
    global_fig = figures_dir / "common_mean__global_all_conditions.png"
    plot_common_mean(
        common_map=global_common,
        title="Global common mean (all trials × all sessions)",
        subtitle=(
            f"Grand mean of every non-blank trial | n_trials={n_global} | "
            f"window {window_label}"
        ),
        output_path=global_fig,
    )
    np.save(rois_dir / "common_mean__global_all_conditions.npy", global_common)
    print(f"Global common mean → {global_fig.name} (n_trials={n_global})")

    review_index: list[dict] = []
    all_rois: list[dict] = []

    for sid in stimulus_ids:
        group = joined[joined["stimulus_id"] == sid].reset_index(drop=True)
        if group.empty:
            print(f"WARNING: no trials for stimulus_id={sid!r} — skip")
            continue
        sessions = (
            group[["date", "condition"]]
            .drop_duplicates()
            .sort_values(["date", "condition"])
        )
        sessions_label = ", ".join(
            f"{r.date}/{r.condition}" for r in sessions.itertuples(index=False)
        )
        shape_type = shape_by_id.get(sid, str(group["shape_type"].iloc[0]))
        print(f"ROI for {sid} ({len(group)} trials; {sessions_label})...")

        bundle = compute_stim_and_residual(
            joined,
            stimulus_id=sid,
            cache=cache,
            global_common=global_common,
            n_global_trials=n_global,
        )
        stim_mean = bundle["stim_mean"]
        residual = bundle["residual"]
        n_used = int(bundle["n_stim_trials"])
        n_common = int(bundle["n_common_trials"])
        method = str(bundle["method"])

        if sid in overrides:
            roi = overrides[sid].clamp(spatial_size)
            proposal_notes = (
                f"Preserved/manual ROI "
                f"(x0={roi.x0}, y0={roi.y0}, w={roi.width}, h={roi.height}). "
                f"Auto placement uses {method} (display = stimulus mean)."
            )
            print(f"  Using preserved/override ROI: {roi}")
            status = status_overrides.get(sid, "proposed")
        else:
            size_raw = group["size_deg"].iloc[0]
            size_deg = float(size_raw) if pd.notna(size_raw) else None
            roi = propose_roi_on_residual(
                residual,
                stim_mean,
                shape_type=shape_type,
                size_deg=size_deg,
                target_h=args.box_height,
                target_w=args.box_width,
                margin=args.margin,
            )
            cls = _shape_class(shape_type)
            proposal_notes = (
                f"Shape-aware auto ROI on residual ({method}), class={cls} "
                f"(shape_type={shape_type}, size_deg={size_deg}): hotspot "
                f"components + bridge (bars/letters), thin margin≈{args.margin}px, "
                f"hard caps max_side≤{MAX_SIDE_PX}px / area≤{MAX_AREA_FRAC:.0%} of "
                f"frame. Display = stimulus mean. n_global_common_trials={n_common}."
            )
            print(f"  Auto ROI ({cls}): {roi}")

            # Preserve accepted status only when the new box is still close.
            if sid in status_overrides:
                status = status_overrides[sid]
            elif sid in PREVIOUS_ACCEPTED and _boxes_similar(
                roi, PREVIOUS_ACCEPTED[sid]
            ):
                prev = PREVIOUS_ACCEPTED[sid]
                roi = RoiBox(
                    x0=prev[0], y0=prev[1], width=prev[2], height=prev[3]
                ).clamp(spatial_size)
                status = "accepted"
                proposal_notes += (
                    f" Kept prior accepted coords {prev} (auto was similar)."
                )
                print(f"  Keeping prior accepted ROI (similar): {roi}")
            elif sid in PREVIOUS_ACCEPTED:
                status = "proposed"
                proposal_notes += (
                    f" Prior accepted {PREVIOUS_ACCEPTED[sid]} changed "
                    f"substantially → re-proposed."
                )
                print(
                    f"  Prior accepted changed substantially → proposed: {roi}"
                )
            else:
                status = "proposed"

        fig_path = figures_dir / f"{sid}__mean_map_roi.png"
        plot_stim_mean_with_roi(
            stim_mean=stim_mean,
            roi=roi,
            stimulus_id=sid,
            n_trials=n_used,
            n_common_trials=n_common,
            window_label=window_label,
            sessions_label=sessions_label,
            status=status,
            output_path=fig_path,
        )

        mask_rel = None
        if not args.no_masks:
            mask_path = rois_dir / f"{sid}__mask.npy"
            np.save(mask_path, roi.to_mask(spatial_size))
            mask_rel = str(mask_path.relative_to(repo))

        payload = {
            "stimulus_id": sid,
            "x0": roi.x0,
            "y0": roi.y0,
            "width": roi.width,
            "height": roi.height,
            "status": status,
            "method": method,
            # ROIs are spatial only (window-independent). Proposal window is
            # recorded in figures / review_index / STATUS.md, not ROI YAML.
            "n_stim_trials": n_used,
            "n_common_trials": n_common,
            "n_sessions": int(len(sessions)),
            "roi": roi.as_dict(),
            "proposal_notes": proposal_notes,
            "method_policy": (
                "global_common = trial-weighted grand mean of ALL non-blank "
                "trials across ALL sessions. residual = stim_mean − "
                "global_common (auto box placement only). Review figure shows "
                "the stimulus mean with the box."
            ),
            "monkey": monkey,
            "dates_conditions": [
                {"date": str(r.date), "condition": str(r.condition)}
                for r in sessions.itertuples(index=False)
            ],
            "session_details": bundle["session_details"],
            "shape_type": shape_type,
            "color": str(group["color"].iloc[0]),
            "size_deg": float(group["size_deg"].iloc[0])
            if pd.notna(group["size_deg"].iloc[0])
            else None,
            "letter": None
            if pd.isna(group["letter"].iloc[0])
            else str(group["letter"].iloc[0]),
            "stimulus_text_example": str(group["stimulus_text"].iloc[0]),
            "common_scope": "global_all_trials_all_sessions",
            "figure": str(fig_path.relative_to(repo)),
            "mask_npy": mask_rel,
            "colormap": VSD_CMAP,
        }
        roi_path = rois_dir / f"{sid}.yaml"
        write_roi_yaml(roi_path, payload)

        np.save(rois_dir / f"{sid}__mean_map.npy", stim_mean)
        np.save(rois_dir / f"{sid}__residual_map.npy", residual)

        entry = {
            "stimulus_id": sid,
            "status": status,
            "x0": roi.x0,
            "y0": roi.y0,
            "width": roi.width,
            "height": roi.height,
            "n_stim_trials": n_used,
            "n_common_trials": n_common,
            "method": method,
            "window": window_label,
            "dates_conditions": sessions_label,
            "figure": str(fig_path.relative_to(repo)),
            "roi_yaml": str(roi_path.relative_to(repo)),
            "mask_npy": mask_rel,
        }
        review_index.append(entry)
        all_rois.append(
            {
                "stimulus_id": sid,
                "x0": roi.x0,
                "y0": roi.y0,
                "width": roi.width,
                "height": roi.height,
                "status": status,
                "method": method,
                "n_stim_trials": n_used,
                "n_common_trials": n_common,
                "figure": str(fig_path.relative_to(repo)),
                "roi_yaml": str(roi_path.relative_to(repo)),
            }
        )
        print(
            f"  ROI ({roi.x0},{roi.y0}) {roi.width}×{roi.height} "
            f"[{status}] → {fig_path.name}"
        )

    index_path = out_dir / "review_index.json"
    index_path.write_text(json.dumps(review_index, indent=2) + "\n")

    all_rois_path = rois_dir / "all_rois.yaml"
    write_roi_yaml(
        all_rois_path,
        {
            "method_default": METHOD_ID,
            "common_mean_definition": (
                "Grand mean of ALL non-blank trials across ALL sessions "
                "(trial-weighted over every date×condition). Used only for "
                "automatic ROI placement (stim_mean − global_common). Review "
                "figures show the stimulus mean with the box. Placement is "
                "shape-aware: point/circle = one hotspot; bar = two poles + "
                "bridge; letter/triangle = corner hotspots + weak edges; "
                f"hard caps max_side≤{MAX_SIDE_PX}px, area≤{MAX_AREA_FRAC:.0%}."
            ),
            "n_stimuli": len(all_rois),
            "common_mean_figures": {
                "global": str(global_fig.relative_to(repo)),
            },
            "rois": all_rois,
        },
    )
    (rois_dir / "all_rois.json").write_text(
        json.dumps(
            {
                "n_stimuli": len(all_rois),
                "rois": all_rois,
            },
            indent=2,
        )
        + "\n"
    )

    status_lines = [
        "# ROI review status index",
        "",
        f"Window: `{window_id}` (`[{start_frame}, {end_frame})`).",
        "Method: stim − **global** common mean (grand mean of all trials in "
        "all sessions) for auto box placement; figures show stimulus mean only.",
        f"Stimuli: **{len(all_rois)}**.",
        "",
        "| stimulus_id | status | x0 | y0 | w | h | figure |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for e in sorted(all_rois, key=lambda r: r["stimulus_id"]):
        status_lines.append(
            f"| `{e['stimulus_id']}` | {e['status']} | {e['x0']} | {e['y0']} | "
            f"{e['width']} | {e['height']} | "
            f"`figures/{Path(e['figure']).name}` |"
        )
    status_lines.append("")
    status_lines.append(
        "Please re-review all boxes on the new single-panel stimulus-mean "
        "figures. Placement is shape-aware (point/circle one spot; bar two "
        "poles+bridge; letter/triangle corners+edges) with hard area/side caps."
    )
    status_lines.append("")
    (out_dir / "STATUS.md").write_text("\n".join(status_lines) + "\n")

    print(f"Review index: {index_path}")
    print(f"Master index: {all_rois_path}")
    print(f"Status table: {out_dir / 'STATUS.md'}")
    print(f"Done — {len(all_rois)} ROI proposals under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
