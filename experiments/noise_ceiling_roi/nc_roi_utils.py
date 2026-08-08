"""Shared helpers for across-condition noise-ceiling hull ROI.

Extracted from the obsolete per-stim / pooled / max-r pipelines so
``across_condition/compute_across_condition_reliability.py`` can run without
those scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy import ndimage
from scipy.spatial import ConvexHull

from src.encoding.ridge import pearson_r
from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.pixel_correlation import load_trial_mean_maps
from src.paths import project_root, resolve_data_path
from src.stimuli.identity import attach_stimulus_ids

DEFAULT_INVENTORY = (
    project_root()
    / "experiments/loo_encoding/roi_review/stimulus_inventory.csv"
)

# Convenient default for *ROI creation* only (raw [35, 42)).
# This is independent of the LOO / ridge analysis window: pass any window YAML
# via --window to build the hull on a different evoked range (e.g. 35–46 while
# encoding stays on 35–42). Normalization follows the chosen window YAML.
DEFAULT_WINDOW = project_root() / "configs/windows/evoked_35_42.yaml"

# Fixed seed for report shuffle controls (stimulus↔response pairing break).
DEFAULT_SHUFFLE_SEED = 17

DEFAULT_CONFIG = project_root() / "configs/default.yaml"

DEFAULT_MIN_COMPONENT_PIXELS = 50

DEFAULT_KEEP_TOP_N = 2

def spearman_brown(r: float) -> float:
    """Spearman-Brown prediction from split-half correlation."""
    if not np.isfinite(r):
        return float("nan")
    if r <= -1.0:
        return float("nan")
    return float(2.0 * r / (1.0 + r))

def spearman_brown_map(r_map: np.ndarray) -> np.ndarray:
    """Element-wise Spearman-Brown, clipped to [-1, 1]."""
    out = np.full_like(r_map, np.nan, dtype=np.float32)
    finite = np.isfinite(r_map)
    r = r_map[finite].astype(np.float64)
    sb = 2.0 * r / (1.0 + r)
    out[finite] = np.clip(sb, -1.0, 1.0).astype(np.float32)
    return out

def split_half_trial_vectors(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Odd-even split along trial axis.

    Returns (group_a, group_b) with shapes (T_a, H, W) and (T_b, H, W).
    """
    group_a = trials[0::2]
    group_b = trials[1::2]
    return group_a, group_b

def overlay_hull(
    ax,
    hull_vertices: np.ndarray,
    *,
    color: str = "lime",
    linewidth: float = 2.5,
    fill: bool = False,
    fill_alpha: float = 0.12,
    underlay: bool = True,
    linestyle: str = "solid",
) -> None:
    """Draw thick high-contrast hull outline (data coords: x=col, y=row).

    Vertices are pixel-center coordinates matching ``imshow`` default extent
    (pixel *i* centered at coordinate *i*). A white underlay improves contrast
    against the reliability colormap.
    """
    if hull_vertices.shape[0] < 3:
        if hull_vertices.shape[0] > 0:
            ax.plot(
                hull_vertices[:, 0],
                hull_vertices[:, 1],
                "o",
                color=color,
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=6,
            )
        return
    closed = np.vstack([hull_vertices, hull_vertices[0]])
    xs, ys = closed[:, 0], closed[:, 1]
    if fill:
        ax.fill(xs, ys, color=color, alpha=fill_alpha, linewidth=0, zorder=4)
    if underlay:
        ax.plot(
            xs,
            ys,
            color="white",
            linewidth=linewidth + 1.5,
            linestyle=linestyle,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5,
        )
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=6,
    )

def overlay_seed_pixels(
    ax,
    r_map: np.ndarray,
    *,
    threshold: float,
    color: str = "cyan",
    alpha: float = 0.55,
    s: float = 14.0,
) -> int:
    """Scatter lightly mark pixels with r >= threshold (seed set for hull)."""
    above = np.isfinite(r_map) & (r_map >= threshold)
    ys, xs = np.nonzero(above)
    if xs.size == 0:
        return 0
    ax.scatter(
        xs,
        ys,
        s=s,
        c=color,
        alpha=alpha,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
        rasterized=True,
    )
    return int(xs.size)

def seed_mask_from_r_map(r_map: np.ndarray, *, threshold: float) -> np.ndarray:
    """Binary mask of finite pixels with r >= threshold."""
    return (np.isfinite(r_map) & (r_map >= threshold)).astype(bool)

def filter_seed_components(
    seed_mask: np.ndarray,
    *,
    min_component_pixels: int = 1,
    keep_top_n: int | None = None,
) -> np.ndarray:
    """
    Keep 8-connected components by min area and/or largest-N.

    1. Drop components with area < ``min_component_pixels`` (if > 1).
    2. If ``keep_top_n`` is set, keep only the largest ``keep_top_n`` among
       survivors (by pixel count).
    """
    labeled, n_labels = ndimage.label(seed_mask.astype(bool))
    if n_labels == 0:
        return np.zeros_like(seed_mask, dtype=bool)

    counts = np.bincount(labeled.ravel())
    eligible = [
        lab
        for lab in range(1, n_labels + 1)
        if counts[lab] >= int(min_component_pixels)
    ]
    if keep_top_n is not None and keep_top_n > 0 and eligible:
        eligible = sorted(eligible, key=lambda lab: counts[lab], reverse=True)[
            : int(keep_top_n)
        ]
    keep = np.zeros(n_labels + 1, dtype=bool)
    for lab in eligible:
        keep[lab] = True
    return keep[labeled]

def convex_hull_from_seed_mask(
    seed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Convex hull of True pixels in ``seed_mask``.

    Returns (filled_hull_mask, hull_vertices, n_seed_pixels).
    Vertices are (N, 2) float (x, y) = (col, row) at pixel centers.
    """
    h, w = seed_mask.shape
    ys, xs = np.nonzero(seed_mask)
    n_seed = int(xs.size)
    if n_seed == 0:
        return np.zeros((h, w), dtype=np.uint8), np.empty((0, 2), dtype=np.float64), 0
    if n_seed < 3:
        pts = np.column_stack([xs, ys]).astype(np.float64)
        return np.zeros((h, w), dtype=np.uint8), pts, n_seed

    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    try:
        hull = ConvexHull(points)
        hull_vertices = points[hull.vertices].astype(np.float64)
    except Exception:
        # Collinear or degenerate seeds: fall back to axis-aligned bbox corners.
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        if x0 == x1 and y0 == y1:
            hull_vertices = np.array([[x0, y0]], dtype=np.float64)
            return np.zeros((h, w), dtype=np.uint8), hull_vertices, n_seed
        hull_vertices = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            dtype=np.float64,
        )

    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    # radius=0.5 includes boundary pixel centers on the hull edges.
    mask = MplPath(hull_vertices).contains_points(grid, radius=0.5).reshape(h, w)
    return mask.astype(np.uint8), hull_vertices, n_seed

def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Axis-aligned bbox (x0, y0, width, height) covering True/1 pixels."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0

def overlay_seed_mask(
    ax,
    seed_mask: np.ndarray,
    *,
    color: str = "cyan",
    alpha: float = 0.55,
    s: float = 14.0,
) -> int:
    """Scatter-mark True pixels in a binary seed mask."""
    ys, xs = np.nonzero(seed_mask)
    if xs.size == 0:
        return 0
    ax.scatter(
        xs,
        ys,
        s=s,
        c=color,
        alpha=alpha,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
        rasterized=True,
    )
    return int(xs.size)

def load_pairs(cfg: dict, repo: Path) -> pd.DataFrame:
    window_id = cfg.get("window_id") or (
        f"win_{cfg['start_frame']:04d}_{cfg['end_frame']:04d}"
    )
    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        cfg["monkey"],
        window_id,
    )
    if not pairs_path.is_file():
        raise FileNotFoundError(
            f"Encoding pairs manifest not found: {pairs_path}\n"
            "Run scripts/01c_build_encoding_pairs.py for this window."
        )
    pairs = pd.read_parquet(pairs_path)
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    pairs = attach_stimulus_ids(pairs)
    return pairs.sort_values("trial_global_id").reset_index(drop=True)

def pixel_reliability_map(trials: np.ndarray) -> np.ndarray:
    """
    Per-pixel Pearson r between odd- and even-index trial vectors.

    At each (y, x), correlate ``trials[0::2, y, x]`` vs ``trials[1::2, y, x]``,
    trimming to ``min(n_odd, n_even)`` when lengths differ.
    """
    if trials.ndim != 3:
        raise ValueError(f"Expected (T, H, W), got {trials.shape}")

    a = trials[0::2].astype(np.float64)
    b = trials[1::2].astype(np.float64)
    n = min(a.shape[0], b.shape[0])
    if n < 2:
        h, w = trials.shape[1], trials.shape[2]
        return np.full((h, w), np.nan, dtype=np.float32)

    a = a[:n]
    b = b[:n]
    a_c = a - np.nanmean(a, axis=0, keepdims=True)
    b_c = b - np.nanmean(b, axis=0, keepdims=True)
    num = np.nansum(a_c * b_c, axis=0)
    denom = np.sqrt(np.nansum(a_c**2, axis=0) * np.nansum(b_c**2, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / denom
    return corr.astype(np.float32)


def global_split_half_r(trials: np.ndarray) -> tuple[float, float, int, int]:
    """
    Global split-half reliability on mean maps of odd/even halves.

    Returns (r_half, r_sb, n_odd, n_even).
    """
    group_a, group_b = split_half_trial_vectors(trials)
    n_odd = int(group_a.shape[0])
    n_even = int(group_b.shape[0])
    if n_odd == 0 or n_even == 0:
        return float("nan"), float("nan"), n_odd, n_even

    mean_a = np.nanmean(group_a, axis=0)
    mean_b = np.nanmean(group_b, axis=0)
    r_half = pearson_r(mean_a, mean_b)
    r_sb = spearman_brown(r_half)
    return r_half, r_sb, n_odd, n_even


def load_stimulus_trial_stack(
    stim_trials: pd.DataFrame,
    *,
    repo: Path,
    cfg: dict,
    spatial_size: tuple[int, int],
) -> np.ndarray:
    """Load window-averaged trial maps (T, H, W), honoring window normalization."""
    stim_trials = stim_trials.sort_values("trial_global_id").reset_index(drop=True)
    return load_trial_mean_maps(
        stim_trials,
        repo=repo,
        spatial_size=spatial_size,
        start_frame=int(cfg["start_frame"]),
        end_frame=int(cfg["end_frame"]),
        avg_method=cfg.get("avg_method", "mean"),
        normalization=str(cfg.get("normalization", "none")),
        baseline_start_frame=int(cfg.get("baseline_start_frame", 2)),
        baseline_end_frame=int(cfg.get("baseline_end_frame", 26)),
        baseline_std_eps=float(cfg.get("baseline_std_eps", 1e-8)),
    )


def half_mean_maps(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Odd/even split → mean map per half.

    Even indices (0, 2, …) → even mean; odd indices (1, 3, …) → odd mean.
    Same deterministic convention as ``split_half_trial_vectors``.
    """
    group_even, group_odd = split_half_trial_vectors(trials)
    n_even = int(group_even.shape[0])
    n_odd = int(group_odd.shape[0])
    if n_even == 0 or n_odd == 0:
        h, w = trials.shape[1], trials.shape[2]
        nan_map = np.full((h, w), np.nan, dtype=np.float32)
        return nan_map, nan_map, n_odd, n_even
    even_mean = np.nanmean(group_even, axis=0).astype(np.float32)
    odd_mean = np.nanmean(group_odd, axis=0).astype(np.float32)
    return odd_mean, even_mean, n_odd, n_even


def pixel_correlate_stacks(
    stack_a: np.ndarray,
    stack_b: np.ndarray,
) -> np.ndarray:
    """
    Per-pixel Pearson r between two stacks of shape ``(N, H, W)``.

    Vector length at each pixel is ``N`` (= n_stimuli), not n_trials.
    """
    if stack_a.shape != stack_b.shape:
        raise ValueError(
            f"Stack shapes must match, got {stack_a.shape} vs {stack_b.shape}"
        )
    if stack_a.ndim != 3:
        raise ValueError(f"Expected (N, H, W), got {stack_a.shape}")
    n = int(stack_a.shape[0])
    if n < 2:
        h, w = stack_a.shape[1], stack_a.shape[2]
        return np.full((h, w), np.nan, dtype=np.float32)

    a = stack_a.astype(np.float64)
    b = stack_b.astype(np.float64)
    a_c = a - np.nanmean(a, axis=0, keepdims=True)
    b_c = b - np.nanmean(b, axis=0, keepdims=True)
    num = np.nansum(a_c * b_c, axis=0)
    denom = np.sqrt(np.nansum(a_c**2, axis=0) * np.nansum(b_c**2, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / denom
    return corr.astype(np.float32)


def global_pattern_r(
    odd_stack: np.ndarray,
    even_stack: np.ndarray,
) -> tuple[float, float]:
    """
    Whole-field pattern reliability from across-condition half-means.

    Flatten(mean of odd maps) vs flatten(mean of even maps), then SB.
    """
    mean_odd = np.nanmean(odd_stack, axis=0)
    mean_even = np.nanmean(even_stack, axis=0)
    r_half = float(pearson_r(mean_odd, mean_even))
    return r_half, spearman_brown(r_half)


def cleaned_hull_from_r_map(
    r_map: np.ndarray,
    *,
    threshold: float,
    min_component_pixels: int,
    keep_top_n: int | None,
) -> dict:
    """Threshold → CC filter → optional keep_top_n → cleaned vs naive hull."""
    above = seed_mask_from_r_map(r_map, threshold=threshold)
    n_above = int(above.sum())
    cleaned = filter_seed_components(
        above,
        min_component_pixels=min_component_pixels,
        keep_top_n=keep_top_n,
    )
    n_after_cc = int(cleaned.sum())
    mask_naive, hull_naive, _ = convex_hull_from_seed_mask(above)
    mask_clean, hull_clean, _ = convex_hull_from_seed_mask(cleaned)
    return {
        "above": above,
        "cleaned_seeds": cleaned,
        "n_above": n_above,
        "n_after_cc": n_after_cc,
        "mask_naive": mask_naive.astype(np.uint8),
        "hull_naive": hull_naive,
        "n_hull_naive": int(mask_naive.sum()),
        "mask_clean": mask_clean.astype(np.uint8),
        "hull_clean": hull_clean,
        "n_hull_clean": int(mask_clean.sum()),
    }


def shuffle_even_stack_pairing(
    odd_stack: np.ndarray,
    even_stack: np.ndarray,
    *,
    seed: int = DEFAULT_SHUFFLE_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Break stimulus↔response pairing for a shuffle control.

    Keeps ``odd_stack`` in original condition order; permutes the condition
    axis of ``even_stack`` with a fixed RNG seed so odd condition *i* is
    correlated against a mismatched even condition.

    Returns ``(odd_stack, even_shuffled, perm)`` where ``perm`` is the
    permutation applied to the even stack's first axis.
    """
    if odd_stack.shape != even_stack.shape:
        raise ValueError(
            f"Stack shapes must match, got {odd_stack.shape} vs {even_stack.shape}"
        )
    n = int(odd_stack.shape[0])
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    # Avoid accidental identity permutation for small n when possible.
    if n >= 2 and np.array_equal(perm, np.arange(n)):
        perm = np.roll(np.arange(n), 1)
    return odd_stack, even_stack[perm], perm.astype(np.int64)


def build_across_condition_half_stacks(
    pairs: pd.DataFrame,
    unit_ids: list[str],
    *,
    repo: Path,
    cfg: dict,
    spatial_size: tuple[int, int],
    unit_col: str = "stimulus_id",
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """
    Build odd/even half-mean stacks for across-condition reliability.

    ``unit_ids`` are values of ``unit_col`` (usually ``stimulus_id``). Each
    unit contributes one odd and one even mean map when both halves exist.

    Returns ``(odd_stack, even_stack, used_ids, trial_rows)``.
    """
    odd_means: list[np.ndarray] = []
    even_means: list[np.ndarray] = []
    used_ids: list[str] = []
    trial_rows: list[dict] = []

    for unit_id in sorted(unit_ids):
        unit_trials = pairs[pairs[unit_col] == unit_id]
        if unit_trials.empty:
            print(f"SKIP {unit_id}: no trials")
            continue
        trials = load_stimulus_trial_stack(
            unit_trials, repo=repo, cfg=cfg, spatial_size=spatial_size
        )
        odd_mean, even_mean, n_odd, n_even = half_mean_maps(trials)
        if n_odd == 0 or n_even == 0:
            print(f"SKIP {unit_id}: need both halves (n_odd={n_odd}, n_even={n_even})")
            continue
        odd_means.append(odd_mean)
        even_means.append(even_mean)
        used_ids.append(str(unit_id))
        trial_rows.append(
            {
                unit_col: unit_id,
                "n_trials": int(trials.shape[0]),
                "n_odd": n_odd,
                "n_even": n_even,
            }
        )
        print(
            f"  {unit_id}: n_trials={trials.shape[0]}  "
            f"n_odd={n_odd}  n_even={n_even}"
        )

    if len(used_ids) < 2:
        raise RuntimeError(
            f"Need >= 2 units with both halves; got {len(used_ids)}"
        )

    return (
        np.stack(odd_means, axis=0),
        np.stack(even_means, axis=0),
        used_ids,
        trial_rows,
    )


def blank_pairs_from_encoding(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return rows flagged blank (``is_blank`` or ``shape_type == blank``)."""
    if pairs.empty:
        return pairs.copy()
    mask = np.zeros(len(pairs), dtype=bool)
    if "is_blank" in pairs.columns:
        mask |= pairs["is_blank"].fillna(False).astype(bool).to_numpy()
    if "shape_type" in pairs.columns:
        mask |= pairs["shape_type"].astype(str).str.lower().eq("blank").to_numpy()
    if "stimulus_text" in pairs.columns:
        mask |= (
            pairs["stimulus_text"]
            .astype(str)
            .str.contains(r"\bblank\b", case=False, regex=True, na=False)
            .to_numpy()
        )
    return pairs.loc[mask].copy()


def attach_blank_unit_ids(blank_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Label blank trials as per-session units for across-condition control.

    Blanks share one catalog identity, so session-level units
    (``blank_<date>``) provide the N≥2 axis analogous to stimuli.
    """
    out = blank_pairs.copy()
    date_col = "date" if "date" in out.columns else "h5_session"
    if date_col not in out.columns:
        raise KeyError(
            "Blank pairs need a 'date' or 'h5_session' column for unit IDs"
        )
    out["blank_unit_id"] = out[date_col].astype(str).map(lambda d: f"blank_{d}")
    return out


def outer_contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    """
    External contours of a binary mask via marching squares (matplotlib).

    Returns list of (N, 2) float arrays in (x, y) = (col, row) pixel-center
    coordinates, matching ``imshow(..., origin='upper')``.
    """
    if not mask.any():
        return []
    # Pad so components touching the FOV edge still form closed paths.
    padded = np.pad(mask.astype(np.float64), 1, mode="constant", constant_values=0.0)
    ys = np.arange(padded.shape[0], dtype=np.float64) - 1.0
    xs = np.arange(padded.shape[1], dtype=np.float64) - 1.0
    # Temporary axes; closed immediately (Agg backend, no display).
    fig, ax = plt.subplots(1, 1)
    try:
        cs = ax.contour(xs, ys, padded, levels=[0.5], origin="upper")
        # Matplotlib ContourSet: allsegs[level_idx] -> list of (N, 2) paths.
        segs = getattr(cs, "allsegs", None)
        if segs is None or len(segs) == 0:
            paths: list[np.ndarray] = []
        else:
            paths = [np.asarray(s, dtype=np.float64) for s in segs[0] if len(s) >= 3]
    finally:
        plt.close(fig)
    return paths


def fill_contours_mask(
    contours: list[np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    """Fill exterior contour polygons into a binary mask (uint8)."""
    h, w = shape
    if not contours:
        return np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    filled = np.zeros(h * w, dtype=bool)
    for verts in contours:
        if verts.shape[0] < 3:
            continue
        filled |= MplPath(verts).contains_points(grid, radius=0.5)
    return filled.reshape(h, w).astype(np.uint8)


def compute_stimulus_metrics(
    stim_trials: pd.DataFrame,
    *,
    repo: Path,
    cfg: dict,
    spatial_size: tuple[int, int],
    trials: np.ndarray | None = None,
) -> dict:
    """Load trial stack and compute reliability metrics for one stimulus."""
    if trials is None:
        trials = load_stimulus_trial_stack(
            stim_trials, repo=repo, cfg=cfg, spatial_size=spatial_size
        )
    n_trials = int(trials.shape[0])
    r_half, r_sb, n_odd, n_even = global_split_half_r(trials)
    r_map = pixel_reliability_map(trials)
    stim_mean = np.nanmean(trials, axis=0).astype(np.float32)

    finite = np.isfinite(r_map)
    peak_pixel_r = (
        float(np.nanmax(r_map[finite])) if finite.any() else float("nan")
    )

    return {
        "stimulus_id": stim_trials["stimulus_id"].iloc[0],
        "n_trials": n_trials,
        "n_odd": n_odd,
        "n_even": n_even,
        "r_half_global": r_half,
        "r_sb_global": r_sb,
        "peak_pixel_r": peak_pixel_r,
        "stim_mean": stim_mean,
        "r_map": r_map,
    }

