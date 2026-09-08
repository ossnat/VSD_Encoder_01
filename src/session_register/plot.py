"""QC figures for session-camera vasculature registration."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.session_register.fit import landmark_id_label
from src.session_register.transform import SessionTransform


def _as_01(image: np.ndarray, *, stretch: bool = True) -> np.ndarray:
    x = np.asarray(image, dtype=np.float64)
    if not stretch:
        return np.clip(x, 0.0, 1.0)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def save_gray_png(
    image: np.ndarray,
    path: str | Path,
    *,
    title: str | None = None,
    stretch: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(_as_01(image, stretch=stretch), cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def rgb_overlay(fixed: np.ndarray, warped: np.ndarray) -> np.ndarray:
    """White field; fixed vessels red, warped vessels green, overlap blue."""
    f = np.clip(1.0 - np.asarray(fixed, dtype=np.float64), 0.0, 1.0)
    g = np.clip(1.0 - np.asarray(warped, dtype=np.float64), 0.0, 1.0)
    red = 1.0 - g
    green = 1.0 - f
    blue = 1.0 - f * (1.0 - g) - g * (1.0 - f)
    return np.stack([red, green, blue], axis=-1).astype(np.float32)


def checkerboard(
    fixed: np.ndarray,
    warped: np.ndarray,
    *,
    tile_px: int = 12,
    stretch: bool = False,
) -> np.ndarray:
    a = _as_01(fixed, stretch=stretch)
    b = _as_01(warped, stretch=stretch)
    h, w = a.shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    choose_a = ((rows // tile_px) + (cols // tile_px)) % 2 == 0
    out = np.where(choose_a, a, b)
    return out.astype(np.float32)


def plot_registration_qc(
    *,
    fixed_vessel: np.ndarray,
    moving_vessel: np.ndarray,
    warped_moving: np.ndarray,
    transform: SessionTransform,
    metric_name: str,
    metric_value: float,
    path: str | Path,
    fixed_session: str,
    moving_session: str,
) -> Path:
    """One 2×2 figure: fixed | moving | overlay | checkerboard."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = rgb_overlay(fixed_vessel, warped_moving)
    board = checkerboard(fixed_vessel, warped_moving)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.6))
    panels = [
        (axes[0, 0], _as_01(fixed_vessel, stretch=False), "gray", f"fixed {fixed_session}"),
        (axes[0, 1], _as_01(moving_vessel, stretch=False), "gray", f"moving {moving_session}"),
        (axes[1, 0], overlay, None, "overlay (vessels only: fixed=R, warped=G, overlap=B)"),
        (axes[1, 1], board, "gray", "checkerboard"),
    ]
    for ax, img, cmap, title in panels:
        if img.ndim == 3:
            ax.imshow(img, origin="upper")
        else:
            ax.imshow(img, cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

    fig.suptitle(
        f"{moving_session} → {fixed_session}  "
        f"rot={transform.rotation_deg:.2f}°  "
        f"t=[{transform.dx_col:.2f}, {transform.dy_row:.2f}] px  "
        f"{metric_name}={metric_value:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _show(ax, img, title: str, cmap: str | None = "gray", *, stretch: bool = True) -> None:
    if img.ndim == 3:
        ax.imshow(img, origin="upper")
    else:
        ax.imshow(_as_01(img, stretch=stretch), cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def plot_vision_panel(
    *,
    fixed_raw: np.ndarray,
    moving_raw: np.ndarray,
    fixed_vessel: np.ndarray,
    moving_vessel: np.ndarray,
    warped_moving: np.ndarray,
    transform: SessionTransform,
    metric_name: str,
    metric_value: float,
    path: str | Path,
    fixed_session: str,
    moving_session: str,
) -> Path:
    """Mean raw vs bleach, then registered overlay and checkerboard."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = rgb_overlay(fixed_vessel, warped_moving)
    board = checkerboard(fixed_vessel, warped_moving)

    fig, axes = plt.subplots(3, 2, figsize=(8.6, 12.2))
    _show(axes[0, 0], fixed_raw, f"fixed raw {fixed_session}", stretch=True)
    _show(axes[0, 1], moving_raw, f"moving raw {moving_session}", stretch=True)
    _show(axes[1, 0], fixed_vessel, f"fixed vessels {fixed_session}", stretch=False)
    _show(axes[1, 1], moving_vessel, f"moving vessels {moving_session}", stretch=False)
    _show(axes[2, 0], overlay, "overlay after T (fixed=R, warped=G, overlap=B)", cmap=None)
    _show(axes[2, 1], board, "checkerboard after T", stretch=False)
    fig.suptitle(
        f"{moving_session} → {fixed_session}  "
        f"rot={transform.rotation_deg:.2f}°  "
        f"t=[{transform.dx_col:.2f}, {transform.dy_row:.2f}] px  "
        f"{metric_name}={metric_value:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_vessel_extraction_gallery(
    panels: list[tuple[str, np.ndarray, np.ndarray]],
    path: str | Path,
) -> Path:
    """Raw mean vs isolated thin trunks, plus red overlay on raw."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    fig, axes = plt.subplots(n, 3, figsize=(10.4, 3.4 * n))
    if n == 1:
        axes = np.asarray([axes])
    for i, panel in enumerate(panels):
        session, raw, vessels = panel[0], panel[1], panel[2]
        raw01 = _as_01(raw, stretch=True)
        vmask = np.asarray(vessels) < 0.5
        factor = 3
        raw3 = np.repeat(np.repeat(raw01, factor, axis=0), factor, axis=1)
        mask3 = np.repeat(np.repeat(vmask, factor, axis=0), factor, axis=1)
        overlay3 = np.stack([raw3, raw3, raw3], axis=-1)
        overlay3[mask3] = (0.9, 0.05, 0.05)
        _show(axes[i, 0], raw, f"{session} raw", stretch=True)
        _show(
            axes[i, 1],
            vessels,
            f"{session} vessels (2 thin trunks)",
            stretch=False,
        )
        axes[i, 2].imshow(overlay3, origin="upper", interpolation="nearest")
        axes[i, 2].set_title(f"{session} traces on raw", fontsize=10)
        axes[i, 2].set_axis_off()
    fig.suptitle(
        "Vessel isolation — two thin trunks inside the chamber, rim excluded",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_stability_overlay_on_raw(
    *,
    session: str,
    mean_raw: np.ndarray,
    score_display: np.ndarray,
    chamber: np.ndarray,
    path: str | Path,
    corr_method: str = "dark_and_stable",
    overlay_alpha: float = 0.38,
    overlay_gamma: float = 2.2,
    upsample: int = 4,
) -> Path:
    """Overlay stability-map yellow ridges directly on the baseline VSD mean.

    No thresholding, skeletonization, or vessel tracing — only the same hot
    colormap used in the stability panel, alpha-blended onto the raw image.

    ``overlay_gamma`` > 1 keeps pale yellow background faint; only brighter
    ridge pixels show strongly on top of the VSD.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw01 = _as_01(np.asarray(mean_raw, dtype=np.float64), stretch=True)
    score = np.asarray(score_display, dtype=np.float64)
    chamber = np.asarray(chamber, dtype=bool)

    lo, hi = float(np.percentile(score[chamber], 2)), float(
        np.percentile(score[chamber], 98)
    )
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((score - lo) / (hi - lo), 0.0, 1.0)
    hot_rgb = plt.cm.hot(norm)[..., :3]
    # Gamma on norm: pale yellow wash → nearly transparent; ridges → visible tint
    alpha = (norm ** float(overlay_gamma)) * float(overlay_alpha)
    alpha[~chamber] = 0.0

    raw_rgb = np.stack([raw01, raw01, raw01], axis=-1)
    blended = raw_rgb * (1.0 - alpha[..., None]) + hot_rgb * alpha[..., None]
    blended = np.clip(blended, 0.0, 1.0)

    if upsample > 1:
        blended = np.repeat(
            np.repeat(blended, upsample, axis=0), upsample, axis=1
        )
        raw_rgb = np.repeat(
            np.repeat(raw_rgb, upsample, axis=0), upsample, axis=1
        )

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    axes[0].imshow(raw_rgb, origin="upper", interpolation="nearest")
    axes[0].set_title(f"{session}  baseline VSD mean", fontsize=10)
    axes[0].set_axis_off()

    stab_rgba = plt.cm.hot(norm)
    stab_rgba[~chamber, :3] = 0.35
    if upsample > 1:
        stab_show = np.repeat(
            np.repeat(stab_rgba, upsample, axis=0), upsample, axis=1
        )
    else:
        stab_show = stab_rgba
    axes[1].imshow(stab_show, origin="upper", interpolation="nearest")
    axes[1].set_title(f"stability ({corr_method})", fontsize=10)
    axes[1].set_axis_off()

    axes[2].imshow(blended, origin="upper", interpolation="nearest")
    axes[2].set_title("yellow stability ridges on VSD (no tracing)", fontsize=10)
    axes[2].set_axis_off()

    fig.suptitle(
        f"{session} — stability overlay only  method={corr_method}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_vsd_yellow_blue_landmarks(
    *,
    session: str,
    mean_raw: np.ndarray,
    score_display: np.ndarray,
    chamber: np.ndarray,
    landmarks: np.ndarray,
    path: str | Path,
    corr_method: str = "dark_and_stable",
    overlay_alpha: float = 0.38,
    overlay_gamma: float = 2.2,
    upsample: int = 4,
    n_landmarks: int | None = None,
    ridge_mask: np.ndarray | None = None,
    landmark_ids: np.ndarray | None = None,
) -> Path:
    """Baseline VSD + yellow stability ridges + blue landmarks on the yellow curve."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw01 = _as_01(np.asarray(mean_raw, dtype=np.float64), stretch=True)
    score = np.asarray(score_display, dtype=np.float64)
    chamber = np.asarray(chamber, dtype=bool)
    lms = np.asarray(landmarks).reshape(-1, 2) if np.asarray(landmarks).size else np.zeros((0, 2), dtype=int)
    if n_landmarks is None:
        n_landmarks = len(lms)

    lo, hi = float(np.percentile(score[chamber], 2)), float(
        np.percentile(score[chamber], 98)
    )
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((score - lo) / (hi - lo), 0.0, 1.0)
    hot_rgb = plt.cm.hot(norm)[..., :3]
    alpha = (norm ** float(overlay_gamma)) * float(overlay_alpha)
    alpha[~chamber] = 0.0

    raw_rgb = np.stack([raw01, raw01, raw01], axis=-1)
    blended = raw_rgb * (1.0 - alpha[..., None]) + hot_rgb * alpha[..., None]
    blended = np.clip(blended, 0.0, 1.0)
    # Paint the 1-px trough (where landmarks live) as a stronger yellow line
    if ridge_mask is not None:
        rmask = np.asarray(ridge_mask, dtype=bool)
        blended[rmask] = (0.98, 0.92, 0.15)

    up = max(int(upsample), 1)
    if up > 1:
        blended = np.repeat(np.repeat(blended, up, axis=0), up, axis=1)

    ids = None
    if landmark_ids is not None and np.asarray(landmark_ids).size:
        ids = np.asarray(landmark_ids, dtype=np.int64).reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(blended, origin="upper", interpolation="nearest")
    for i, (row, col) in enumerate(lms):
        x = int(col) * up + (up - 1) / 2.0
        y = int(row) * up + (up - 1) / 2.0
        ax.plot(
            x,
            y,
            marker="+",
            color="#1558c0",
            markersize=14,
            markeredgewidth=2.2,
            linestyle="none",
            zorder=5,
        )
        if ids is not None and i < len(ids):
            ax.annotate(
                landmark_id_label(int(ids[i, 0]), int(ids[i, 1])),
                (x + 1.6 * up, y - 1.6 * up),
                color="#1558c0",
                fontsize=8,
                fontweight="bold",
                zorder=6,
            )
    ax.set_title(
        f"{session}  homologous stations  U/L/J  (N={n_landmarks})",
        fontsize=11,
    )
    ax.set_axis_off()
    fig.suptitle(
        f"Stability-ridge landmarks  method={corr_method}",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_landmark_register_qc(
    *,
    fixed_raw: np.ndarray,
    moving_raw: np.ndarray,
    warped_moving: np.ndarray,
    fixed_landmarks: np.ndarray,
    moving_landmarks: np.ndarray,
    warped_moving_landmarks: np.ndarray,
    transform: SessionTransform,
    path: str | Path,
    fixed_session: str,
    moving_session: str,
    n_matched: int,
    landmark_rmsd: float,
    upsample: int = 3,
    fixed_ids: np.ndarray | None = None,
    moving_ids: np.ndarray | None = None,
) -> Path:
    """VSD pair + warped overlay with blue (fixed) / magenta (warped moving) landmarks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    up = max(int(upsample), 1)

    def _gray(img: np.ndarray) -> np.ndarray:
        g = _as_01(np.asarray(img, dtype=np.float64), stretch=True)
        if up > 1:
            g = np.repeat(np.repeat(g, up, axis=0), up, axis=1)
        return np.stack([g, g, g], axis=-1)

    def _marks(ax, pts, color: str, ids: np.ndarray | None = None) -> None:
        pts = np.asarray(pts).reshape(-1, 2)
        if pts.size == 0:
            return
        xs = pts[:, 1] * up + (up - 1) / 2.0
        ys = pts[:, 0] * up + (up - 1) / 2.0
        ax.plot(
            xs,
            ys,
            marker="+",
            color=color,
            markersize=11,
            markeredgewidth=1.8,
            linestyle="none",
        )
        if ids is None or not np.asarray(ids).size:
            return
        id_arr = np.asarray(ids, dtype=np.int64).reshape(-1, 2)
        for i, (x, y) in enumerate(zip(xs, ys)):
            if i >= len(id_arr):
                break
            ax.annotate(
                landmark_id_label(int(id_arr[i, 0]), int(id_arr[i, 1])),
                (x + 1.4 * up, y - 1.4 * up),
                color=color,
                fontsize=7,
                fontweight="bold",
            )

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    axes[0].imshow(_gray(fixed_raw), origin="upper", interpolation="nearest")
    _marks(axes[0], fixed_landmarks, "#1558c0", fixed_ids)
    axes[0].set_title(f"fixed {fixed_session}  blue +", fontsize=10)
    axes[0].set_axis_off()

    axes[1].imshow(_gray(moving_raw), origin="upper", interpolation="nearest")
    _marks(axes[1], moving_landmarks, "#c01880", moving_ids)
    axes[1].set_title(f"moving {moving_session}  magenta +", fontsize=10)
    axes[1].set_axis_off()

    axes[2].imshow(_gray(warped_moving), origin="upper", interpolation="nearest")
    _marks(axes[2], fixed_landmarks, "#1558c0", fixed_ids)
    _marks(axes[2], warped_moving_landmarks, "#c01880", moving_ids)
    axes[2].set_title(
        f"warped overlay  matched={n_matched}  rmsd={landmark_rmsd:.2f}px",
        fontsize=10,
    )
    axes[2].set_axis_off()
    fig.suptitle(
        f"{moving_session} → {fixed_session}  "
        f"rot={transform.rotation_deg:.2f}°  "
        f"t=[{transform.dx_col:.2f}, {transform.dy_row:.2f}] px",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation_vessel_gallery(
    *,
    session: str,
    mean_raw: np.ndarray,
    stability_map: np.ndarray,
    debug: dict,
    vessel_maps,
    path: str | Path,
    corr_method: str = "inter_trial_corr",
    row0: int = 0,
    row1: int | None = None,
    upsample: int = 3,
) -> Path:
    """Rich diagnostic for correlation-based vessel detection.

    Layout (2 rows × 3 cols):
      Row 0: mean raw | stability map (hot) | thresholded vessels on stability
      Row 1: vessel mask on raw | zoomed raw + vessel overlay | zoomed + landmarks

    Parameters
    ----------
    session : str
        Session label used in titles.
    mean_raw : np.ndarray (H, W)
        Mean VSD image (for raw-overlay panels).
    stability_map : np.ndarray (H, W)
        Output of ``correlation_vessel_map`` — raw stability scores.
    debug : dict
        The ``debug`` dict returned by ``correlation_vessel_map``.
    vessel_maps : VesselMaps
        Output of ``correlation_vessel_map``.
    path : str | Path
        Output PNG path.
    corr_method : str
        Method label for the title.
    row0, row1 : int
        Row crop for the zoom panels (default = full image).
    upsample : int
        Upsampling factor for zoom panels.
    """
    import matplotlib.colors as mcolors

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    H, W = mean_raw.shape
    if row1 is None:
        row1 = H

    raw01 = _as_01(np.asarray(mean_raw, dtype=np.float64), stretch=True)
    stab = np.asarray(stability_map, dtype=np.float64)
    smoothed = np.asarray(debug.get("smoothed", stab), dtype=np.float64)
    chamber = np.asarray(debug.get("chamber", np.ones((H, W), dtype=bool)), dtype=bool)
    threshold = float(debug.get("threshold", 0.5))
    vmask = np.asarray(vessel_maps.vessel_mask, dtype=bool)
    landmarks = np.asarray(vessel_maps.landmarks).reshape(-1, 2) if vessel_maps.landmarks.size else np.zeros((0, 2), dtype=int)
    n_trials = int(debug.get("n_trials", 0))

    # Panel helpers
    def _make_stab_rgba(arr: np.ndarray) -> np.ndarray:
        """Stability map: hot colormap, gray outside chamber."""
        lo, hi = float(np.percentile(arr[chamber], 2)), float(np.percentile(arr[chamber], 98))
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        rgba = plt.cm.hot(norm)
        rgba[~chamber, :3] = 0.35  # gray outside
        return rgba.astype(np.float32)

    def _vessel_on_stab(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Stability map with vessel pixels outlined in cyan."""
        rgba = _make_stab_rgba(arr).copy()
        # Slightly dim non-vessel pixels
        rgba[~mask, :3] *= 0.55
        # Paint vessel pixels cyan
        rgba[mask] = [0.0, 0.9, 0.9, 1.0]
        return rgba

    def _vessel_on_raw(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Grayscale raw with vessel pixels in red."""
        rgb = np.stack([raw, raw, raw], axis=-1).copy()
        rgb[mask] = [0.92, 0.08, 0.08]
        return rgb.astype(np.float32)

    def _zoom_overlay_with_landmarks(
        raw: np.ndarray, mask: np.ndarray, lms: np.ndarray, r0: int, r1: int, up: int
    ) -> np.ndarray:
        crop = raw[r0:r1, :]
        cmask = mask[r0:r1, :]
        crop4 = np.repeat(np.repeat(crop, up, axis=0), up, axis=1)
        mask4 = np.repeat(np.repeat(cmask, up, axis=0), up, axis=1)
        rgb = np.stack([crop4, crop4, crop4], axis=-1).copy()
        rgb[mask4] = [0.92, 0.08, 0.08]
        return rgb.astype(np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))

    # (0,0) mean raw
    axes[0, 0].imshow(raw01, cmap="gray", origin="upper", vmin=0, vmax=1)
    axes[0, 0].set_title(f"{session}  mean raw (N={n_trials} trials)", fontsize=10)
    axes[0, 0].set_axis_off()

    # (0,1) stability map
    axes[0, 1].imshow(_make_stab_rgba(smoothed), origin="upper", interpolation="nearest")
    axes[0, 1].set_title(f"stability ({corr_method})", fontsize=10)
    cb_ax = axes[0, 1].inset_axes([1.02, 0.0, 0.04, 1.0])
    sm = plt.cm.ScalarMappable(cmap="hot")
    sm.set_array([])
    fig.colorbar(sm, cax=cb_ax, label="stability score")
    axes[0, 1].set_axis_off()

    # (0,2) thresholded vessel pixels on stability
    axes[0, 2].imshow(_vessel_on_stab(smoothed, vmask), origin="upper", interpolation="nearest")
    axes[0, 2].set_title(f"vessels (pct={debug.get('threshold', threshold):.3f} thr)", fontsize=10)
    axes[0, 2].set_axis_off()

    # (1,0) vessel mask painted on raw mean
    axes[1, 0].imshow(_vessel_on_raw(raw01, vmask), origin="upper", interpolation="nearest")
    axes[1, 0].set_title("vessel mask on raw", fontsize=10)
    axes[1, 0].set_axis_off()

    # (1,1) zoom of vessel overlay (no landmarks)
    crop_rgb = _zoom_overlay_with_landmarks(raw01, vmask, landmarks, row0, row1, upsample)
    axes[1, 1].imshow(crop_rgb, origin="upper", interpolation="nearest")
    axes[1, 1].set_title(f"zoom rows {row0}–{row1}  ×{upsample}  (red=vessel)", fontsize=10)
    axes[1, 1].set_axis_off()

    # (1,2) zoom + landmark crosses
    axes[1, 2].imshow(crop_rgb, origin="upper", interpolation="nearest")
    for row, col in landmarks:
        if row0 <= int(row) < row1:
            axes[1, 2].plot(
                int(col) * upsample + (upsample - 1) / 2.0,
                (int(row) - row0) * upsample + (upsample - 1) / 2.0,
                marker="+",
                color="cyan",
                markersize=11,
                markeredgewidth=1.8,
            )
    axes[1, 2].set_title(
        f"landmarks (N={len(landmarks)}) cyan +", fontsize=10
    )
    axes[1, 2].set_axis_off()

    fig.suptitle(
        f"Correlation-based vessel detection — {session}  "
        f"method={corr_method}  vessel_pct={debug.get('threshold', threshold):.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_vessel_zoom_overlay(
    panels: list[tuple],
    path: str | Path,
    *,
    row0: int = 38,
    row1: int = 82,
) -> Path:
    """Zoomed raw mean with red traces — check they sit in the dark rivers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    if n == 1:
        axes = np.asarray([axes])
    for ax, panel in zip(axes, panels):
        session, raw, vessels = panel[0], panel[1], panel[2]
        landmarks = panel[3] if len(panel) > 3 else None
        raw01 = _as_01(raw, stretch=True)
        crop = raw01[row0:row1, :]
        vmask = np.asarray(vessels)[row0:row1, :] < 0.5
        factor = 4
        crop4 = np.repeat(np.repeat(crop, factor, axis=0), factor, axis=1)
        mask4 = np.repeat(np.repeat(vmask, factor, axis=0), factor, axis=1)
        overlay = np.stack([crop4, crop4, crop4], axis=-1)
        overlay[mask4] = (0.95, 0.05, 0.05)
        ax.imshow(overlay, origin="upper", interpolation="nearest")
        if landmarks is not None and len(np.asarray(landmarks)):
            pts = np.asarray(landmarks).reshape(-1, 2)
            for row, col in pts:
                if row0 <= int(row) < row1:
                    ax.plot(
                        int(col) * factor + (factor - 1) / 2.0,
                        (int(row) - row0) * factor + (factor - 1) / 2.0,
                        marker="+",
                        color="cyan",
                        markersize=9,
                        markeredgewidth=1.4,
                    )
        ax.set_title(session, fontsize=11)
        ax.set_axis_off()
    fig.suptitle(
        f"zoom r{row0}-{row1}: red = trunks, cyan + = junctions",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
