"""Matplotlib mosaic UI for megapixel traces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from tools.space_conv.megapixel import MegapixelStack

# Inclusive last frame index shown in mosaic (if window extends there) and large plots.
MAX_DISPLAY_FRAME = 200


def display_frame_end_exclusive(n_frames: int, *, max_frame: int = MAX_DISPLAY_FRAME) -> int:
    """Exclusive end for slicing frames with index ≤ max_frame (and < n_frames)."""
    return int(min(n_frames, max_frame + 1))


def _fill_between(
    ax: Axes,
    frames: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    color: str = "C0",
    alpha: float = 0.25,
    linewidth: float = 1.0,
) -> None:
    ax.plot(frames, mean, color=color, lw=linewidth)
    ax.fill_between(
        frames,
        mean - std,
        mean + std,
        color=color,
        alpha=alpha,
        linewidth=0,
    )


def _window_ylim(
    mean: np.ndarray,
    std: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> tuple[float, float]:
    """Y-limits from analysis-window mean±std so enlarged plots match the mosaic."""
    m = mean[start_frame:end_frame]
    s = std[start_frame:end_frame]
    y_lo = float(np.nanmin(m - s))
    y_hi = float(np.nanmax(m + s))
    pad = 0.05 * (y_hi - y_lo + 1e-6)
    return y_lo - pad, y_hi + pad


def show_mosaic(
    stack: MegapixelStack,
    *,
    start_frame: int,
    end_frame: int,
    title: str,
    h5_path: Path | None = None,
    max_display_frame: int = MAX_DISPLAY_FRAME,
) -> None:
    """
    Show 10×10 mosaic of mean±std over the analysis window.

    Click a cell to open a large full-trial figure (window highlighted).
    Multiple clicks open multiple figures. Close the mosaic to exit.

    Large traces use the same megapixel series as the mosaic cell, extended
    through frame ``max_display_frame`` (inclusive). Y-limits are taken from the
    analysis window so early-frame outliers do not flatten the plot.
    """
    n_mh, n_mw = stack.grid_shape
    n_frames = stack.n_frames
    display_end = display_frame_end_exclusive(n_frames, max_frame=max_display_frame)
    # Mosaic only draws the analysis window; clamp to displayable frames.
    plot_start = start_frame
    plot_end = min(end_frame, display_end)
    if plot_start < 0 or plot_end > n_frames or plot_start >= plot_end:
        raise ValueError(
            f"Invalid analysis window [{start_frame}, {end_frame}) "
            f"for stack with {n_frames} frames "
            f"(displayable [0, {display_end}))"
        )

    frames_win = np.arange(plot_start, plot_end)
    fig, axes = plt.subplots(
        n_mh,
        n_mw,
        figsize=(14, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_mh == 1 and n_mw == 1:
        axes = np.array([[axes]])
    elif n_mh == 1:
        axes = axes.reshape(1, -1)
    elif n_mw == 1:
        axes = axes.reshape(-1, 1)

    # Shared y-limits over window for readability
    win_means = stack.mean[:, :, plot_start:plot_end]
    win_stds = stack.std[:, :, plot_start:plot_end]
    y_lo = float(np.nanmin(win_means - win_stds))
    y_hi = float(np.nanmax(win_means + win_stds))
    pad = 0.05 * (y_hi - y_lo + 1e-6)
    y_lo, y_hi = y_lo - pad, y_hi + pad

    ax_to_ij: dict[Axes, tuple[int, int]] = {}
    for i in range(n_mh):
        for j in range(n_mw):
            ax = axes[i, j]
            m = stack.mean[i, j, plot_start:plot_end]
            s = stack.std[i, j, plot_start:plot_end]
            _fill_between(ax, frames_win, m, s, linewidth=0.8)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xticks([])
            ax.set_yticks([])
            ax_to_ij[ax] = (i, j)

    std_note = (
        "std across pixels in block"
        if stack.mode == "single"
        else f"std across {stack.n_trials} trials (block-mean)"
    )
    fig.suptitle(
        f"{title}\n"
        f"window [{plot_start}, {plot_end})  ·  {stack.normalization}  ·  {std_note}\n"
        f"click a cell → detail through frame {max_display_frame}",
        fontsize=11,
    )

    def on_click(event) -> None:
        if event.inaxes is None:
            return
        ij = ax_to_ij.get(event.inaxes)
        if ij is None:
            return
        i, j = ij
        _show_large_trace(
            stack,
            i,
            j,
            start_frame=start_frame,
            end_frame=end_frame,
            title=title,
            h5_path=h5_path,
            max_display_frame=max_display_frame,
        )

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


def _show_large_trace(
    stack: MegapixelStack,
    mega_i: int,
    mega_j: int,
    *,
    start_frame: int,
    end_frame: int,
    title: str,
    h5_path: Path | None,
    max_display_frame: int = MAX_DISPLAY_FRAME,
) -> None:
    display_end = display_frame_end_exclusive(
        stack.n_frames, max_frame=max_display_frame
    )
    frames = np.arange(display_end)
    # Same series as mosaic[i, j], extended (not a different aggregate / axis).
    mean = np.asarray(stack.mean[mega_i, mega_j, :display_end], dtype=np.float64)
    std = np.asarray(stack.std[mega_i, mega_j, :display_end], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    _fill_between(ax, frames, mean, std, linewidth=1.2)

    win_end = min(end_frame, display_end)
    win_start = min(start_frame, win_end)
    if win_start < win_end:
        ax.axvspan(
            win_start,
            win_end - 1e-6,
            color="orange",
            alpha=0.2,
            label="analysis window",
        )
        ax.axvline(win_start, color="orange", ls="--", lw=1)
        ax.axvline(win_end, color="orange", ls="--", lw=1)
        # Match mosaic shape: autoscale from early-frame outliers flattens the window.
        y_lo, y_hi = _window_ylim(mean, std, win_start, win_end)
        ax.set_ylim(y_lo, y_hi)

    ax.set_xlim(0, max(display_end - 1, 0))
    ax.set_xlabel("frame index")
    ax.set_ylabel("megapixel value")
    ax.set_title(
        f"{title}  ·  megapixel ({mega_i}, {mega_j})  "
        f"[{mega_i * stack.block_size}:{(mega_i + 1) * stack.block_size}, "
        f"{mega_j * stack.block_size}:{(mega_j + 1) * stack.block_size})  "
        f"· frames 0–{display_end - 1}"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    if h5_path is not None:
        fig.canvas.manager.set_window_title(
            f"megapixel ({mega_i},{mega_j}) — {h5_path.name}"
        )
    # Non-blocking so mosaic stays interactive and multiple figures can open
    fig.show()
    fig.canvas.draw_idle()
