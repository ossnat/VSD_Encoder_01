#!/usr/bin/env python3
"""Schira-only comparison: same legacy stimuli → legacy ``trans_cortex`` vs repo.

No registration / chamber alignment. Input panels share ``trans_cortex`` VF
degree axes (``m1``/``m2`` from ``retino_mat``).

Example::

    scripts/py shira_revised/scripts/04_schira_only_stim_compare.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLBACKEND", "Agg")

from shira_revised.src.legacy_schira import joanna_params
from shira_revised.src.legacy_stimuli import (
    create_cart_stim,
    create_polar_stim,
    vertical_line,
)
from shira_revised.src.legacy_warp import (
    legacy_folded_cortical_xy,
    legacy_trans_cortex_om,
    load_retino_mat_crop,
    repo_schira_warp_on_legacy_vf,
    trans_cortex_vf_grid,
)
from shira_revised.src.plot_vf import imshow_cortical, imshow_vf_degrees


def _plot_row(
    fig: plt.Figure,
    row: int,
    rgb: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    x_deg: np.ndarray,
    y_deg: np.ndarray,
    *,
    name: str,
    params,
    a: float,
    alpha: float,
    k: float,
    grid: int,
) -> None:
    ax0 = fig.add_subplot(3, 3, row * 3 + 1)
    imshow_vf_degrees(
        ax0,
        rgb,
        m1,
        m2,
        title=f"{name}\ninput (legacy VF grid)",
        tick_deg=1.0,
    )

    om, _ = legacy_trans_cortex_om(rgb, x_deg, y_deg, params)
    ax1 = fig.add_subplot(3, 3, row * 3 + 2)
    ax1.imshow(om, origin="upper", aspect="equal", interpolation="nearest")
    ax1.set_xlabel("OM column (trans_cortex x1 index)")
    ax1.set_ylabel("OM row (trans_cortex y1 index)")
    ax1.set_title(f"{name}\nlegacy Schira (trans_cortex OM)", fontsize=11)

    repo_rgb, repo_ext = repo_schira_warp_on_legacy_vf(
        rgb, x_deg, y_deg, a=a, alpha=alpha, k=k, grid=grid
    )
    ax2 = fig.add_subplot(3, 3, row * 3 + 3)
    imshow_cortical(
        ax2,
        repo_rgb,
        repo_ext,
        title=f"{name}\ncurrent repo Schira (P=αθ, signed VF)",
        xlabel="Re(w)",
        ylabel="Im(w)",
    )


def main() -> None:
    out_dir = REPO / "shira_revised" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    mpl_cfg = out_dir / ".mplconfig"
    mpl_cfg.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cfg)

    retino = load_retino_mat_crop()
    n_rows, n_cols = 301, 512
    m1, m2, x_deg, y_deg = trans_cortex_vf_grid(retino, n_rows, n_cols)

    params = joanna_params()
    a, alpha, k = params.a, params.alpha, params.k
    grid = 520  # large cortical sampling grid

    examples = [
        ("Cartesian grid", create_cart_stim()),
        ("Polar grid", create_polar_stim(is_left_hem=True)),
        (
            "Vertical bar (100718)",
            vertical_line((0.6, -0.75), input_size_deg=1.0, vert=True),
        ),
    ]

    # Large figure: full legacy canvas width.
    fig_w = 18
    fig_h = 16
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        "Schira-only (no registration): same stimuli & VF degree grid\n"
        f"Joanna params a={a}, α={alpha} (legacy T=θ/α), k={k} | "
        f"retino_mat corners → m1∈[{m1[0]:.2f},{m1[-1]:.2f}]°, "
        f"m2∈[{m2[0]:.2f},{m2[-1]:.2f}]°",
        fontsize=13,
    )

    for i, (name, rgb) in enumerate(examples):
        _plot_row(
            fig,
            i,
            rgb,
            m1,
            m2,
            x_deg,
            y_deg,
            name=name,
            params=params,
            a=a,
            alpha=alpha,
            k=k,
            grid=grid,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = out_dir / "schira_only_stim_compare.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")

    # VF grid verification inset (tick values vs m1/m2)
    verify = out_dir / "vf_degree_grid_verify.txt"
    lines = [
        "VF degree grid (trans_cortex linspace from retino_mat crop)",
        f"m1[0], m1[-1] = {m1[0]}, {m1[-1]}",
        f"m2[0], m2[-1] = {m2[0]}, {m2[-1]}",
        f"At pixel (row=0,col=0): x={x_deg[0,0]:.4f}, y_abs={y_deg[0,0]:.4f}, y_catalog={-y_deg[0,0]:.4f}",
        f"At pixel (row=-1,col=-1): x={x_deg[-1,-1]:.4f}, y_catalog={-y_deg[-1,-1]:.4f}",
        "Stimulus cart grid lines every 35 px ≈ 1° in pixel space from top-left;",
        "overlay with m1/m2 shows whether linspace VF matches line spacing.",
    ]
    verify.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {verify}")


if __name__ == "__main__":
    main()
