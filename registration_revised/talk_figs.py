"""Stand-alone figures for the PI talk (one figure per file, slide-sized) -> figs/talk/.

talk_01a_plate_alignment.png   how the thesis Fig. 11B plate was placed on the camera grid
talk_01b_points_on_thesis.png  the xlsx registration points on thesis Fig. 11B vs Joanna's marks
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figs as MF  # noqa: E402
import points  # noqa: E402

OUT = HERE / "figs" / "talk"
MEAS = "lime"


def plate_and_reg():
    rgb, grey, red_xy, blue_xy = MF.fig11b()
    bv = np.asarray(Image.open(MF.LEG / "images/100718/BV_resized.png").convert("L")).astype(float)
    reg = MF.register_plate(grey, bv)
    return rgb, grey, red_xy, blue_xy, bv, reg


def fig_alignment(rgb, grey, bv, reg):
    s = reg["s"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 6.0))
    ax[0].imshow(rgb); ax[0].set_title(f"A. Thesis Fig. 11B, cropped from page 28\n{rgb.shape[1]}×{rgb.shape[0]} px (page rendered at 144 dpi)")
    ax[1].imshow(bv, cmap="gray"); ax[1].set_title("B. Joanna's vessel image, 10/07/18\n(BV_resized.png, 100×100 = camera grid)")
    # C: plate shrunk to camera px, edges of both overlaid
    t = zoom(grey, 1 / s, order=1)
    hb, ht = MF.highpass(bv), MF.highpass(t)
    canvas = np.full_like(hb, np.nan)
    r0, c0 = int(round(reg["row0"])), int(round(reg["col0"]))
    h, w = min(ht.shape[0], 100 - r0), min(ht.shape[1], 100 - c0)
    canvas[r0:r0 + h, c0:c0 + w] = ht[:h, :w]
    z = lambda m: (m - np.nanmean(m)) / np.nanstd(m)
    over = np.dstack([np.clip(0.5 + 0.25 * z(hb), 0, 1), np.clip(0.5 + 0.25 * np.nan_to_num(z(canvas)), 0, 1),
                      np.clip(0.5 + 0.25 * z(hb), 0, 1)])
    ax[2].imshow(over)
    ax[2].set_title(f"C. Overlay after matching (high-passed)\nmagenta = camera vessels, green = thesis plate\n"
                    f"scale {s:.2f} thesis px per camera px, shift ≈ {reg['row0']:.1f} px, NCC {reg['ncc']:.2f}")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "talk_01a_plate_alignment.png", dpi=150); plt.close(fig)


def fig_points(rgb, red_xy, blue_xy, reg):
    s = reg["s"]
    to_plate = lambda r, c: ((np.asarray(c) - reg["col0"]) * s, (np.asarray(r) - reg["row0"]) * s)
    stim_pts, reg_pts = points.load()
    tbl = reg_pts.assign(row=reg_pts.vsd_y - 1, col=reg_pts.vsd_x - 1)          # (1,1) = top-left, 0-based
    vbar = stim_pts[stim_pts.stimulus.str.contains("upper", case=False)].iloc[[0]]
    vbar = vbar.assign(row=vbar.vsd_y - 1, col=vbar.vsd_x - 1)
    allp = np.r_[tbl[["row", "col"]].to_numpy(), vbar[["row", "col"]].to_numpy()]
    names = list(tbl.stimulus) + ["Vertical bar (upper point)*"]
    marks = np.r_[red_xy, blue_xy]
    kinds = ["red"] * len(red_xy) + ["blue"] * len(blue_xy)

    fig, a = plt.subplots(figsize=(9.5, 9.5))
    a.imshow(rgb); a.set_xticks([]); a.set_yticks([])
    px, py = to_plate(allp[:, 0], allp[:, 1])
    a.scatter(px[:len(tbl)], py[:len(tbl)], s=260, marker="s", facecolors="none", edgecolors=MEAS, lw=2.8,
              label="xlsx registration table (right table)")
    a.scatter(px[len(tbl):], py[len(tbl):], s=260, marker="s", facecolors="none", edgecolors="orange", lw=2.8,
              label="xlsx stimulus table only (vertical bar upper)*")
    for i, (x, y) in enumerate(zip(px[:len(tbl)], py[:len(tbl)])):
        a.text(x + 11, y - 9, str(tbl.point.iloc[i]), color=MEAS, fontsize=14, weight="bold",
               path_effects=[__import__("matplotlib.patheffects").patheffects.withStroke(linewidth=3, foreground="k")])
    lines, ds = [], []
    for (mx, my), k in zip(marks, kinds):
        d = np.hypot(px - mx, py - my) / s; j = int(np.argmin(d)); ds.append(d[j])
        a.plot([mx, px[j]], [my, py[j]], color="k", lw=1.3)
        a.text(mx - 12, my + 26, f"{d[j]:.1f}", fontsize=12, weight="bold", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        lines.append(f"{'red dot ' if k == 'red' else 'blue ✕  '} ↔ {names[j]:30s} {d[j]:.1f} px")
    ds = np.array(ds)
    a.text(6, rgb.shape[0] - 8, "\n".join(lines), fontsize=9, family="monospace", va="bottom",
           bbox=dict(fc="white", ec="none", alpha=0.85))
    a.text(6, 8, "point 1 (triangle top) has no mark in the thesis", fontsize=10, color="k", va="top",
           bbox=dict(fc="white", ec="none", alpha=0.85))
    a.legend(loc="upper right", fontsize=9)
    a.set_title("xlsx points (10/07/18) on thesis Fig. 11B\n"
                f"red dots = scored points, blue ✕ = anchors (thesis) · mean distance {ds.mean():.2f} camera px, "
                f"RMSE {np.sqrt((ds ** 2).mean()):.2f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "talk_01b_points_on_thesis.png", dpi=150); plt.close(fig)
    return ds


def fig_native(rgb, bv):
    """The six xlsx points on 100x100 camera images, no resizing: (x, y) -> row y-1, col x-1."""
    _, reg_pts = points.load()
    rr, cc = reg_pts.vsd_y.to_numpy() - 1, reg_pts.vsd_x.to_numpy() - 1
    import matplotlib.patheffects as pe
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 6.6))
    ax[0].imshow(rgb); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("A. Thesis Fig. 11B (as printed, for comparison)\nred dots = scored points, blue ✕ = anchors")
    for a, img, title in [(ax[1], bv, "B. Joanna's vessel image BV_resized.png\n100×100 camera px, raw, no resizing")]:
        a.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99), interpolation="nearest")
        MF.mark(a, rr, cc, color=MEAS, size=3)
        for i, (r, c) in enumerate(zip(rr, cc)):
            a.text(c + 2.3, r - 1.8, str(reg_pts.point.iloc[i]), color=MEAS, fontsize=13, weight="bold",
                   path_effects=[pe.withStroke(linewidth=3, foreground="k")])
        a.set_xlim(-0.5, 99.5); a.set_ylim(99.5, -0.5)
        a.set_xticks(range(0, 100, 20)); a.set_yticks(range(0, 100, 20))
        a.set_xlabel("camera col (xlsx x − 1)"); a.set_ylabel("camera row (xlsx y − 1)")
        a.set_title(title)
    fig.suptitle("The six xlsx registration points placed directly on the 100×100 camera grid "
                 "\n(VSD (x, y) → row y−1, col x−1; green = the pixel and its 8 neighbours)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "talk_01c_points_on_camera_grid.png", dpi=150); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rgb, grey, red_xy, blue_xy, bv, reg = plate_and_reg()
    fig_native(rgb, bv)
    print("plate", rgb.shape, "reg", reg)
    fig_alignment(rgb, grey, bv, reg)
    print("distances", np.round(fig_points(rgb, red_xy, blue_xy, reg), 2))


if __name__ == "__main__":
    main()
