"""Check Joanna's 10/07/18 point tables (points.py) against the stimuli, the VSD maps and thesis Fig. 11B.

Figures (registration_revised/figs/):
  stimulus_points_100718{a,b}.png   per condition: stimulus with the VF point(s) in red | VSD response map with the
                                    VSD point(s) in red, read with (1,1) at the top-left corner | same, top-right corner
  registration_points_vessels.png   the 6 registration points on high-passed vessel images (top-left and top-right
                                    readings), next to thesis Fig. 11B, whose red dots / blue crosses are registered
                                    into the camera frame and drawn as rings
  origin_scores.csv                 how well each reading fits: response percentile at the stimulus points, and
                                    distance of the registration points to the thesis Fig. 11B marks

Camera frame: 100 x 100, 0-based (row, col). A VSD point (x, y) as written (1-based) is read as
  top-left : row = y - 1, col = x - 1          top-right: row = y - 1, col = 100 - x
(also scored: the same two with x/y swapped).

Vessel image: the blank trials of 10/07/18 are not on this machine and the condsAN trials are already divided by their
baseline (values ~1.0), so the vessels are mostly gone from them. Shown instead: (1) Joanna's BV_resized.png (chamber
vessel image, camera frame, the image of thesis Fig. 11) high-passed, and (2) the mean of frames 1-150 over all condsAN
trials of each session, high-passed, to show what is left. With the blank .mat files, swap (2) for the blank trials.

Run:  scripts/py registration_revised/make_figs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, label, center_of_mass, median_filter, zoom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import points  # noqa: E402

REPO = HERE.parent
DATA = REPO.parent / "Data"
LEG = REPO.parent / "legacy_code/Retinotopic model Joanna/Gandalf"
FIGS = HERE / "figs"
SESSIONS = ("100718a", "100718b")
MAPS = REPO / "schira_dashboard/cache/gandalf_maps.npz"          # mean(frames 35-46) - mean(frames 5-25), per condition
CATALOG = REPO / "schira_dashboard/cache/gandalf_catalog.csv"
THESIS_PAGE = REPO / "schira_dashboard/stimuli/thesis_pages/thesis_page_029.png"   # thesis page 28 (Figs. 10-11)
FIG11B = (630, 605, 389)                                          # x0, y0, size of the Fig. 11B plate on that page
READINGS = {
    "top-left":           lambda x, y: (y - 1, x - 1),
    "top-right":          lambda x, y: (y - 1, 100 - x),
    "top-left, x/y swapped":  lambda x, y: (x - 1, y - 1),
    "top-right, x/y swapped": lambda x, y: (x - 1, 100 - y),
}
SHOWN = ("top-left", "top-right")


# ---------------------------------------------------------------- helpers
def mark(ax, rows, cols, color="red", size=3):
    """The pixel and its 8 neighbours, as a filled square (data coords = pixel centres)."""
    for r, c in zip(np.atleast_1d(rows), np.atleast_1d(cols)):
        ax.add_patch(plt.Rectangle((c - size / 2, r - size / 2), size, size, color=color, lw=0))


def highpass(img, sigma=4.0, smooth=0.7):
    img = np.where(np.isfinite(img), img, np.nanmedian(img))
    return gaussian_filter(img, smooth) - gaussian_filter(img, sigma)


def pct_at(m, rows, cols):
    """Percentile of the smoothed map at each point (within the finite pixels of the map)."""
    s = gaussian_filter(np.nan_to_num(m, nan=np.nanmedian(m)), 1.5)
    fin = s[np.isfinite(m)]
    out = []
    for r, c in zip(rows, cols):
        r, c = int(round(r)), int(round(c))
        out.append(100.0 * (fin < s[r, c]).mean() if 0 <= r < 100 and 0 <= c < 100 else np.nan)
    return np.array(out)


def mean_early_frames(session, frames=(0, 150)):
    acc, n = np.zeros(10000), 0
    with h5py.File(DATA / f"FoundationData/ProcessedData/gandalf/session_{session}_condsAN.h5", "r") as f:
        meta = json.loads(f.attrs["trial_metadata_json"])
        for i in range(len(meta)):
            d = f[f"trial_{i:06d}"][:, frames[0]:frames[1]].astype(float)
            d[~np.isfinite(d) | (np.abs(d) > 1e3)] = np.nan
            v = np.nanmean(d, 1)
            ok = np.isfinite(v); acc[ok] += v[ok]; n += 1
    return (acc / n).reshape(100, 100)


def fig11b():
    """Thesis Fig. 11B plate, its red dots and blue crosses (plate pixel x, y)."""
    x0, y0, n = FIG11B
    rgb = np.asarray(Image.open(THESIS_PAGE).convert("RGB")).astype(float)[y0:y0 + n, x0:x0 + n]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = (R > 150) & (G < 110) & (B < 110)
    blue = (B > 140) & (R < 110) & (B - R > 60)
    def blobs(m, min_px=4):
        lab, k = label(m)
        cs = [center_of_mass(m, lab, i + 1) for i in range(k) if (lab == i + 1).sum() >= min_px]
        return np.array([(c[1], c[0]) for c in cs])      # (x, y)
    grey = rgb.mean(-1)
    marks = gaussian_filter((red | blue).astype(float), 2) > 0.02
    grey = np.where(marks, median_filter(grey, 11), grey)
    return rgb.astype(np.uint8), grey, blobs(red), blobs(blue)


def register_plate(grey, bv, scales=np.arange(3.4, 4.41, 0.02)):
    """Plate (thesis px) -> camera: scale s (thesis px per camera px) and the camera position (row0, col0) of the plate's
    top-left corner; no rotation (the plate is the same vessel image). Normalised cross-correlation of high-passed images."""
    from skimage.feature import match_template
    B = np.pad(highpass(bv, 4, 0.7), 20, mode="constant")
    best = None
    for s in scales:
        t = zoom(grey, 1 / s, order=1)
        t = highpass(t, 4, 0.7)[8:-8, 8:-8]                 # drop the plate border
        cc = match_template(B, t)
        i = np.unravel_index(np.argmax(cc), cc.shape)
        if best is None or cc[i] > best["ncc"]:
            best = dict(ncc=float(cc[i]), s=float(s), row0=float(i[0] - 20 - 8 + 0.5 - 0.5 / s),
                        col0=float(i[1] - 20 - 8 + 0.5 - 0.5 / s))
    return best


def plate_to_cam(reg, xy):
    xy = np.atleast_2d(xy)
    return reg["row0"] + xy[:, 1] / reg["s"], reg["col0"] + xy[:, 0] / reg["s"]     # rows, cols


# ---------------------------------------------------------------- main
def main():
    FIGS.mkdir(exist_ok=True)
    stim_pts, reg_pts = points.load()
    maps = np.load(MAPS)
    cat = pd.read_csv(CATALOG)
    scores = []

    # 1) stimulus points: stimulus | VSD map (top-left) | VSD map (top-right)
    for s in SESSIONS:
        conds = sorted(stim_pts.cond.unique())
        fig, axes = plt.subplots(len(conds), 3, figsize=(12, 3.7 * len(conds)))
        for ax_row, c in zip(axes, conds):
            P = stim_pts[stim_pts.cond == c]
            st = cat[(cat.session == s) & (cat.condition == f"condAN{c}")].iloc[0]
            img = np.asarray(Image.open(DATA.parent / st.image_path).convert("RGB"))
            lim = 60                                                       # canvas px shown (0-1.7 deg)
            a = ax_row[0]; a.imshow(img[:lim, :lim], interpolation="nearest")
            mark(a, P.vf_y_px.values, P.vf_x_px.values)
            a.set_title(f"cond {c}: {st.stimulus_text.split(': ', 1)[-1]}\nxlsx: {P.stimulus.iloc[0].split(' (')[0]}"
                        f" · VF px (x,y) {', '.join(f'({x:g},{y:g})' for x, y in zip(P.vf_x_px, P.vf_y_px))}",
                        fontsize=9, loc="left")
            a.set_xlabel("canvas col (35 px/°, fixation = top-left corner)"); a.set_ylabel("canvas row")
            m = maps[f"{s}|condAN{c}"]
            lo, hi = np.nanpercentile(m, [2, 99.5])
            for a, rd in zip(ax_row[1:], SHOWN):
                rr, cc = READINGS[rd](P.vsd_x.values, P.vsd_y.values)
                p = pct_at(m, rr, cc)
                a.imshow(gaussian_filter(np.nan_to_num(m, nan=np.nanmedian(m)), 1.0), cmap="jet", vmin=lo, vmax=hi)
                mark(a, rr, cc)
                a.set_title(f"{s} VSD, (1,1) = {rd}\nresponse percentile at the point(s): "
                            + ", ".join(f"{v:.0f}" for v in p), fontsize=9, loc="left")
                a.set_xlabel("camera col"); a.set_ylabel("camera row")
            for rd, f in READINGS.items():
                rr, cc = f(P.vsd_x.values, P.vsd_y.values)
                for (_, q), v in zip(P.iterrows(), pct_at(m, rr, cc)):
                    scores.append(dict(test="stimulus point: response percentile (higher = better)", session=s,
                                       reading=rd, item=f"cond {c} {q.stimulus}", value=v))
        fig.suptitle(f"Joanna's stimulus points, session {s} (opt-points-10-07-18.xlsx, left table)\nred = the point and "
                     "its 8 neighbours · VSD map: mean(frames 35-46) - mean(5-25), lightly smoothed", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(FIGS / f"stimulus_points_{s}.png", dpi=90); plt.close(fig)

    # 2) registration points on vessel images, vs thesis Fig. 11B
    bv = np.asarray(Image.open(LEG / "images/100718/BV_resized.png").convert("L")).astype(float)
    rgb, grey, red_xy, blue_xy = fig11b()
    reg = register_plate(grey, bv)
    th_red = plate_to_cam(reg, red_xy) if len(red_xy) else (np.array([]), np.array([]))
    th_blue = plate_to_cam(reg, blue_xy) if len(blue_xy) else (np.array([]), np.array([]))
    th_all = np.c_[np.r_[th_red[0], th_blue[0]], np.r_[th_red[1], th_blue[1]]]
    for rd, f in READINGS.items():
        rr, cc = f(reg_pts.vsd_x.values, reg_pts.vsd_y.values)
        for (_, q), r_, c_ in zip(reg_pts.iterrows(), rr, cc):
            d = np.hypot(th_all[:, 0] - r_, th_all[:, 1] - c_).min() if len(th_all) else np.nan
            scores.append(dict(test="registration point: distance to nearest thesis Fig. 11B mark, camera px (lower = better)",
                               session="100718", reading=rd, item=f"point {q.point} {q.stimulus}", value=d))
    early = {s: mean_early_frames(s) for s in SESSIONS}

    fig, ax = plt.subplots(2, 3, figsize=(16, 11))
    ax[0, 0].imshow(rgb); ax[0, 0].set_title("thesis Fig. 11B (10/07/18): red dots = RMSD points, blue crosses = "
                                             "anchor points", fontsize=9, loc="left")
    def overlay(a, base, title, rd, cmap="gray"):
        a.imshow(base, cmap=cmap, vmin=np.percentile(base, 1), vmax=np.percentile(base, 99))
        rr, cc = READINGS[rd](reg_pts.vsd_x.values, reg_pts.vsd_y.values)
        mark(a, rr, cc)
        for i, (r_, c_) in enumerate(zip(rr, cc)):
            a.text(c_ + 2.2, r_ - 1.5, str(reg_pts.point.iloc[i]), color="red", fontsize=10, weight="bold")
        extra = stim_pts[~stim_pts.set_index(["vsd_x", "vsd_y"]).index.isin(reg_pts.set_index(["vsd_x", "vsd_y"]).index)]
        er, ec = READINGS[rd](extra.vsd_x.values, extra.vsd_y.values)
        a.scatter(ec, er, s=40, marker="s", facecolors="none", edgecolors="orange", lw=1.5,
                  label="other left-table points (e.g. vertical bar upper)")
        a.scatter(th_red[1], th_red[0], s=110, facecolors="none", edgecolors="yellow", lw=1.6, label="thesis red dot")
        a.scatter(th_blue[1], th_blue[0], s=110, facecolors="none", edgecolors="cyan", lw=1.6, label="thesis blue cross")
        a.set_title(title, fontsize=9, loc="left"); a.set_xlim(-0.5, 99.5); a.set_ylim(99.5, -0.5)
    for a, rd in zip(ax[0, 1:], SHOWN):
        dd = [r["value"] for r in scores if r["reading"] == rd and r["test"].startswith("registration")]
        overlay(a, highpass(bv), f"Joanna's BV_resized.png, high-passed · xlsx points read with (1,1) = {rd}\n"
                                 f"mean distance to the thesis marks: {np.nanmean(dd):.1f} px", rd)
    ax[0, 1].legend(loc="lower left", fontsize=8)
    best = min(SHOWN, key=lambda rd: np.nanmean([r["value"] for r in scores if r["reading"] == rd
                                                and r["test"].startswith("registration")]))
    for a, s in zip(ax[1, :2], SESSIONS):
        overlay(a, highpass(early[s], 3, 0.7), f"{s}: mean of frames 1-150, all condsAN trials,\nhigh-passed "
                                               f"(no blank trials here) · points read with (1,1) = {best}", best)
    a = ax[1, 2]
    a.imshow(highpass(bv), cmap="gray", vmin=np.percentile(highpass(bv), 1), vmax=np.percentile(highpass(bv), 99))
    a.imshow(np.ma.masked_less(gaussian_filter(maps["100718b|condAN5"], 1), np.nanpercentile(maps["100718b|condAN5"], 90)),
             cmap="autumn", alpha=0.5)
    rr, cc = READINGS[best](stim_pts.vsd_x.values, stim_pts.vsd_y.values); mark(a, rr, cc)
    a.set_title(f"all left-table VSD points ({best}) + top 10% of the\n100718b vertical-bar response, on the vessel image", fontsize=9, loc="left")
    fig.suptitle(f"10/07/18 registration points (xlsx right table, red squares numbered as in the table)\nthesis plate "
                 f"registered to BV_resized.png: {reg['s']:.2f} thesis px per camera px, plate corner at camera "
                 f"(row {reg['row0']:.1f}, col {reg['col0']:.1f}), NCC {reg['ncc']:.2f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "registration_points_vessels.png", dpi=90); plt.close(fig)

    S = pd.DataFrame(scores)
    S.to_csv(FIGS / "origin_scores.csv", index=False)
    summ = S.groupby(["test", "session", "reading"]).value.mean().unstack("reading")
    print(summ.round(1).to_string())
    print(f"\nthesis Fig. 11B: {len(red_xy)} red dots, {len(blue_xy)} blue crosses; plate registration {reg}")


if __name__ == "__main__":
    main()
