"""Build the dashboard cache for one monkey (run once, or again after new sessions are added).

Writes to schira_dashboard/cache/:
  <monkey>_maps.npz        response map per (session, condition): mean(frames 35-46) - mean(frames 5-25), 100x100, dF/F
  <monkey>_catalog.csv     stimulus per (session, condition): shape, size, position (deg), text, image path
  <monkey>_landmarks.csv   landmark table: camera (row, col; 0-based) <-> visual field (x, y deg), with source and quality

Landmarks
  auto   : peak of the smoothed response for points / circles (-> stimulus centre) and the two strongest peaks for 1 deg
           bars (-> the two bar ends; which peak is which end is left open, pair_id links them and the fit tries both).
           One row per (session, condition); quality = peak height / robust SD of the map.
  manual : Joanna's hand-picked points: 20/11/18 letters (opt_points_8_5_22.mat, 8 points, day-level: attached to every
           201118 session) and 10/07/18 bars (opt-points-10-07-18.xlsx, 10 points, attached to 100718a and 100718b).
Not detected yet: triangle corners, the 0.95 deg circle around fixation, letters.

Usage:  python schira_dashboard/prepare_data.py [--monkey gandalf]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import gaussian_filter, maximum_filter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO.parent / "Data"
LEGACY = REPO.parent / "legacy_code/Retinotopic model Joanna/Gandalf"
WINDOW, BASELINE = (35, 46), (5, 25)
PPD = 35.0


def response_maps(h5_path: Path) -> dict[str, np.ndarray]:
    """Trial-average per condition, then window - baseline. Bad pixels (non-finite or |x|>1e3) are skipped per trial."""
    acc: dict[str, list] = {}
    with h5py.File(h5_path, "r") as f:
        meta = json.loads(f.attrs["trial_metadata_json"])
        for i, m in enumerate(meta):
            d = f[f"trial_{i:06d}"][...].astype(np.float64)          # (10000, frames)
            ok = np.isfinite(d) & (np.abs(d) < 1e3)
            win = np.where(ok[:, WINDOW[0]:WINDOW[1]], d[:, WINDOW[0]:WINDOW[1]], np.nan)
            base = np.where(ok[:, BASELINE[0]:BASELINE[1]], d[:, BASELINE[0]:BASELINE[1]], np.nan)
            with np.errstate(all="ignore"):
                r = np.nanmean(win, 1) - np.nanmean(base, 1)
            s = acc.setdefault(m["condition"], [np.zeros(10000), np.zeros(10000), 0])
            good = np.isfinite(r); s[0][good] += r[good]; s[1][good] += 1; s[2] += 1
    out = {}
    for c, (s, k, n) in acc.items():
        m = s / np.maximum(k, 1); m[k < 0.5 * n] = np.nan
        out[c] = m.reshape(100, 100).astype(np.float32)
    return out


def peaks(r: np.ndarray, n: int, sigma=2.0, min_dist=15, border=8):
    """n strongest local maxima of the smoothed map, at least min_dist px apart. Returns [(row, col, height)]."""
    nanm = ~np.isfinite(r)
    s = gaussian_filter(np.nan_to_num(r), sigma); s[nanm] = -np.inf
    s[:border] = s[-border:] = -np.inf; s[:, :border] = s[:, -border:] = -np.inf
    loc = (s == maximum_filter(s, size=7)) & np.isfinite(s)
    cand = np.argwhere(loc); cand = cand[np.argsort(-s[loc])]
    out = []
    for p in cand:
        if all(np.hypot(*(p - q[:2])) >= min_dist for q in out):
            out.append((float(p[0]), float(p[1]), float(s[tuple(p)])))
        if len(out) == n:
            break
    fin = s[np.isfinite(s)]
    sd = 1.4826 * np.median(np.abs(fin - np.median(fin)))
    return out, sd


def manual_landmarks(sessions: list[str]) -> list[dict]:
    rows = []
    mat = LEGACY / "VSDI points/201118/opt_points_8_5_22.mat"
    if mat.is_file():
        P = sio.loadmat(mat)["opt_points"].astype(float)   # [stim row, stim col, 0, 0, cam row, cam col], 1-based
        for s in [s for s in sessions if s.startswith("201118")]:
            for i, p in enumerate(P):
                rows.append(dict(session=s, condition="", source="manual", label=f"Joanna 201118 #{i+1}",
                                 x_deg=p[1] / PPD, y_deg=-p[0] / PPD, cam_row=p[4] - 1, cam_col=p[5] - 1,
                                 pair_id="", quality=np.nan, note="day-level points (session within 20/11/18 not recorded)"))
    # opt-points-10-07-18.xlsx: ((stim col, stim row), (cam col, cam row)), 1-based
    pts = [((21, 26), (44, 39)), ((14, 14), (21, 36)), ((14, 38), (60, 30)), ((35, 26), (47, 45)), ((21, 14), (25, 41)),
           ((21, 37), (58, 36)), ((21, 9), (24, 46)), ((21, 43), (70, 36)), ((4, 26), (37, 24)), ((38, 26), (51, 48))]
    # last point (horizontal-bar right): the xlsx has (48, 51); Joanna's .mat files and thesis Fig. 11B have x/y the other
    # way round (registration_revised/README.md)
    for s in [s for s in sessions if s.startswith("100718")]:
        for i, (st, cm) in enumerate(pts):
            rows.append(dict(session=s, condition="", source="manual", label=f"Joanna 100718 #{i+1}",
                             x_deg=st[0] / PPD, y_deg=-st[1] / PPD, cam_row=cm[1] - 1, cam_col=cm[0] - 1,
                             pair_id="", quality=np.nan, note="day-level points (opt-points-10-07-18.xlsx)"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--monkey", default="gandalf")
    args = ap.parse_args()
    monkey = args.monkey.lower()

    man = pd.read_parquet(DATA / f"VSD_Encoder_01/encoding_pairs/{monkey}/win_0035_0046/manifest.parquet")
    cat = (man.drop_duplicates(["date", "condition"])
              [["date", "condition", "shape_type", "color", "size_deg", "pos_x_deg", "pos_y_deg", "stimulus_text", "image_path"]]
              .rename(columns={"date": "session"}).reset_index(drop=True))
    sessions = sorted(cat.session.unique())

    maps, lm = {}, []
    for s in sessions:
        h5 = DATA / f"FoundationData/ProcessedData/{monkey}/session_{s}_condsAN.h5"
        if not h5.is_file():
            print(f"skip {s}: no {h5.name}"); continue
        R = response_maps(h5)
        for c, r in R.items():
            if not ((cat.session == s) & (cat.condition == c)).any():
                continue
            maps[f"{s}|{c}"] = r
            st = cat[(cat.session == s) & (cat.condition == c)].iloc[0]
            shape, x0, y0, size = st.shape_type, float(st.pos_x_deg), float(st.pos_y_deg), float(st.size_deg)
            if shape in ("point", "filled_circle") or (shape == "circle_contour" and size < 0.5):
                (pk,), sd = peaks(r, 1)
                lm.append(dict(session=s, condition=c, source="auto", label=f"{shape} centre", x_deg=x0, y_deg=y0,
                               cam_row=pk[0], cam_col=pk[1], pair_id="", quality=pk[2] / sd, note=""))
            elif shape in ("bar_vertical", "bar_horizontal"):
                pks, sd = peaks(r, 2)
                half = size / 2
                ends = [(x0, y0 + half), (x0, y0 - half)] if shape == "bar_vertical" else [(x0 - half, y0), (x0 + half, y0)]
                for pk, (xe, ye) in zip(pks, ends):    # provisional pairing; the fit tries both pairings (pair_id)
                    lm.append(dict(session=s, condition=c, source="auto", label=f"{shape} end", x_deg=xe, y_deg=ye,
                                   cam_row=pk[0], cam_col=pk[1], pair_id=f"{s}|{c}", quality=pk[2] / sd, note="end order unknown"))
        print(f"{s}: {len(R)} conditions", flush=True)

    # blank trials (session_<s>_blank.h5, made by scripts/extract_blank_trials_h5.py from the raw condsAN .mat 'blankAN'):
    # same window - baseline map, key '<session>|blank'; no catalog row, no landmarks
    for h5 in sorted((DATA / f"FoundationData/ProcessedData/{monkey}").glob("session_*_blank.h5")):
        s = h5.stem[len("session_"):-len("_blank")]
        R = response_maps(h5)
        for c, r in R.items():
            maps[f"{s}|blank"] = r
        print(f"{s}: blank trials ({', '.join(R)})", flush=True)

    lm += manual_landmarks(sessions)
    HERE.joinpath("cache").mkdir(exist_ok=True)
    np.savez_compressed(HERE / f"cache/{monkey}_maps.npz", **maps)
    cat.to_csv(HERE / f"cache/{monkey}_catalog.csv", index=False)
    L = pd.DataFrame(lm); L.to_csv(HERE / f"cache/{monkey}_landmarks.csv", index=False)
    print(f"wrote {len(maps)} maps, {len(cat)} catalog rows, {len(L)} landmarks "
          f"({(L.source == 'auto').sum()} auto, {(L.source == 'manual').sum()} manual)")


if __name__ == "__main__":
    main()
