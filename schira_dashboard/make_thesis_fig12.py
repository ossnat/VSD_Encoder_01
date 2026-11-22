"""Thesis Fig. 12 material for the dashboard (run once; ~1 min). Writes schira_dashboard/stimuli/thesis_fig12/:

  cartesian_grid_0-3deg.png, polar_grid_0-3deg.png
      Joanna's grids as dashboard shapes: 210 x 210 canvas (6 deg at 35 px/deg, fixation top-left), grey 130,
      white (255) lines, 0-3 deg. Pixel-identical to the 0-3 deg corner of create_cart_stim.m / create_polar_stim.m
      (ports in src/legacy_joanna/legacy_stimuli.py; the Cartesian port was 1 px off before Nov 17 2026). Not shown to
      the monkey -> kept out of the stimulus library on purpose.
  grids.csv                        uid, label, shape_type, image (read by schira_dash_app.py)
  thesis_<kind>_<date>.png         the chamber plate of thesis Fig. 12 (page 29; kind polar = A, cart = B;
                                   date 201118 = column ii, 100718 = column iii)
  thesis_registration.json         plate -> camera frame (similarity), found by matching the plate's blood vessels to
                                   Joanna's BV_resized.png of that day (normalised cross-correlation, grid lines masked)
  joanna_<kind>_<date>.png         Joanna's prediction re-run with her own pipeline: full 301 x 512 MATLAB grid,
                                   trans_cortex with the retino_mat degree grid, her session parameters and 2-point
                                   registration (src/legacy_joanna/configs/gandalf_legacy.yaml) -> 100 x 100 camera frame
  bv_<date>.png                    Joanna's BV_resized.png (chamber blood-vessel image, camera frame)
  check_thesis_fig12.png           QC: thesis lines vs the re-run, both in the camera frame

Usage:  scripts/py schira_dashboard/make_thesis_fig12.py
"""
from __future__ import annotations

import json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.ndimage import affine_transform, gaussian_filter, median_filter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from src.legacy_joanna.legacy_stimuli import create_cart_stim, create_polar_stim          # noqa: E402
from src.legacy_joanna.legacy_schira import LegacySchiraParams                             # noqa: E402
from src.legacy_joanna.legacy_trans_cortex import m1_m2_from_retino_mat                   # noqa: E402
from src.legacy_joanna.legacy_warp import load_retino_mat_crop                             # noqa: E402
from src.legacy_joanna.legacy_registration import trans_and_register                      # noqa: E402

OUT = HERE / "stimuli/thesis_fig12"
LEG = REPO.parent / "legacy_code/Retinotopic model Joanna/Gandalf"
PAGE = REPO / "schira_dashboard/stimuli/thesis_pages/thesis_page_030.png"     # thesis page 29 (Fig. 12), 144 dpi
PLATES = {("polar", "201118"): (523, 613), ("polar", "100718"): (790, 613),   # top-left (x, y) of the 214 px plates
          ("cart", "201118"): (523, 899), ("cart", "100718"): (790, 899)}
PLATE = 214
GRIDS = [("thesis_grid__cartesian__0-3deg", "cartesian grid · white lines every 0.5° · 0–3° [thesis Fig. 12B(i)]",
          "grid_cartesian", "cartesian_grid_0-3deg.png", "cart"),
         ("thesis_grid__polar__0-3deg", "polar grid · rings every 0.5°, rays every 10° · 0–3° [thesis Fig. 12A(i)]",
          "grid_polar", "polar_grid_0-3deg.png", "polar")]


def canvas(grid105: np.ndarray) -> np.ndarray:
    c = np.full((210, 210, 3), 130, np.uint8)
    c[:105, :105] = grid105
    return c


def register_plate(plate: np.ndarray, bv: np.ndarray) -> dict:
    """Similarity plate -> camera: scale (thesis px per camera px), rotation, and the camera position of the plate's
    top-left corner; full translation search by normalised cross-correlation of high-passed vessel images."""
    from skimage.feature import match_template
    hp = lambda x, a, b: gaussian_filter(x, a) - gaussian_filter(x, b)
    line = (plate > 200) | (plate < 15)
    pf = plate.copy(); pf[line] = median_filter(plate, 9)[line]
    yy, xx = np.mgrid[:PLATE, :PLATE]
    valid = (np.hypot(xx - 106.5, yy - 106.5) < 100) & ~(gaussian_filter(line.astype(float), 1.2) > 0.05)
    B = hp(bv, 0.8, 4); best = None
    for spx in np.arange(2.40, 2.90, 0.025):
        for rot in np.arange(-4, 4.01, 0.5):
            th = np.deg2rad(rot); c, s = np.cos(th), np.sin(th); n = int(PLATE / spx)
            M = spx * np.array([[c, s], [-s, c]])
            tc = affine_transform(hp(pf, 0.8 * spx, 4 * spx), M, output_shape=(n, n), order=1)
            wc = affine_transform(valid.astype(float), M, output_shape=(n, n), order=1) > 0.99
            cc = match_template(B, np.where(wc, tc, 0.0))
            i = np.unravel_index(np.argmax(cc), cc.shape)
            if best is None or cc[i] > best["ncc"]:
                best = dict(ncc=float(cc[i]), thesis_px_per_cam_px=float(spx), rotation_deg=float(rot),
                            col0=float(i[1]), row0=float(i[0]))
    return best


def joanna_prediction(kind: str, date: str, cfg: dict) -> np.ndarray:
    b = cfg[f"joanna_{date}"]
    full = create_polar_stim() if kind == "polar" else create_cart_stim()       # full 301 x 512 MATLAB canvas
    m1, m2 = m1_m2_from_retino_mat(load_retino_mat_crop(), *full.shape[:2])
    p = LegacySchiraParams(a=b["a"], alpha=b["alpha"], k=b["k"], alpha_mode="divide", vf_abs=True)
    r = trans_and_register(full, m1, m2, p, vf_pix_anchors=[tuple(x) for x in b["vf_pix_anchors"]],
                           brain_anchors=[tuple(x) for x in b["brain_anchors"]])
    return r.out_image


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for uid, label, stype, fname, kind in GRIDS:
        g = create_cart_stim(extent_deg=3.0) if kind == "cart" else create_polar_stim(extent_deg=3.0)
        if kind == "cart":   # MATLAB 1-based rows/cols 1, 17, 35, 52, 70, 87, 105
            assert np.where(g[:, 50, 0] == 255)[0].tolist() == [0, 16, 34, 51, 69, 86, 104]
        Image.fromarray(canvas(g)).save(OUT / fname)
        rows.append(dict(uid=uid, label=label, shape_type=stype, image=fname, kind=kind))
    pd.DataFrame(rows).to_csv(OUT / "grids.csv", index=False)

    cfg = yaml.safe_load((REPO / "src/legacy_joanna/configs/gandalf_legacy.yaml").read_text())
    page = np.asarray(Image.open(PAGE).convert("L"))
    reg = {}
    for date in ("201118", "100718"):
        shutil.copyfile(LEG / f"images/{date}/BV_resized.png", OUT / f"bv_{date}.png")
    for (kind, date), (x0, y0) in PLATES.items():
        plate = page[y0:y0 + PLATE, x0:x0 + PLATE]
        Image.fromarray(plate).save(OUT / f"thesis_{kind}_{date}.png")
        bv = np.asarray(Image.open(OUT / f"bv_{date}.png").convert("L")).astype(float)
        reg[f"{kind}_{date}"] = register_plate(plate.astype(float), bv)
        Image.fromarray(joanna_prediction(kind, date, cfg)).save(OUT / f"joanna_{kind}_{date}.png")
        print(f"{kind:5s} {date}: plate registration {reg[f'{kind}_{date}']}", flush=True)
    for date in ("201118", "100718"):
        b = cfg[f"joanna_{date}"]
        reg[f"joanna_{date}"] = dict(a=b["a"], alpha_thesis=b["alpha"], k=b["k"], mean_rmsd_px=b.get("mean_rmsd_px"),
                                     vf_pix_anchors=b["vf_pix_anchors"], brain_anchors=b["brain_anchors"])
    (OUT / "thesis_registration.json").write_text(json.dumps(reg, indent=1))

    # QC figure: thesis lines (registered) vs the re-run of Joanna's pipeline, camera frame
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE)); import thesis_fig12 as T   # noqa: E402
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    for a, (kind, date) in zip(ax, PLATES):
        cen, rad = T.chamber_circle(kind, date)
        th = T.thesis_lines_cam(kind, date); jo = T.inside(T.joanna_lines_cam(kind, date), cen, rad)
        a.imshow(T.bv(date), cmap="gray", extent=(-0.5, 99.5, 99.5, -0.5))
        a.plot(th.real, th.imag, ".", ms=1.5, color="#e34948", label="thesis Fig. 12 (registered)")
        a.plot(jo.real, jo.imag, ".", ms=1.5, color="#17becf", label="Joanna's pipeline, re-run")
        d = T.chamfer(th, jo)
        a.set_title(f"{kind} {date}: thesis→re-run {d[0]:.2f} px, re-run→thesis {d[1]:.2f} px", fontsize=9)
        a.set_xlim(-0.5, 99.5); a.set_ylim(99.5, -0.5)
    ax[0].legend(loc="lower left", fontsize=7, markerscale=5)
    plt.tight_layout(); plt.savefig(OUT / "check_thesis_fig12.png", dpi=90)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
