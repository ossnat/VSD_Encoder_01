"""Audit of the stimulus images in Data/VSD_Encoder_01/stimuli/<monkey>/images against the stimulus requirements.

Requirements (from the lab, Nov 2026):
  * one image store: Data/VSD_Encoder_01/stimuli/<monkey>/images
  * position in the catalog = centre of the circle in which the shape is inscribed; every vertex touches that circle;
    the circle size in the catalog is sometimes a radius, sometimes a diameter (text says which)
  * triangle: equilateral, inscribed in that circle
  * straight horizontal / vertical lines (triangle's vertical side, bars, letter strokes) must be perfectly straight
    (one column / row, never broken)
  * letters: taken from the pre-rendered letter files (EncoderData/letters_stimuli/*.bmp), not drawn by us
Also compared with the legacy MATLAB placement (create_letter_stim.m) for letters.
Canvas: 210 x 210 px, 35 px/deg, fixation = top-left corner (pixel (0,0) spans [0,1) x [0,1)).

Run: python tools/stimulus_audit/audit_stimuli.py [--monkey gandalf]
Writes: audit_<monkey>.csv, audit_<monkey>_sheet.png
"""
import argparse, math, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as nd

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]; WS = REPO.parent
PPD = 35.0

def size_reading(text, size):
    """Expected circle radius (deg) from the catalog TEXT (number + 'radius'/'diameter'); contrast circles 'r=..'."""
    t = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(diameter|radius|d/r)", t)
    if m and "d/r" not in t and "diameter/radius" not in t:
        v = float(m.group(1)); return m.group(2), (v if m.group(2) == "radius" else v / 2)
    if "contrast curve" in t and size == size: return "catalog r= (diameter stored)", size / 2
    if "d/r" in t or "diameter/radius" in t: return "unclear (read as diameter)", size / 2
    return "unclear", np.nan

def ink_mask(img, bg=None):
    g = img[..., :3].astype(int)
    bg = int(np.median(g[..., 0])) if bg is None else bg
    return np.any(np.abs(g - bg) > 0, axis=2)

def check(row, img):
    out = dict(session=row.h5_session, condition=row.condition, shape=row.shape_type, text=row.stimulus_text,
               color=row.color, size_deg=row.size_deg, x_deg=row.pos_x_deg, y_deg=row.pos_y_deg)
    m = ink_mask(img); out["n_ink_px"] = int(m.sum())
    vals = np.unique(img[m][:, 0]) if m.any() else []
    out["ink_values"] = ",".join(str(v) for v in vals[:4])
    if row.is_blank:
        out["verdict"] = "OK (blank)" if not m.any() else "ink in blank"; return out, None
    if not m.any():
        out["verdict"] = "EMPTY image"; return out, None
    lab, n = nd.label(m, structure=np.ones((3, 3))); out["n_components"] = int(n)
    ys, xs = np.nonzero(m)
    cx_exp, cy_exp = row.pos_x_deg * PPD, -row.pos_y_deg * PPD            # continuous canvas coords
    out["centre_err_px"] = round(math.hypot(xs.mean() + 0.5 - cx_exp, ys.mean() + 0.5 - cy_exp), 2)
    reading, R_exp = size_reading(row.stimulus_text, row.size_deg)
    out["size_reading"] = reading; out["R_expected_px"] = round(R_exp * PPD, 2) if np.isfinite(R_exp) else np.nan
    issues = []
    sh = row.shape_type
    if sh in ("point", "filled_circle", "circle_contour"):
        h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
        out["drawn_diameter_px"] = f"{w}x{h}"
        if sh in ("point", "filled_circle") and np.isfinite(R_exp) and abs((w + h) / 4 - R_exp * PPD) > 1.0: issues.append("size differs from catalog")
        if sh == "circle_contour" and np.isfinite(R_exp) and abs((w + h) / 4 - R_exp * PPD) > 1.0 and abs(r_cx := 0) == 0 and not (row.pos_x_deg == 0 and row.pos_y_deg == 0): issues.append("size differs from catalog")
        out["R_drawn_px"] = round((w + h) / 4, 2)
        if sh == "circle_contour" and n != 1: issues.append(f"contour broken into {n} pieces")
        if abs(w - h) > 1: issues.append("not round")
    elif sh == "triangle_contour":
        # vertices: right-most pixel (tip) and the top/bottom pixels of the left side
        tip = (xs.max() + 0.5, ys[xs == xs.max()].mean() + 0.5)
        left_col = xs.min(); lv = ys[xs == left_col]
        v_top, v_bot = (left_col + 0.5, lv.min() + 0.5), (left_col + 0.5, lv.max() + 0.5)
        V = np.array([tip, v_top, v_bot])
        sides = [math.dist(V[i], V[j]) for i, j in ((0, 1), (1, 2), (2, 0))]
        out["sides_px"] = "/".join(f"{s:.1f}" for s in sides)
        out["equilateral_err_pct"] = round(100 * (max(sides) - min(sides)) / np.mean(sides), 1)
        cr = [math.hypot(v[0] - cx_exp, v[1] - cy_exp) for v in V]
        out["vertex_radii_px"] = "/".join(f"{r:.1f}" for r in cr)
        out["R_drawn_px"] = round(float(np.mean(cr)), 2)
        # vertical side: every row between the left vertices must have ink in exactly the same left column
        rows = np.arange(ys.min(), ys.max() + 1)          # the vertical side spans the triangle's full height
        leftmost = np.array([xs[ys == r].min() if np.any(ys == r) else -1 for r in rows])
        bad = int(np.sum(leftmost != left_col))
        out["vertical_side"] = "straight" if bad == 0 else f"{bad}/{len(rows)} rows off the column"
        if bad: issues.append("vertical side not straight")
        if n != 1: issues.append(f"contour broken into {n} pieces")
        if out["equilateral_err_pct"] > 8: issues.append("not equilateral")
        if np.isfinite(R_exp) and abs(out["R_drawn_px"] - R_exp * PPD) > 1.5: issues.append("vertices not on the catalog circle")
    elif sh in ("bar_vertical", "bar_horizontal"):
        along, across = (ys, xs) if sh == "bar_vertical" else (xs, ys)
        out["bar_length_px"] = int(np.ptp(along) + 1); out["bar_width_px"] = int(np.ptp(across) + 1)
        # straight: the same set of across-positions on every along-position
        sets = {a: tuple(sorted(across[along == a])) for a in np.unique(along)}
        straight = len(set(sets.values())) == 1
        out["straight"] = straight
        if not straight: issues.append("bar not straight")
        if abs(out["bar_length_px"] - row.size_deg * PPD) > 1: issues.append(f"length {out['bar_length_px']} px vs {row.size_deg * PPD:.0f}")
    elif sh == "letter":
        bmp = Image.open(WS / "Data/EncoderData/letters_stimuli" / f"{row.letter}.bmp")
        idx = np.asarray(bmp)                                   # palette indices: 130 = letter, 129 = background
        glyph = idx[281:316, 381:416] == 130
        # legacy placement (create_letter_stim.m): cols round(35x)-17 .. +17, rows round(35|y|)-17 .. +17 (1-based)
        mround = lambda v: math.floor(v + 0.5)
        c0 = mround(35 * abs(row.pos_x_deg)) - 17 - 1; r0 = mround(35 * abs(row.pos_y_deg)) - 17 - 1
        legacy = np.zeros_like(m); legacy[r0:r0 + 35, c0:c0 + 35] = glyph
        out["letter_pixels_identical_to_bmp"] = bool(np.array_equal(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], glyph[np.ix_(np.any(glyph, 1), np.any(glyph, 0))]))
        out["offset_vs_legacy_px"] = f"dx {xs.min() - (c0 + np.argmax(np.any(glyph, 0)))}, dy {ys.min() - (r0 + np.argmax(np.any(glyph, 1)))}"
        out["iou_vs_legacy"] = round(float((m & legacy).sum() / (m | legacy).sum()), 3)
        pal = bmp.getpalette(); letter_rgb = pal[130 * 3:130 * 3 + 3]
        out["bmp_palette_letter_rgb"] = str(letter_rgb)
        if row.color == "white" and np.all(np.asarray(vals) < 128): issues.append("letter drawn BLACK but catalog / legacy say WHITE")
        if out["offset_vs_legacy_px"] != "dx 0, dy 0": issues.append("1-px offset vs legacy placement")
        legacy_out = legacy
    bgv = int(row.background_gray) if row.background_gray == row.background_gray and row.background_gray is not None else 128
    if row.color == "white" and sh != "letter" and len(vals) and np.all(np.asarray(vals) < bgv): issues.append("darker than background but catalog says white")
    if row.color == "black" and sh != "letter" and len(vals) and np.all(np.asarray(vals) > bgv): issues.append("lighter than background but catalog says black")
    if out["centre_err_px"] > 1.0 and sh != "letter" and not (row.pos_x_deg == 0 and row.pos_y_deg == 0): issues.append(f"centre off by {out['centre_err_px']} px")
    out["verdict"] = "OK" if not issues else "; ".join(issues)
    return out, (legacy_out if sh == "letter" else None)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--monkey", default="gandalf"); args = ap.parse_args()
    root = WS / f"Data/VSD_Encoder_01/stimuli/{args.monkey}"
    man = pd.read_parquet(root / "manifest.parquet")
    dup = man[man.duplicated("image_path", keep=False)]
    rows, sheet = [], []
    for _, r in man.iterrows():
        p = WS / r.image_path
        if not p.is_file():
            rows.append(dict(session=r.h5_session, condition=r.condition, text=r.stimulus_text, verdict="MISSING image")); continue
        img = np.asarray(Image.open(p).convert("RGB"))
        o, extra = check(r, img)
        if r.image_path in set(dup.image_path):
            o["verdict"] = ("SAME FILE as another condition (image overwritten); " + o["verdict"])
        rows.append(o); sheet.append((r, img, o, extra))
    T = pd.DataFrame(rows); T.to_csv(HERE / f"audit_{args.monkey}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(T[["session", "condition", "shape", "size_reading", "R_expected_px", "R_drawn_px", "verdict"]].to_string())
    # contact sheet: one tile per non-blank stimulus, zoomed on the stimulus, with the catalog circle and centre
    S = [s for s in sheet if not s[0].is_blank]
    nc = 8; nr = math.ceil(len(S) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.6 * nr))
    for ax, (r, img, o, extra) in zip(axs.flat, S):
        cx, cy = r.pos_x_deg * PPD, -r.pos_y_deg * PPD
        half = max(12, (r.size_deg or 1) * PPD * (1.2 if o.get("size_reading") == "radius" else 0.7) + 4)
        if r.shape_type == "circle_contour" and r.size_deg > 0.9: half = 40
        ax.imshow(img, interpolation="nearest", extent=(0, img.shape[1], img.shape[0], 0))
        if np.isfinite(o.get("R_expected_px", np.nan)):
            ax.add_patch(plt.Circle((cx, cy), o["R_expected_px"], fill=False, ec="#e34948", lw=1, ls="--"))
        if extra is not None: ax.contour(np.arange(img.shape[1]) + 0.5, np.arange(img.shape[0]) + 0.5, extra, [0.5], colors="#2a78d6", linewidths=0.8)
        ax.plot(cx, cy, "+", color="#e34948", ms=7)
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half); ax.set_xticks([]); ax.set_yticks([])
        ok = o["verdict"].startswith("OK")
        ax.set_title(f"{r.h5_session} {r.condition}\n{r.shape_type} {r.size_deg:g} ({o.get('size_reading', '')})", fontsize=7,
                     color="#1a7f37" if ok else "#c62828")
    for ax in list(axs.flat)[len(S):]: ax.axis("off")
    fig.suptitle(f"Stimulus images ({args.monkey}): red + = catalog position, red dashed = catalog circle "
                 f"(radius or diameter as labelled), blue = legacy letter placement. Title red = issue found.", fontsize=10, x=0.01, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 0.98)); plt.savefig(HERE / f"audit_{args.monkey}_sheet.png", dpi=110)

if __name__ == "__main__":
    main()
