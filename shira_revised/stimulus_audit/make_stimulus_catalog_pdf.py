"""PDF catalogue of the stimulus images: one row per distinct stimulus (shape + size + colour + location).

Columns: zoomed image (with a 0.25 deg grid; red + = catalog position) | sessions that use it | description from the
catalog (Excel/CSV) | location | size as read.  Same stimulus at two locations -> two rows.
Source: Data/VSD_Encoder_01/stimuli/<monkey>/manifest.parquet + images/.  Output next to the images:
Data/VSD_Encoder_01/stimuli/<monkey>/stimulus_catalog_<monkey>.pdf
Run: python shira_revised/stimulus_audit/make_stimulus_catalog_pdf.py [--monkey gandalf]
"""
import argparse, re, textwrap
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

HERE = Path(__file__).resolve().parent; WS = HERE.parents[2]
PPD = 35.0
ORDER = ["point", "filled_circle", "circle_contour", "triangle_contour", "bar_vertical", "bar_horizontal", "letter", "blank"]
NAMES = {"point": "Point", "filled_circle": "Filled circle", "circle_contour": "Circle contour", "triangle_contour": "Triangle contour",
         "bar_vertical": "Vertical bar", "bar_horizontal": "Horizontal bar", "letter": "Letter", "blank": "Blank"}

def clean(t):
    return re.sub(r"^\s*cond\s*\d+\s*:\s*", "", str(t), flags=re.I).strip()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--monkey", default="gandalf"); args = ap.parse_args()
    root = WS / f"Data/VSD_Encoder_01/stimuli/{args.monkey}"
    m = pd.read_parquet(root / "manifest.parquet")
    m["rgb_s"] = m.rgb.apply(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"({int(v[0])},{int(v[1])},{int(v[2])})")
    m["bg"] = m.background_gray.fillna(128).astype(int)
    key = ["stimulus_uid"]                      # one row per library image
    chk_path = root / f"catalog_check_{args.monkey}.csv"
    chk = pd.read_csv(chk_path).fillna("") if chk_path.exists() else pd.DataFrame(columns=["uid", "message", "accepted", "h5_session", "condition"])
    m["letter"] = m.letter.fillna(""); m["size_deg"] = m.size_deg.fillna(-1)
    groups = []
    for k, g in m.groupby(key, dropna=False, sort=False):
        r = g.iloc[0]
        uses = ", ".join(f"{s} ({c})" for s, c in zip(g.h5_session, g.condition))
        texts = sorted({clean(t) for t in g.stimulus_text})
        notes = sorted({n for n in g.size_note.dropna()})
        flags = chk[chk.uid == r.stimulus_uid]
        flag_lines = sorted({f"{m}" + (" (accepted)" if a else "  <-- NOT ACCEPTED") for m, a in zip(flags.message, flags.accepted)})
        groups.append(dict(shape=r.shape_type, row=r, uses=uses, n=len(g), texts=texts, notes=notes, flags=flag_lines))
    groups.sort(key=lambda d: (ORDER.index(d["shape"]) if d["shape"] in ORDER else 99, d["row"].size_deg, d["row"].pos_x_deg,
                               d["row"].letter, d["row"].rgb_s))

    out = root / f"stimulus_catalog_{args.monkey}.pdf"
    per_page = 4
    with PdfPages(out) as pdf:
        for p0 in range(0, len(groups), per_page):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.05, 0.975, f"Stimulus catalogue — {args.monkey.capitalize()}   (page {p0 // per_page + 1}/{-(-len(groups) // per_page)})",
                     fontsize=12, weight="bold", va="top")
            fig.text(0.05, 0.957, "Images: Data/VSD_Encoder_01/stimuli/%s/images (canvas 210×210 px, 35 px/deg, fixation = top-left corner).\n"
                     "Each image is zoomed, with a 0.25° grid; red + = catalog position. Sizes are diameters of the drawn shape." % args.monkey,
                     fontsize=7, va="top", color="#444", linespacing=1.3)
            for i, d in enumerate(groups[p0:p0 + per_page]):
                r = d["row"]; y0 = 0.93 - (i + 1) * 0.228
                img = np.asarray(Image.open(WS / r.image_path).convert("RGB"))
                ax = fig.add_axes([0.06, y0 + 0.03, 0.27, 0.185])
                cx, cy = r.pos_x_deg * PPD, -r.pos_y_deg * PPD
                sz = r.size_deg if r.size_deg > 0 else 1.0
                half = {"letter": 0.7, "bar_vertical": 0.7, "bar_horizontal": 0.7}.get(r.shape_type, max(0.3, 0.75 * sz)) * PPD
                if r.shape_type == "circle_contour" and r.pos_x_deg == 0 and r.pos_y_deg == 0:
                    cx, cy, half = 0.6 * PPD, 0.6 * PPD, 0.7 * PPD
                if r.shape_type == "blank": cx, cy, half = 0.6 * PPD, 0.75 * PPD, 0.7 * PPD
                ax.imshow(img, interpolation="nearest", extent=(0, img.shape[1] / PPD, -img.shape[0] / PPD, 0))
                cx, cy = max(cx, half), max(cy, half)                      # keep the window on the canvas
                xl = ((cx - half) / PPD, (cx + half) / PPD); yl = (-(cy + half) / PPD, -(cy - half) / PPD)
                ax.set_xlim(*xl); ax.set_ylim(*yl)
                t = np.arange(np.floor(xl[0] * 4) / 4, xl[1] + 1e-9, 0.25); ax.set_xticks(t)
                t = np.arange(np.floor(yl[0] * 4) / 4, yl[1] + 1e-9, 0.25); ax.set_yticks(t)
                ax.grid(color="#ffffff", alpha=0.25, lw=0.5); ax.tick_params(labelsize=6)
                ax.set_xlabel("x (deg)", fontsize=6, labelpad=1); ax.set_ylabel("y (deg)", fontsize=6, labelpad=1)
                if r.shape_type != "blank": ax.plot(r.pos_x_deg, r.pos_y_deg, "+", color="#e34948", ms=8, mew=1.2)
                # text block
                tx = 0.40; ty = y0 + 0.215
                title = NAMES.get(r.shape_type, r.shape_type) + (f" '{r.letter}'" if r.letter else "")
                fig.text(tx, ty, title, fontsize=10, weight="bold", va="top")
                lines = []
                lines.append(("Catalog description", "; ".join(d["texts"])))
                loc = f"x = {r.pos_x_deg:g}°, y = {r.pos_y_deg:g}° (right of VM, below HM < 0)"
                if r.shape_type == "letter": loc += " — centre of the 1°×1° letter box"
                elif r.shape_type not in ("blank",): loc += " — centre of the stimulus"
                lines.append(("Location", loc))
                size = (f"diameter {r.size_deg:g}°" if r.shape_type in ("point", "filled_circle", "circle_contour", "triangle_contour")
                        else f"length {r.size_deg:g}°, width 1 px" if r.shape_type.startswith("bar") else "1° box" if r.shape_type == "letter" else "—")
                if r.shape_type == "triangle_contour": size += " (circumscribed circle; equilateral, tip to the right)"
                if d["notes"]: size += "  [" + "; ".join(n for n in d["notes"] if not n.startswith("bar length")) + "]"
                lines.append(("Size as drawn", size.replace("  []", "")))
                col = r.color + (f", RGB {r.rgb_s}" if r.rgb_s else "") + f", background {r.bg}"
                if r.shape_type == "letter": col = "catalog: white; image: black (letter bitmap as provided)" + f", background {r.bg}"
                lines.append(("Colour", col))
                lines.append((f"Used in ({d['n']})", d["uses"]))
                lines.append(("Library file", f"library/{r.stimulus_uid}.png"))
                yy = ty - 0.022
                for lab, val in lines:
                    fig.text(tx, yy, lab + ":", fontsize=7.2, weight="bold", va="top", color="#333")
                    w = textwrap.wrap(val, 62) or ["—"]
                    fig.text(tx + 0.155, yy, "\n".join(w), fontsize=7.2, va="top", linespacing=1.25)
                    yy -= 0.0135 * max(1, len(w)) + 0.004
                if d["flags"]:
                    fig.text(tx, yy - 0.002, "EXCEL ≠ IMAGE:\n" + "\n".join(textwrap.fill(f, 70) for f in d["flags"]),
                             fontsize=8.5, weight="bold", color="#d00000", va="top", linespacing=1.15)
                fig.add_artist(plt.Line2D([0.05, 0.95], [y0, y0], color="#dddddd", lw=0.6))
            pdf.savefig(fig); plt.close(fig)
    print(f"{len(groups)} distinct stimuli -> {out}")

if __name__ == "__main__":
    main()
