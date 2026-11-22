#!/usr/bin/env python3
"""Check every stimulus image against the INSTRUCTIONS in the Excel/CSV catalogs.

Independent of the catalog parser in src/stimuli: the CSV text is re-read here and turned into expectations
(shape, colour, size with its radius/diameter wording, position, background), which are compared with what is
actually drawn in the library image the condition points to. Any difference is printed in LARGE RED letters,
written to stimuli/<monkey>/catalog_check_<monkey>.csv, and shown in red in the PDF catalogue.

Usage:  scripts/py scripts/01d_check_stimuli_against_catalog.py [--monkey gandalf]
Exit code 1 if a mismatch was found that is not on the accepted list below.
"""
from __future__ import annotations

import argparse, math, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]; WS = REPO.parent
PPD = 35.0
RED, BOLD, YEL, END = "\033[91m", "\033[1m", "\033[93m", "\033[0m"

# Differences the lab has looked at and accepted (key: (h5_session, condition, check)).
ACCEPTED = {
    "letter_colour": "letters come from the legacy bitmaps; colour left as is (user + PI, 15/11/2026)",
    "excel_cond6_bar": "24/7/2018: Excel lists the horizontal bar as cond6 next to 'Cond 6 Blank'; recordings have condAN7 (user, 15/11/2026)",
    "size_wording_unclear": "29/5/2018 sizes 'd/r?': read as diameter like the majority (user + PI, 15/11/2026)",
}


def big_red(lines: list[str]) -> str:
    bar = "█" * 78
    body = "\n".join(f"██  {l:<72}██" for l in lines)
    return f"{RED}{BOLD}{bar}\n██{'':74}██\n{body}\n██{'':74}██\n{bar}{END}"


# ------------------------------------------------------------------ expectations from the raw CSV text
def h5(date: str, letter: str) -> str:
    d, m, y = [int(x) for x in re.split(r"[/.]", date.strip())]
    return f"{d:02d}{m:02d}{y % 100:02d}{letter.strip().lower()}"


def parse_shapes_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    out, date, sess, pos = [], None, None, None
    for _, r in df.iterrows():
        if isinstance(r.get("Date"), str) and r["Date"].strip():
            date, sess, pos = r["Date"].strip(), str(r.get("Session") or ""), None
        if isinstance(r.get("Stimulus Position"), str) and r["Stimulus Position"].strip():
            x, y = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", r["Stimulus Position"])[:2]]
            pos = (x, y)
        t = r.get("stimulus (need to check r/d)")
        if not isinstance(t, str) or not t.strip():
            continue
        t = t.strip(); low = t.lower()
        m = re.match(r"cond\s*(\d+)", low)
        cond = int(m.group(1)) if m else None
        e = dict(source=path.name, date=date, text=t, cond=cond, x=pos[0], y=pos[1], rgb=None, background=None)
        if "blank" in low:
            e.update(shape="blank")
        else:
            e["colour"] = "white" if "white" in low else "black"
            for key, sh in (("triangle", "triangle_contour"), ("contour", "circle_contour"), ("filled circle", "filled_circle"),
                            ("point", "point"), ("bar vertical", "bar_vertical"), ("bar horizontal", "bar_horizontal")):
                if key in low:
                    e["shape"] = sh; break
            num = re.search(r"(\d+(?:\.\d+)?)\s*(diameter/radius\?|d/r\?|diameter|radius|deg)?", low.split(":", 1)[-1])
            if e["shape"].startswith("bar"):
                e.update(length=1.0)
            elif num:
                v, unit = float(num.group(1)), (num.group(2) or "")
                if unit in ("radius",):
                    e.update(R=v, size_wording="radius")
                elif unit == "diameter":
                    e.update(R=v / 2, size_wording="diameter")
                else:
                    e.update(R=v / 2, size_wording="unclear")
        for L in [s.strip() for s in sess.split(",") if s.strip()]:
            out.append(dict(e, h5_session=h5(date, L)))
    return out


def parse_contrast_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    # the 'Cond1..Cond8' names are in the second header row, under 'Contrast (RGB)' / 'Unnamed: n'
    first = df.iloc[0]
    df = df.rename(columns={c: first[c] for c in df.columns if isinstance(first[c], str) and first[c].startswith("Cond")}).iloc[1:]
    out, date = [], None
    for _, r in df.iterrows():
        if isinstance(r.get("Date"), str) and re.match(r"\d", r["Date"].strip()):
            date = r["Date"].strip()
            if len(date.split("/")[-1]) == 2: date = date[:-2] + "20" + date[-2:]
        s, par = r.get("Session"), r.get("Paradigm")
        if not (isinstance(s, str) and s.strip() and isinstance(par, str) and par.strip()):
            continue
        if "control attention" in par.lower():
            continue
        y, x = [float(v.replace(" ", "")) for v in re.findall(r"[-+]?\s*\d+(?:\.\d+)?", r["Target Location (below HM; from VM)"])[:2]]
        sess = h5(date, s)
        blank = next((c for c in r.values if isinstance(c, str) and "blank" in c.lower()), "")
        bgm = re.search(r"\((\d+)\)", blank); bg = int(bgm.group(1)) if bgm else None
        size = str(r["Target Size (diameter in deg)"])
        for k in range(1, 9):
            cell = r.get(f"Cond{k}")
            if not isinstance(cell, str) or not cell.strip() or cell.strip().lower() in ("error", "nan"):
                continue
            base = dict(source=path.name, date=date, h5_session=sess, cond=k, x=x, y=y, background=bg, text=f"{par}: Cond{k} = {cell.strip()}")
            if "blank" in cell.lower():
                out.append(dict(base, shape="blank")); continue
            if "letter" in par.lower():
                out.append(dict(base, shape="letter", letter=cell.strip().upper(), colour="white", box=float(re.findall(r"\d+(?:\.\d+)?", size)[0])))
                continue
            g = int(re.search(r"\((\d+)\)", cell).group(1))
            rm = re.search(r"r\s*=\s*(\d+(?:\.\d+)?)", size)
            out.append(dict(base, shape="filled_circle", rgb=g, colour="white" if bg is not None and g > bg else "black",
                            R=float(rm.group(1)), size_wording="radius (r=)"))
    return out


# ------------------------------------------------------------------ measurements on the image
def measure(img: np.ndarray, e: dict) -> dict:
    g = img[..., 0].astype(int)
    bg = int(np.bincount(g.ravel()).argmax())
    ink = g != bg; ys, xs = np.nonzero(ink)
    m = dict(bg=bg, n_ink=int(ink.sum()))
    if not ink.any():
        return m
    m["ink_values"] = sorted(set(g[ink].tolist()))
    cx, cy = e["x"] * PPD, -e["y"] * PPD
    m["centre_err_px"] = math.hypot((xs.min() + xs.max() + 1) / 2 - cx, (ys.min() + ys.max() + 1) / 2 - cy)
    sh = e.get("shape")
    if sh in ("point", "filled_circle", "circle_contour"):
        if e["x"] == 0 and e["y"] == 0:      # circle around fixation: only its lower-right quarter is on the canvas
            d = np.hypot(xs + 0.5, ys + 0.5); m["R_px"] = float(np.median(d)); m["centre_err_px"] = 0.0
        else:
            m["R_px"] = ((xs.max() - xs.min() + 1) + (ys.max() - ys.min() + 1)) / 4
    elif sh == "triangle_contour":
        tip = np.array([xs.max() + 0.5, ys[xs == xs.max()].mean() + 0.5])
        lc = xs.min(); top = np.array([lc + 0.5, ys.min() + 0.5]); bot = np.array([lc + 0.5, ys.max() + 0.5])
        V = [tip, top, bot]; c = np.array([cx, cy])
        m["R_px"] = float(np.mean([np.linalg.norm(v - c) for v in V]))
        sides = [np.linalg.norm(V[i] - V[j]) for i, j in ((0, 1), (1, 2), (2, 0))]
        m["equilateral_err_pct"] = 100 * (max(sides) - min(sides)) / np.mean(sides)
        rows = range(ys.min(), ys.max() + 1)
        m["vertical_side_straight"] = all(xs[ys == r].min() == lc for r in rows if np.any(ys == r))
        m["centre_err_px"] = 0.0     # a triangle's bbox centre is not its circumcentre; the vertex radii test the position
        m["vertex_radius_spread_px"] = float(np.ptp([np.linalg.norm(v - c) for v in V]))
    elif sh in ("bar_vertical", "bar_horizontal"):
        along, across = (ys, xs) if sh == "bar_vertical" else (xs, ys)
        m["length_px"] = int(np.ptp(along) + 1); m["width_px"] = int(np.ptp(across) + 1)
        m["straight"] = len({tuple(sorted(across[along == a])) for a in np.unique(along)}) == 1
    elif sh == "letter":
        m["box_px"] = (int(np.ptp(xs) + 1), int(np.ptp(ys) + 1))
    return m


def compare(e: dict, m: dict) -> list[tuple[str, str]]:
    """Return [(check_name, message)] for every instruction that the image does not follow."""
    bad = []
    sh = e.get("shape")
    if sh == "blank":
        if m["n_ink"]: bad.append(("blank_has_ink", "Excel says BLANK but the image is not empty"))
        return bad
    if not m["n_ink"]:
        return [("empty", "image is EMPTY")]
    if e.get("background") is not None and m["bg"] != e["background"]:
        bad.append(("background", f"Excel background {e['background']}, image background {m['bg']}"))
    vals = m["ink_values"]
    if e.get("rgb") is not None and vals != [e["rgb"]]:
        bad.append(("rgb", f"Excel RGB {e['rgb']}, image {vals}"))
    if sh != "letter" and e.get("colour") == "white" and max(vals) <= m["bg"]:
        bad.append(("colour", "Excel says WHITE, image is darker than the background"))
    if sh != "letter" and e.get("colour") == "black" and min(vals) >= m["bg"]:
        bad.append(("colour", "Excel says BLACK, image is lighter than the background"))
    if sh == "letter" and min(vals) < m["bg"]:
        bad.append(("letter_colour", "Excel says WHITE letters, image letters are BLACK"))
    if m.get("centre_err_px", 0) > 1.0:
        bad.append(("position", f"centre {m['centre_err_px']:.1f} px from Excel position ({e['x']:g}, {e['y']:g})"))
    if "R" in e and "R_px" in m:
        tol = 1.0 if sh != "triangle_contour" else 1.5
        if abs(m["R_px"] - e["R"] * PPD) > tol:
            bad.append(("size", f"Excel radius {e['R']:g} deg ({e['size_wording']}) = {e['R'] * PPD:.1f} px, image {m['R_px']:.1f} px"))
    if e.get("size_wording") == "unclear":
        bad.append(("size_wording_unclear", f"Excel size wording is UNCLEAR ('{e['text']}'): drawn as diameter"))
    if sh == "triangle_contour":
        if not m["vertical_side_straight"]: bad.append(("triangle_side", "vertical side NOT straight"))
        if m["equilateral_err_pct"] > 8: bad.append(("triangle_shape", f"not equilateral ({m['equilateral_err_pct']:.0f}%)"))
    if sh and sh.startswith("bar"):
        if not m["straight"]: bad.append(("bar_straight", "bar NOT straight"))
        if m["length_px"] != round(e["length"] * PPD) or m["width_px"] != 1:
            bad.append(("bar_size", f"bar {m['length_px']}x{m['width_px']} px, Excel 1 deg = 35x1 px"))
    if sh == "letter" and m["box_px"] != (35, 35):
        bad.append(("letter_box", f"letter box {m['box_px']} px, Excel {e['box']:g} deg = 35 px"))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--monkey", default="gandalf"); args = ap.parse_args()
    root = WS / f"Data/VSD_Encoder_01/stimuli/{args.monkey}"
    man = pd.read_parquet(root / "manifest.parquet")
    exp = parse_shapes_csv(WS / "Data/EncoderData/Gandalf_VSDI_hs27Aug20_YN.csv") + \
          parse_contrast_csv(WS / "Data/EncoderData/ContrastCurve_Letters_GandalfLeft_ExpSummary.csv")
    rows, seen = [], set()
    for e in exp:
        cond = f"condAN{e['cond']}"
        mm = man[(man.h5_session == e["h5_session"]) & (man.condition == cond)]
        sh = e.get("shape")
        if not mm.empty and sh is not None:
            want = "blank" if sh == "blank" else sh
            mm = mm[(mm.shape_type == want) | (mm.is_blank & (want == "blank"))]
        if mm.empty:
            alt = man[(man.h5_session == e["h5_session"]) & (man.shape_type == sh)]
            if sh == "bar_horizontal" and not alt.empty and e["h5_session"].startswith("240718"):
                rows.append(dict(h5_session=e["h5_session"], condition=cond, check="excel_cond6_bar",
                                 message=f"Excel puts the horizontal bar in cond{e['cond']}; image is {alt.condition.iloc[0]}",
                                 excel=e["text"], uid=alt.stimulus_uid.iloc[0]))
                mm = alt
            else:
                rows.append(dict(h5_session=e["h5_session"], condition=cond, check="missing",
                                 message="Excel condition has NO image in the stimulus library", excel=e["text"], uid=""))
                continue
        r = mm.iloc[0]; seen.add((r.h5_session, r.condition))
        img = np.asarray(Image.open(WS / r.image_path).convert("RGB"))
        for chk, msg in compare(e, measure(img, e)):
            rows.append(dict(h5_session=r.h5_session, condition=r.condition, check=chk, message=msg, excel=e["text"], uid=r.stimulus_uid))
    extra = man[np.array([(s, c) not in seen for s, c in zip(man.h5_session, man.condition)]) & ~man.is_blank.to_numpy()]
    for _, r in extra.iterrows():
        rows.append(dict(h5_session=r.h5_session, condition=r.condition, check="not_in_excel",
                         message="image has no matching Excel instruction", excel="", uid=r.stimulus_uid))
    R = pd.DataFrame(rows, columns=["h5_session", "condition", "check", "message", "excel", "uid"])
    R["accepted"] = R.check.map(lambda c: ACCEPTED.get(c, ""))
    R.to_csv(root / f"catalog_check_{args.monkey}.csv", index=False)
    print(f"Checked {len(exp)} Excel instructions against {man.stimulus_uid.nunique()} library images "
          f"({len(man)} session-conditions).")
    new = R[R.accepted == ""]; old = R[R.accepted != ""]
    if len(new):
        lines = ["EXCEL INSTRUCTION  ≠  IMAGE", ""] + [f"{a} {b}: {m}"[:72] for a, b, m in zip(new.h5_session, new.condition, new.message)]
        print(big_red(lines))
    else:
        print(f"{BOLD}No unexpected differences between the Excel instructions and the images.{END}")
    if len(old):
        print(f"{RED}{BOLD}Known, accepted differences ({len(old)}):{END}")
        for (chk, why), g in old.groupby(["check", "accepted"]):
            print(f"{RED}  • {chk}: {len(g)} condition(s) — {why}{END}")
    print(f"Report: {root / f'catalog_check_{args.monkey}.csv'}")
    return 1 if len(new) else 0


if __name__ == "__main__":
    sys.exit(main())
