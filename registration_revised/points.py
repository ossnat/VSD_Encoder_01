"""Joanna's 10/07/18 point tables (legacy_code/.../Gandalf/VSDI points/100718/opt-points-10-07-18.xlsx).

The sheet holds two tables side by side (header row 7):
  left  (B:F)  'stimulus points'   : condition number, stimulus, VF degrees, VF cartesian, VSD point
  right (K:O)  'registration points': point number,     stimulus, VF degrees, VF cartesian, VSD point
VF cartesian = (x, y) stimulus-canvas pixels at 35 px/deg from fixation (x right, y down), e.g. (0.6, -0.75) deg -> (21, 26).
VSD point    = (x, y) camera pixels as written by Joanna, 1-based; which corner is (1, 1) is what we test.

Parsed straight from the xlsx XML (no openpyxl in the venv).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
XLSX = REPO.parent / "legacy_code/Retinotopic model Joanna/Gandalf/VSDI points/100718/opt-points-10-07-18.xlsx"
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _cells(path: Path) -> dict[str, str]:
    z = zipfile.ZipFile(path)
    ss = ["".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t"))
          for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS)]
    out = {}
    for c in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{{{_NS['m']}}}c"):
        v = c.find("m:v", _NS)
        if v is not None:
            out[c.get("r")] = ss[int(v.text)] if c.get("t") == "s" else v.text
    return out


def _pair(s: str) -> tuple[float, float]:
    a, b = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(a), float(b)


def _table(cells: dict, cols: str, id_name: str) -> pd.DataFrame:
    rows, r = [], 8
    while f"{cols[0]}{r}" in cells:
        c = [cells.get(f"{k}{r}", "") for k in cols]
        (xd, yd), (xc, yc), (xv, yv) = _pair(c[2]), _pair(c[3]), _pair(c[4])
        rows.append({id_name: int(float(c[0])), "stimulus": " ".join(c[1].split()),
                     "vf_x_deg": xd, "vf_y_deg": yd, "vf_x_px": xc, "vf_y_px": yc, "vsd_x": xv, "vsd_y": yv})
        r += 1
    return pd.DataFrame(rows)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(stimulus points, registration points)."""
    cells = _cells(XLSX)
    return _table(cells, "BCDEF", "cond"), _table(cells, "KLMNO", "point")


if __name__ == "__main__":
    s, r = load()
    print(s.to_string(index=False)); print(); print(r.to_string(index=False))
