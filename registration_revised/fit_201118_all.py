"""20/11/18 with ALL point pairs of Joanna's 'points Gandalf.xlsx' (lower table, rows 25-32), not only the 8 she kept
in opt_points_8_5_22.mat. Same model and fit as fit_stretch.py (stretch r -> Schira -> exact similarity through her
2 anchors; score = mean distance over the non-anchor points).

The lower table lists, per letter-corner position, the stimulus pixel (row, col) in session A and in session D (the
letters were shifted by -7 rows, +14 cols in D) and the camera point (row, col) in each session. 15 pairs:
7 from session A, 8 from session D; position (25, 66) of session D has two camera values ('F·' (54, 63), 'D·' (60, 60)).
Mixing A and D assumes the camera did not move between the two sessions (as Joanna did).

Run: scripts/py registration_revised/fit_201118_all.py   ->  figs/fit_201118_all.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_stretch as F  # noqa: E402

# (stim row, stim col) -> (cam row, cam col), 1-based, transcribed from 'points Gandalf.xlsx' rows 25-32
A = [((15, 18), (36, 25)), ((15, 52), (51, 42)), ((15, 35), (47, 37)), ((49, 52), (44, 81)),
     ((49, 18), (31, 78)), ((32, 18), (32, 53)), ((32, 52), (46, 62))]
D = [((8, 32), (47, 32)), ((8, 66), (67, 47)), ((8, 49), (57, 40)), ((42, 66), (51, 80)),
     ((42, 32), (38, 77)), ((25, 32), (41, 48)), ((25, 66), (54, 63)), ((25, 66), (60, 60))]   # last = 'D·'
ANCH_A = [(49, 18), (15, 18)]          # Joanna's winner anchors (session A)
ANCH_D = [(42, 32), (8, 32)]           # the same letter corners in session D


def build(pairs, anchors):
    return [p for p, _ in pairs], [c for _, c in pairs], anchors


def main():
    sets = {
        "Joanna's 8 (opt_points_8_5_22)": F.set_201118(),
        "all 15 (A 7 + D 8)": build(A + D, ANCH_A),
        "14: without D· (25,66)->(60,60)": build(A + D[:-1], ANCH_A),
        "14: without F· (25,66)->(54,63)": build(A + D[:-2] + D[-1:], ANCH_A),
        "session A only (7)": build(A, ANCH_A),
        "session D only (8), D anchors": build(D, ANCH_D),
    }
    rows = []
    for name, (s, c, an) in sets.items():
        f = F.fit(s, c, an)
        lo = F.loo(s, c, an)
        d1 = F.fit(s, c, an, fit_r=False, r_fixed=1.0)
        rows.append(dict(set=name, n_scored=int(f["keep"].sum()), r=f["r"], a=f["a"], alpha=f["alpha"], mean=f["mean"],
                         rmse=f["rmse"], loo_mean=lo.mean(), mean_r1=d1["mean"], worst=f["d"][f["keep"]].max()))
        print(f"{name:34s} n={rows[-1]['n_scored']:2d}  r {f['r']:.3f}  a {f['a']:.2f}  α {f['alpha']:.2f}  "
              f"mean {f['mean']:.2f}  LOO {lo.mean():.2f}  (r=1: {d1['mean']:.2f})  per point "
              + " ".join(f"{v:.1f}" for v in f["d"][f["keep"]]))
    for name, parts in (("A + D, own camera each (A and D anchors)", [sets["session A only (7)"], sets["session D only (8), D anchors"]]),
                        ("A + D without D·, own camera each", [sets["session A only (7)"], build(D[:-1], ANCH_D)])):
        j = F.fit_joint(parts)
        rows.append(dict(set=name, n_scored=11 if "D·" not in name else 10, r=j["r"], a=j["a"], alpha=j["alpha"],
                         mean=j["mean"], rmse=j["rmse"], per_day=" / ".join(f"{v:.2f}" for v in j["per_day"])))
        print(f"{name:34s} r {j['r']:.3f}  a {j['a']:.2f}  α {j['alpha']:.2f}  mean {j['mean']:.2f}  per session A / D "
              f"{rows[-1]['per_day']}")
    thesis = (*F.mat_points("opt_points_25_09_23_23_01.mat"), [(9, 21), (37, 21)])
    for name in ("all 15 (A 7 + D 8)", "14: without D· (25,66)->(60,60)"):
        j = F.fit_joint([thesis, sets[name]])
        rows.append(dict(set=f"joint: 10/07 thesis set + 20/11 {name}", n_scored=np.nan, r=j["r"], a=j["a"],
                         alpha=j["alpha"], mean=j["mean"], rmse=j["rmse"], per_day=" / ".join(f"{v:.2f}" for v in j["per_day"])))
        print(f"joint thesis + {name:24s} r {j['r']:.3f}  a {j['a']:.2f}  α {j['alpha']:.2f}  mean {j['mean']:.2f}  "
              f"per day {rows[-1]['per_day']}")
    pd.DataFrame(rows).to_csv(F.FIGS / "fit_201118_all.csv", index=False)


if __name__ == "__main__":
    main()
