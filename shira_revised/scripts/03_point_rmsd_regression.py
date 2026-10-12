#!/usr/bin/env python3
"""Landmark RMSD under legacy α convention (skeleton toward Joanna ~1.5 px).

Loads ``opt_points_8_5_22.mat`` + Joanna winner anchors, maps VF landmarks
through ``forward_legacy_schira`` + 2-pt similarity, reports chamber RMSD.

Full parity with MATLAB also needs integer ``xcoord/ycoord`` + low-hem flips
from ``trans_and_registration.m`` — this script documents the gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shira_revised.src.legacy_schira import LegacySchiraParams, forward_legacy_schira
from shira_revised.src.legacy_vf import load_gandalf_legacy_yaml, load_paths


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def fit_similarity_w_to_brain(
    w_anch: np.ndarray, brain_rc: np.ndarray
) -> tuple[complex, complex]:
    """Return (s, t) with chamber = s * w + t, chamber as col+i*row."""
    z = w_anch
    p = brain_rc[:, 1] + 1j * brain_rc[:, 0]
    s = (p[1] - p[0]) / (z[1] - z[0])
    t = p[0] - s * z[0]
    return s, t


def main() -> None:
    paths = load_paths()
    cfg = load_gandalf_legacy_yaml()
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise SystemExit("scipy required") from e

    op = paths["opt_points_201118"]
    if not op.is_file():
        raise SystemExit(f"missing {op}")
    pts = np.asarray(loadmat(op, squeeze_me=True)["opt_points"], dtype=np.float64)
    # cols: VF_row, VF_col, deg_y, deg_x, brain_row, brain_col
    vf_row, vf_col = pts[:, 0], pts[:, 1]
    brain = pts[:, 4:6]

    # Deg from pixels (top-left abs convention): x=col/ppd, y=-row/ppd
    ppd = float(cfg["ppd"])
    x_deg = vf_col / ppd
    y_deg = -vf_row / ppd

    block = cfg["joanna_201118"]
    params = LegacySchiraParams(
        a=float(block["a"]),
        alpha=float(block["alpha"]),
        k=float(block["k"]),
        alpha_mode="divide",
        vf_abs=True,
    )
    # Anchor indices: rows whose VF pix match winner anchors
    want = [tuple(map(int, a)) for a in block["vf_pix_anchors"]]
    idx = []
    for wr, wc in want:
        hits = np.where((pts[:, 0] == wr) & (pts[:, 1] == wc))[0]
        if hits.size == 0:
            raise SystemExit(f"anchor VF ({wr},{wc}) not in opt_points")
        idx.append(int(hits[0]))
    idx = np.asarray(idx)

    w = forward_legacy_schira(x_deg, y_deg, params)
    s, t = fit_similarity_w_to_brain(w[idx], brain[idx])
    pred = s * w + t
    pred_rc = np.column_stack([pred.imag, pred.real])  # row, col
    ok = np.isfinite(pred_rc).all(axis=1)
    err = rmsd(pred_rc[ok], brain[ok])
    print(f"legacy divide+abs 2-pt similarity RMSD over {ok.sum()} landmarks: {err:.4f} px")
    print(f"Joanna reported mean_rmsd_px: {block['mean_rmsd_px']}")
    print(
        "If far from ~1.5: next port integer xcoord/ycoord + lowH flips "
        "(trans_and_registration), not only continuous w."
    )

    # Contrasting multiply+signed
    params_m = LegacySchiraParams(
        a=float(block["a"]),
        alpha=float(block["alpha"]),
        k=float(block["k"]),
        alpha_mode="multiply",
        vf_abs=False,
        ecc_width=0.76,
    )
    w_m = forward_legacy_schira(x_deg, y_deg, params_m)
    s_m, t_m = fit_similarity_w_to_brain(w_m[idx], brain[idx])
    pred_m = s_m * w_m + t_m
    pred_m_rc = np.column_stack([pred_m.imag, pred_m.real])
    ok_m = np.isfinite(pred_m_rc).all(axis=1)
    err_m = rmsd(pred_m_rc[ok_m], brain[ok_m])
    print(f"thesis multiply+signed 2-pt RMSD: {err_m:.4f} px")


if __name__ == "__main__":
    main()
