"""The revised Schira mapping: stretch r -> Schira 2007 (thesis alpha) -> exact 2-anchor camera.

Forward (stimulus degrees -> camera px) is the same chain as ``registration_revised/fit_stretch.predict`` and the
dashboard's "Our joint fit" model; ``tests/test_schira_five_steps.py`` and ``tests/test_revised_geometry.py`` check
both directions. Config: ``configs/schira/revised.yaml``.

Camera coordinates are 0-based ``(row, col)`` on the VSD map, as complex ``c = col + i*row``.
Stimulus degrees follow ``src.retinotopy.visual_field`` (x right, y up, fixation at the canvas corner).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.retinotopy.schira import SchiraParams, cartesian_to_polar, forward_schira, inverse_schira, polar_to_cartesian

STIM_PPD = 35.0     # Joanna's anchor stimulus px -> degrees (step 1)


@dataclass(frozen=True)
class RevisedGeometry:
    a: float
    alpha_thesis: float
    r: float
    anchor_deg: tuple[tuple[float, float], tuple[float, float]]     # (x_deg, y_deg) of anchors A, B
    anchor_cam: tuple[complex, complex]                             # 0-based col + i*row of anchors A, B
    k: float = 1.0
    sech_ecc_k: float = 0.76
    sech_amp: float = 0.1821
    name: str = ""

    @property
    def params(self) -> SchiraParams:
        """Schira params in the repo convention (P = alpha_repo * theta, alpha_repo = 1/alpha_thesis)."""
        return SchiraParams(a=self.a, alpha=1.0 / self.alpha_thesis, k=self.k,
                            sech_ecc_k=self.sech_ecc_k, sech_amp=self.sech_amp)

    # ---- steps 2-4
    def deg_to_w(self, x_deg, y_deg) -> np.ndarray:
        ecc, theta = cartesian_to_polar(self.r * np.asarray(x_deg, float), np.asarray(y_deg, float))
        return forward_schira(ecc, theta, self.params)

    def w_to_deg(self, w) -> tuple[np.ndarray, np.ndarray]:
        ecc, theta = inverse_schira(w, self.params, min_real_z=0.0)
        xs, ys = polar_to_cartesian(ecc, theta)
        return xs / self.r, ys

    # ---- step 5
    def _similarity(self) -> tuple[complex, complex]:
        wa, wb = self.deg_to_w(*np.array(self.anchor_deg).T)
        ca, cb = self.anchor_cam
        c1 = (cb - ca) / (wb - wa)
        return c1, ca - c1 * wa

    def w_to_cam(self, w) -> np.ndarray:
        c1, c0 = self._similarity()
        return c1 * np.asarray(w) + c0

    def cam_to_w(self, row, col) -> np.ndarray:
        c1, c0 = self._similarity()
        return (np.asarray(col, float) + 1j * np.asarray(row, float) - c0) / c1

    # ---- full chain
    def deg_to_cam(self, x_deg, y_deg) -> tuple[np.ndarray, np.ndarray]:
        c = self.w_to_cam(self.deg_to_w(x_deg, y_deg))
        return np.imag(c), np.real(c)

    def cam_to_deg(self, row, col) -> tuple[np.ndarray, np.ndarray]:
        return self.w_to_deg(self.cam_to_w(row, col))


def load_revised_geometry(path: Path | str, set_name: str) -> RevisedGeometry:
    cfg = yaml.safe_load(Path(path).read_text())
    if set_name not in cfg.get("sets", {}):
        raise KeyError(f"set {set_name!r} not in {path} (have {sorted(cfg.get('sets', {}))})")
    day = str(cfg["sets"][set_name]["day"])
    m = cfg["model"]
    an = cfg["days"][day]["anchors"]
    if len(an) != 2:
        raise ValueError(f"day {day}: need exactly 2 anchors, got {len(an)}")
    deg = tuple((s["stim_rc"][1] / STIM_PPD, -s["stim_rc"][0] / STIM_PPD) for s in an)
    cam = tuple(complex(s["cam_rc"][1] - 1, s["cam_rc"][0] - 1) for s in an)
    return RevisedGeometry(a=float(m["a"]), alpha_thesis=float(m["alpha_thesis"]), r=float(m["r"]),
                           anchor_deg=deg, anchor_cam=cam, k=float(m.get("k", 1.0)),
                           sech_ecc_k=float(m.get("sech_ecc_k", 0.76)), sech_amp=float(m.get("sech_amp", 0.1821)),
                           name=set_name)
