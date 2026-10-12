"""Legacy visual-field helpers (301×512 @ 35 ppd, top-left origin)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

_PKG = Path(__file__).resolve().parents[1]


def load_paths() -> dict:
    with open(_PKG / "configs" / "paths.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["legacy_root"])
    env = __import__("os").environ.get("SHIRA_REVISED_LEGACY_ROOT")
    if env:
        root = Path(env)
    out = {"legacy_root": root}
    for key, rel in cfg.get("artifacts", {}).items():
        out[key] = root / rel
    return out


def load_gandalf_legacy_yaml() -> dict:
    with open(_PKG / "configs" / "gandalf_legacy.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deg_to_stim_px(
    x_deg: np.ndarray | float,
    y_deg: np.ndarray | float,
    *,
    ppd: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy ``conv_pix_deg(..., pix2deg=0)``: deg → px with top-left origin.

    Uses ``round(deg * ppd)``. Signs are kept (negative y → negative row index
    before clipping); letter placement in MATLAB uses ``abs`` of centers when
    indexing — callers should pass the same convention as ``create_letter_stim``.
    """
    x = np.round(np.asarray(x_deg, dtype=np.float64) * ppd)
    y = np.round(np.asarray(y_deg, dtype=np.float64) * ppd)
    return x, y


def letter_center_stim_px(letter: str, cfg: dict | None = None) -> tuple[float, float]:
    """Return (col, row) pixel center for a letter on the legacy canvas.

    MATLAB pastes with ``round(35*|x|)`` etc. (abs of deg center).
    """
    cfg = cfg or load_gandalf_legacy_yaml()
    centers = cfg["letter_centers_deg"]
    key = letter.upper()
    if key not in centers:
        raise KeyError(f"no center for letter {letter!r}; known={sorted(centers)}")
    xd, yd = centers[key]
    col = float(round(35.0 * abs(xd)))
    row = float(round(35.0 * abs(yd)))
    return col, row


def make_abs_vf_grid(
    n_rows: int,
    n_cols: int,
    *,
    x_max_deg: float,
    y_max_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linspace abs VF map matching ``trans_cortex`` m1/m2 construction.

    ``m1`` spans cols (x), ``m2`` spans rows (y); both positive.
    """
    m1 = np.linspace(0.0, float(x_max_deg), int(n_cols))
    m2 = np.linspace(0.0, float(y_max_deg), int(n_rows))
    xx, yy = np.meshgrid(m1, m2)
    return xx, yy
