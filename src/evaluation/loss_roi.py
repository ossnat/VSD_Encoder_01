"""Named loss / training-target ROIs for Ridge MSE.

Resolve CLI aliases such as ``none``, ``disk``, ``box_union``,
``noise_ceiling_hull``, ``roi``, or an arbitrary mask path into a boolean
``(H, W)`` mask (or ``None`` for full-frame MSE).

The global noise-ceiling hull mask is the across-condition **naive** (magenta)
convex hull at thr=0.85 (``r >= 0.85``), installed at
``NOISE_CEILING_HULL_MASK_RELPATH``. That artifact is built with the NC ROI
``--window`` (default ``win_0035_0046`` raw) and is **independent** of the
LOO / ridge analysis window — analysis uses its own config and simply loads
the installed mask. Selecting ``noise_ceiling_hull`` raises a clear error if
the file is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.mask import region_mask
from src.evaluation.roi_mask import (
    DEFAULT_ROIS_DIR,
    load_mask_from_path,
    load_roi_mask,
)
from src.paths import project_root

# Fixed named-mask locations (repo-relative).
BOX_UNION_MASK_RELPATH = Path(
    "experiments/loo_encoding/roi_compare/union_of_boxes__mask.npy"
)
# Official global noise-ceiling hull (naive / magenta across-condition thr0.85).
# Built via NC ROI --window (default win_0035_0046); independent of LOO window.
NOISE_CEILING_HULL_MASK_RELPATH = Path(
    "experiments/noise_ceiling_roi/rois/global_noise_ceiling_hull__mask.npy"
)

DEFAULT_DISK_RADIUS = 50
DEFAULT_SPATIAL_SIZE = (100, 100)

# Modes that mean "no spatial mask" (full-frame Y / MSE).
_NONE_ALIASES = frozenset({"none", "full", "off", "false"})
# Circular eval-disk aliases (center FOV, radius from ridge config).
_DISK_ALIASES = frozenset({"disk", "circular", "circle"})
# Per-fold held-out stimulus box from --roi-dir.
_ROI_ALIASES = frozenset({"roi", "box", "box_roi"})
# Fixed named file aliases.
_NAMED_FILE_MODES = frozenset({"box_union", "noise_ceiling_hull"})


def parse_loss_roi_arg(raw: str) -> tuple[str, Path | None]:
    """
    Parse ``--target-mask`` / ``--loss-roi`` into ``(mode, path)``.

    Modes:
      - ``none`` / ``full`` — full-frame Y (default)
      - ``disk`` / ``circular`` — centered circle (radius from ridge eval cfg)
      - ``box_union`` — union-of-boxes mask at ``BOX_UNION_MASK_RELPATH``
      - ``noise_ceiling_hull`` — official global naive hull mask (thr0.85)
      - ``roi`` — held-out stimulus box from ``--roi-dir``
      - any other string — filesystem path to ``.npy`` / ``.yaml``
    """
    key = str(raw).strip()
    lowered = key.lower()
    if lowered in _NONE_ALIASES:
        return "none", None
    if lowered in _DISK_ALIASES:
        return "disk", None
    if lowered in _ROI_ALIASES:
        return "roi", None
    if lowered == "box_union":
        return "box_union", BOX_UNION_MASK_RELPATH
    if lowered == "noise_ceiling_hull":
        return "noise_ceiling_hull", NOISE_CEILING_HULL_MASK_RELPATH
    return "path", Path(key)


def named_mask_relpath(mode: str) -> Path | None:
    """Return the repo-relative path for a fixed named mode, else ``None``."""
    if mode == "box_union":
        return BOX_UNION_MASK_RELPATH
    if mode == "noise_ceiling_hull":
        return NOISE_CEILING_HULL_MASK_RELPATH
    return None


def protocol_dir_suffix(
    mode: str,
    *,
    run_tag: str | None = None,
    mask_path: Path | None = None,
) -> str | None:
    """
    Protocol-dir suffix after ``protocol_{A,B}``.

    Returns ``None`` for full-frame (``none``) so the dir stays ``protocol_B``.
    Named modes use a stable token (``disk``, ``box_union``, …).
    """
    if mode == "none":
        return None
    if mode == "roi":
        return "box_roi"
    if mode in {"disk", "box_union", "noise_ceiling_hull"}:
        return mode
    if run_tag:
        return _safe_dir_token(run_tag)
    return _mask_path_dir_suffix(mask_path)


def _safe_dir_token(raw: str) -> str:
    token = "".join(c if (c.isalnum() or c in "-_") else "_" for c in raw)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "mask"


def _mask_path_dir_suffix(mask_path: Path | None) -> str:
    if mask_path is None:
        return "mask"
    stem = mask_path.stem
    if stem.endswith("__mask"):
        stem = stem[: -len("__mask")]
    elif stem.endswith("_mask"):
        stem = stem[: -len("_mask")]
    return _safe_dir_token(stem)


def resolve_loss_roi(
    mode: str,
    *,
    mask_path: Path | None = None,
    spatial_size: tuple[int, int] = DEFAULT_SPATIAL_SIZE,
    disk_radius: int = DEFAULT_DISK_RADIUS,
    heldout_stimulus_id: str | None = None,
    repo: Path | None = None,
    roi_dir: Path | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Resolve a loss/target ROI mode to ``(mask | None, meta)``.

    ``None`` mask means full-frame MSE (mode ``none``).
    """
    root = repo or project_root()
    h, w = int(spatial_size[0]), int(spatial_size[1])
    spatial = (h, w)
    meta: dict[str, Any] = {
        "target_mask_mode": mode,
        "n_pixels_full": int(h * w),
    }

    if mode == "none":
        meta["n_pixels_train"] = meta["n_pixels_full"]
        meta["train_targets"] = "full_frame"
        return None, meta

    if mode == "disk":
        mask = region_mask(spatial, mask_type="circle", radius=int(disk_radius))
        meta.update(
            {
                "train_targets": "disk",
                "disk_radius": int(disk_radius),
                "n_pixels_train": int(mask.sum()),
            }
        )
        return mask, meta

    if mode == "roi":
        if not heldout_stimulus_id:
            raise ValueError(
                "loss ROI mode 'roi' requires heldout_stimulus_id "
                "(per-fold stimulus box)"
            )
        mask = load_roi_mask(
            heldout_stimulus_id,
            repo=root,
            spatial_size=spatial,
            roi_dir=roi_dir,
        )
        meta.update(
            {
                "train_targets": "roi_box",
                "roi_stimulus_id": heldout_stimulus_id,
                "roi_dir": str((roi_dir or (root / DEFAULT_ROIS_DIR)).resolve()),
                "n_pixels_train": int(mask.sum()),
            }
        )
        return mask, meta

    # Named file modes + arbitrary path.
    if mode in _NAMED_FILE_MODES:
        rel = named_mask_relpath(mode)
        assert rel is not None
        resolved = root / rel
        if mode == "noise_ceiling_hull" and not resolved.is_file():
            raise FileNotFoundError(
                f"loss ROI mode 'noise_ceiling_hull' selected, but mask file "
                f"is missing: {resolved}. "
                f"Install the across-condition naive hull (thr=0.85) at this "
                f"path, or pass an explicit --target-mask / --loss-roi path."
            )
        if not resolved.is_file():
            raise FileNotFoundError(
                f"loss ROI mode {mode!r}: mask not found: {resolved}"
            )
        mask = load_mask_from_path(resolved, spatial_size=spatial, repo=root)
        meta.update(
            {
                "train_targets": mode,
                "mask_path": str(resolved.resolve()),
                "n_pixels_train": int(mask.sum()),
            }
        )
        return mask, meta

    if mode != "path":
        raise ValueError(f"Unknown loss ROI mode: {mode!r}")
    if mask_path is None:
        raise ValueError("loss ROI mode 'path' requires mask_path")
    resolved = mask_path if mask_path.is_absolute() else root / mask_path
    if not resolved.is_file():
        raise FileNotFoundError(f"Mask path not found: {resolved}")
    mask = load_mask_from_path(resolved, spatial_size=spatial, repo=root)
    meta.update(
        {
            "train_targets": "mask_path",
            "mask_path": str(resolved.resolve()),
            "n_pixels_train": int(mask.sum()),
        }
    )
    return mask, meta
