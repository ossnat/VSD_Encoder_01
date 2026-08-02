"""Load frozen stimulus ROI boxes / masks for LOO evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.paths import project_root


DEFAULT_ROIS_DIR = Path("experiments/loo_encoding/rois")


def rois_dir(repo: Path | None = None, *, roi_dir: Path | None = None) -> Path:
    """Return ROI directory; ``roi_dir`` overrides the frozen default."""
    if roi_dir is not None:
        return Path(roi_dir)
    root = repo or project_root()
    return root / DEFAULT_ROIS_DIR


def load_roi_yaml(
    stimulus_id: str,
    *,
    repo: Path | None = None,
    roi_dir: Path | None = None,
) -> dict[str, Any]:
    path = rois_dir(repo, roi_dir=roi_dir) / f"{stimulus_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ROI YAML for {stimulus_id!r}: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"ROI YAML must be a mapping: {path}")
    return data


def roi_box_from_yaml(data: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return (x0, y0, width, height)."""
    return (
        int(data["x0"]),
        int(data["y0"]),
        int(data["width"]),
        int(data["height"]),
    )


def box_to_mask(
    x0: int,
    y0: int,
    width: int,
    height: int,
    *,
    spatial_size: tuple[int, int] = (100, 100),
) -> np.ndarray:
    """Boolean mask True inside the inclusive-start exclusive-end box."""
    h, w = spatial_size
    mask = np.zeros((h, w), dtype=bool)
    x1 = min(w, x0 + width)
    y1 = min(h, y0 + height)
    x0c = max(0, x0)
    y0c = max(0, y0)
    if x0c < x1 and y0c < y1:
        mask[y0c:y1, x0c:x1] = True
    return mask


def load_roi_mask(
    stimulus_id: str,
    *,
    repo: Path | None = None,
    spatial_size: tuple[int, int] = (100, 100),
    prefer_npy: bool = True,
    roi_dir: Path | None = None,
) -> np.ndarray:
    """
    Load the frozen ROI mask for ``stimulus_id``.

    Prefers ``{stimulus_id}__mask.npy`` when present; otherwise builds from YAML.
    """
    root = repo or project_root()
    rdir = rois_dir(root, roi_dir=roi_dir)
    npy_path = rdir / f"{stimulus_id}__mask.npy"
    if prefer_npy and npy_path.is_file():
        mask = np.load(npy_path)
        mask = np.asarray(mask).astype(bool)
        if mask.shape != spatial_size:
            raise ValueError(
                f"ROI mask shape {mask.shape} != spatial_size {spatial_size} "
                f"for {stimulus_id!r}"
            )
        return mask

    data = load_roi_yaml(stimulus_id, repo=root, roi_dir=roi_dir)
    x0, y0, width, height = roi_box_from_yaml(data)
    map_shape = data.get("map_shape")
    if map_shape is not None:
        expected = (int(map_shape[0]), int(map_shape[1]))
        if expected != spatial_size:
            raise ValueError(
                f"ROI map_shape {expected} != spatial_size {spatial_size} "
                f"for {stimulus_id!r}"
            )
    return box_to_mask(x0, y0, width, height, spatial_size=spatial_size)


def load_mask_from_path(
    path: Path,
    *,
    spatial_size: tuple[int, int] = (100, 100),
    repo: Path | None = None,
) -> np.ndarray:
    """
    Load a training/eval mask from ``.npy`` or box ``.yaml``.

    YAML may be a box (``x0/y0/width/height``) or reference ``stimulus_id`` /
    ``mask_npy`` relative to the YAML's directory.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask path not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        mask = np.asarray(np.load(path)).astype(bool)
        if mask.shape != spatial_size:
            raise ValueError(
                f"Mask shape {mask.shape} != spatial_size {spatial_size} ({path})"
            )
        return mask
    if suffix in {".yaml", ".yml"}:
        with path.open() as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Mask YAML must be a mapping: {path}")
        mask_npy = data.get("mask_npy")
        if mask_npy:
            npy = Path(str(mask_npy))
            if not npy.is_absolute():
                npy = path.parent / npy
            return load_mask_from_path(npy, spatial_size=spatial_size, repo=repo)
        if all(k in data for k in ("x0", "y0", "width", "height")):
            x0, y0, width, height = roi_box_from_yaml(data)
            return box_to_mask(x0, y0, width, height, spatial_size=spatial_size)
        sid = data.get("stimulus_id")
        if sid:
            return load_roi_mask(
                str(sid),
                repo=repo,
                spatial_size=spatial_size,
                roi_dir=path.parent,
            )
        raise ValueError(
            f"Mask YAML needs box coords, mask_npy, or stimulus_id: {path}"
        )
    raise ValueError(f"Unsupported mask file type {suffix!r}: {path}")


def list_roi_stimulus_ids(
    *,
    repo: Path | None = None,
    roi_dir: Path | None = None,
) -> list[str]:
    """Stimulus IDs with accepted ROIs from ``all_rois.yaml``."""
    path = rois_dir(repo, roi_dir=roi_dir) / "all_rois.yaml"
    with path.open() as f:
        data = yaml.safe_load(f)
    rois = data.get("rois", [])
    return [str(r["stimulus_id"]) for r in rois if r.get("status") == "accepted"]
