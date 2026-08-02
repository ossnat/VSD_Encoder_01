"""Unit tests for named loss / training-target ROI resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.loss_roi import (
    BOX_UNION_MASK_RELPATH,
    DEFAULT_DISK_RADIUS,
    NOISE_CEILING_HULL_MASK_RELPATH,
    parse_loss_roi_arg,
    resolve_loss_roi,
)
from src.evaluation.mask import region_mask
from src.paths import project_root


def test_parse_loss_roi_named_modes():
    assert parse_loss_roi_arg("none") == ("none", None)
    assert parse_loss_roi_arg("full") == ("none", None)
    assert parse_loss_roi_arg("disk") == ("disk", None)
    assert parse_loss_roi_arg("circular") == ("disk", None)
    assert parse_loss_roi_arg("roi") == ("roi", None)
    mode, path = parse_loss_roi_arg("box_union")
    assert mode == "box_union"
    assert path == BOX_UNION_MASK_RELPATH
    mode, path = parse_loss_roi_arg("noise_ceiling_hull")
    assert mode == "noise_ceiling_hull"
    assert path == NOISE_CEILING_HULL_MASK_RELPATH
    mode, path = parse_loss_roi_arg("foo/bar__mask.npy")
    assert mode == "path"
    assert path == Path("foo/bar__mask.npy")


def test_resolve_none_is_full_frame():
    mask, meta = resolve_loss_roi("none", spatial_size=(100, 100))
    assert mask is None
    assert meta["train_targets"] == "full_frame"
    assert meta["n_pixels_train"] == 10_000
    assert meta["target_mask_mode"] == "none"


def test_resolve_disk_shape_and_radius():
    spatial = (100, 100)
    mask, meta = resolve_loss_roi(
        "disk", spatial_size=spatial, disk_radius=DEFAULT_DISK_RADIUS
    )
    assert mask is not None
    assert mask.shape == spatial
    assert mask.dtype == bool
    expected = region_mask(spatial, mask_type="circle", radius=DEFAULT_DISK_RADIUS)
    assert np.array_equal(mask, expected)
    assert meta["train_targets"] == "disk"
    assert meta["disk_radius"] == DEFAULT_DISK_RADIUS
    assert meta["n_pixels_train"] == int(expected.sum())
    assert meta["n_pixels_train"] < meta["n_pixels_full"]


def test_resolve_noise_ceiling_hull_missing_errors_clearly():
    repo = project_root()
    placeholder = repo / NOISE_CEILING_HULL_MASK_RELPATH
    if placeholder.is_file():
        pytest.skip(f"placeholder already exists: {placeholder}")
    with pytest.raises(FileNotFoundError, match="noise_ceiling_hull") as exc:
        resolve_loss_roi("noise_ceiling_hull", repo=repo, spatial_size=(100, 100))
    msg = str(exc.value)
    assert "global_noise_ceiling_hull__mask.npy" in msg
    assert "missing" in msg.lower() or "Create" in msg


def test_resolve_box_union_when_present():
    repo = project_root()
    path = repo / BOX_UNION_MASK_RELPATH
    if not path.is_file():
        pytest.skip(f"box_union mask not in repo: {path}")
    mask, meta = resolve_loss_roi("box_union", repo=repo, spatial_size=(100, 100))
    assert mask is not None
    assert mask.shape == (100, 100)
    assert mask.sum() > 0
    assert meta["train_targets"] == "box_union"
    assert "union_of_boxes" in meta["mask_path"]
