from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation.dual_metrics import dual_mask_metrics
from src.evaluation.mask import region_mask
from src.evaluation.roi_mask import box_to_mask, load_roi_mask
from src.loo.folds import (
    audit_protocol_a_leakage,
    audit_protocol_b_leakage,
    build_protocol_a_folds,
    build_protocol_b_folds,
    select_one_fold_per_stimulus,
)
from src.paths import project_root
from src.stimuli.identity import stimulus_id_from_row


def test_stimulus_id_shapes_and_letters():
    row = pd.Series(
        {
            "shape_type": "point",
            "color": "white",
            "size_deg": 0.1,
            "is_blank": False,
        }
    )
    assert stimulus_id_from_row(row) == "white_point_0.1"
    letter = pd.Series(
        {
            "shape_type": "letter",
            "color": "white",
            "size_deg": 1.0,
            "letter": "A",
            "is_blank": False,
        }
    )
    assert stimulus_id_from_row(letter) == "letter_A_white_1"


def test_box_to_mask_shape():
    mask = box_to_mask(10, 20, 5, 6, spatial_size=(100, 100))
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    assert mask.sum() == 5 * 6
    assert mask[20, 10]
    assert not mask[0, 0]


def test_load_frozen_roi_mask():
    repo = project_root()
    mask = load_roi_mask("white_point_0.1", repo=repo)
    assert mask.shape == (100, 100)
    assert mask.sum() > 0


def test_dual_mask_metrics_keys():
    rng = np.random.default_rng(0)
    orig = rng.normal(size=(8, 20, 20)).astype(np.float32)
    recon = orig + 0.1 * rng.normal(size=orig.shape).astype(np.float32)
    disk = region_mask((20, 20), mask_type="circle", radius=8)
    roi = box_to_mask(5, 5, 8, 8, spatial_size=(20, 20))
    m = dual_mask_metrics(orig, recon, disk_mask=disk, roi_mask=roi, disk_radius=8)
    assert "mean_r_disk" in m
    assert "mean_r_roi" in m
    assert "mean_trial_spatial_r_disk" in m
    assert "mean_trial_spatial_r_roi" in m
    assert m["n_pixels_roi"] == 64


def _toy_pairs() -> pd.DataFrame:
    rows = []
    # stimulus A in two sessions
    for date, split, n in [("d1", "train", 4), ("d2", "val", 3)]:
        for i in range(n):
            rows.append(
                {
                    "trial_global_id": len(rows),
                    "date": date,
                    "condition": "condAN1",
                    "split": split,
                    "shape_type": "point",
                    "color": "white",
                    "size_deg": 0.1,
                    "stimulus_text": "white point",
                    "is_blank": False,
                    "nc_exists": True,
                    "stimulus_exists": True,
                }
            )
    # filler stimulus
    for i in range(5):
        rows.append(
            {
                "trial_global_id": len(rows),
                "date": "d3",
                "condition": "condAN2",
                "split": "train",
                "shape_type": "point",
                "color": "black",
                "size_deg": 0.1,
                "stimulus_text": "black point",
                "is_blank": False,
                "nc_exists": True,
                "stimulus_exists": True,
            }
        )
    return pd.DataFrame(rows)


def test_protocol_b_no_stimulus_leakage():
    pairs = _toy_pairs()
    folds = build_protocol_b_folds(pairs, ["white_point_0.1"])
    assert len(folds) == 1
    spec, fold_df = folds[0]
    ok, _ = audit_protocol_b_leakage(fold_df, "white_point_0.1")
    assert ok
    assert spec.leakage_ok
    assert (fold_df.loc[fold_df["loo_split"] == "test", "stimulus_id"] == "white_point_0.1").all()
    rem = fold_df[fold_df["loo_split"].isin(["train", "val"])]
    assert (rem["stimulus_id"] != "white_point_0.1").all()


def test_protocol_a_condition_holdout_allows_other_sessions():
    pairs = _toy_pairs()
    folds = build_protocol_a_folds(pairs, ["white_point_0.1"])
    assert len(folds) == 2
    # Hold out d1/condAN1 → other session d2 of same stim may remain in remainder
    spec, fold_df = next(f for f in folds if f[0].heldout_date == "d1")
    ok, note = audit_protocol_a_leakage(fold_df, "white_point_0.1")
    assert ok
    assert "same stimulus_id" in note
    rem = fold_df[fold_df["loo_split"].isin(["train", "val"])]
    assert (rem["stimulus_id"] == "white_point_0.1").any()


def test_select_one_fold_per_stimulus_reproducible():
    pairs = _toy_pairs()
    folds = build_protocol_a_folds(
        pairs, ["white_point_0.1", "black_point_0.1"]
    )
    assert len(folds) == 3  # white: 2 sessions, black: 1
    a = select_one_fold_per_stimulus(folds, seed=17)
    b = select_one_fold_per_stimulus(folds, seed=17)
    assert len(a) == 2
    assert {s.heldout_stimulus_id for s, _ in a} == {
        "white_point_0.1",
        "black_point_0.1",
    }
    assert [s.fold_id for s, _ in a] == [s.fold_id for s, _ in b]
