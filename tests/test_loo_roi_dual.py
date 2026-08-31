from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.dual_metrics import dual_mask_metrics
from src.evaluation.mask import region_mask
from src.evaluation.roi_mask import box_to_mask, load_roi_mask
from src.loo.folds import (
    audit_protocol_a_leakage,
    audit_protocol_b_leakage,
    audit_protocol_c_leakage,
    build_protocol_a_folds,
    build_protocol_b_folds,
    build_protocol_c_folds,
    expand_stimulus_id_patterns,
    filter_folds_excluding_train_only_dates,
    load_heldout_config,
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
    bar = pd.Series(
        {
            "shape_type": "bar_vertical",
            "color": "black",
            "size_deg": 0.3,
            "is_blank": False,
        }
    )
    assert stimulus_id_from_row(bar) == "black_bar_vertical_1"


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


def test_protocol_c_excludes_same_stimulus_from_train_val():
    pairs = _toy_pairs()
    folds_a = build_protocol_a_folds(pairs, ["white_point_0.1"])
    folds_c = build_protocol_c_folds(pairs, ["white_point_0.1"])
    assert len(folds_a) == 2
    assert len(folds_c) == 1
    spec, fold_df = folds_c[0]
    assert spec.heldout_date == "d1"
    assert spec.heldout_condition == "condAN1"
    assert spec.fold_id.startswith("C__")
    assert spec.protocol == "C"
    ok, note = audit_protocol_c_leakage(
        fold_df,
        "white_point_0.1",
        heldout_date=str(spec.heldout_date),
        heldout_condition=str(spec.heldout_condition),
    )
    assert ok
    assert spec.leakage_ok
    rem = fold_df[fold_df["loo_split"].isin(["train", "val"])]
    assert (rem["stimulus_id"] != "white_point_0.1").all()
    test = fold_df[fold_df["loo_split"] == "test"]
    assert (test["date"].astype(str) == spec.heldout_date).all()
    assert (test["condition"].astype(str) == spec.heldout_condition).all()
    # Other sessions of the held-out stimulus are dropped (not in fold)
    assert not (
        (fold_df["stimulus_id"] == "white_point_0.1")
        & ~((fold_df["date"].astype(str) == spec.heldout_date)
            & (fold_df["condition"].astype(str) == spec.heldout_condition))
    ).any()
    # Train is smaller than protocol A (same-stim sessions removed)
    a_spec = next(
        s
        for s, _ in folds_a
        if s.heldout_date == spec.heldout_date
        and s.heldout_condition == spec.heldout_condition
    )
    assert spec.n_train + spec.n_val < a_spec.n_train + a_spec.n_val
    assert spec.n_test == a_spec.n_test


def test_protocol_c_all_sessions_matches_protocol_a_count():
    pairs = _toy_pairs()
    folds_a = build_protocol_a_folds(pairs, ["white_point_0.1"])
    folds_c = build_protocol_c_folds(
        pairs, ["white_point_0.1"], all_sessions=True
    )
    assert len(folds_c) == len(folds_a) == 2


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


def test_expand_stimulus_id_patterns_glob_and_exact():
    available = [
        "letter_A_white_1",
        "letter_G_white_1",
        "white_point_0.1",
    ]
    assert expand_stimulus_id_patterns(["letter_*"], available) == [
        "letter_A_white_1",
        "letter_G_white_1",
    ]
    assert expand_stimulus_id_patterns(["white_point_0.1"], available) == [
        "white_point_0.1"
    ]
    # Exact ids that are absent still pass through (fold builder skips empty).
    assert expand_stimulus_id_patterns(["missing_id"], available) == ["missing_id"]
    with pytest.raises(ValueError, match="circle_\\*"):
        expand_stimulus_id_patterns(["circle_*"], available)


def test_filter_folds_excluding_train_only_dates():
    pairs = _toy_pairs()
    folds = build_protocol_a_folds(pairs, ["white_point_0.1"])
    assert {s.heldout_date for s, _ in folds} == {"d1", "d2"}
    kept = filter_folds_excluding_train_only_dates(folds, ["d1"])
    assert len(kept) == 1
    assert kept[0][0].heldout_date == "d2"
    assert filter_folds_excluding_train_only_dates(folds, []) == folds


def test_load_heldout_config_train_only_sessions(tmp_path: Path):
    path = tmp_path / "heldout.yaml"
    path.write_text(
        "heldout_stimulus_ids:\n  - letter_*\ntrain_only_sessions:\n  - 201118a\n"
    )
    cfg = load_heldout_config(path)
    assert cfg.stimulus_ids == ["letter_*"]
    assert cfg.train_only_sessions == ["201118a"]


def test_heldout_non_letters_yaml_has_no_letter_ids():
    repo = project_root()
    cfg = load_heldout_config(repo / "experiments/loo_encoding/heldout_non_letters.yaml")
    assert cfg.train_only_sessions == ["201118a"]
    assert cfg.stimulus_ids
    assert all(not sid.startswith("letter_") for sid in cfg.stimulus_ids)
    letters = load_heldout_config(
        repo / "experiments/loo_encoding/heldout_letters.yaml"
    )
    default = load_heldout_config(repo / "experiments/loo_encoding/heldout_list.yaml")
    # Do not mutate the shared/default lists; non-letters is an extra file.
    assert "letter_*" in letters.stimulus_ids
    assert any(sid.startswith("letter_") or sid == "letter_*" for sid in default.stimulus_ids)
    assert set(cfg.stimulus_ids).isdisjoint(set(letters.stimulus_ids))
