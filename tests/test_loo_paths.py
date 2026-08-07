"""Unit tests for flat LOO path naming helpers."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from src.loo.paths import (
    cleanliness_leaf_tag,
    flat_leaf_name,
    flat_run_root_name,
    normalization_leaf_tag,
    resolve_flat_out_dir,
    roi_leaf_tag,
    short_layer_slug,
    short_model_slug,
    uniquify_leaf_dir,
)


def test_short_model_and_layer_slugs():
    assert short_model_slug("resnet18_imagenet") == "resnet18"
    assert short_model_slug("cornet_s_random") == "cornet_s"
    assert short_model_slug("gabor_serre_gwp") == "gabor_serre_gwp"
    assert short_layer_slug("layer3") == "l3"
    assert short_layer_slug("layer4") == "l4"
    assert short_layer_slug("V4") == "V4"


def test_normalization_and_roi_leaf_tags():
    assert normalization_leaf_tag("baseline_zscore") == "zscore"
    assert normalization_leaf_tag("zscore_baseline") == "zscore"
    assert normalization_leaf_tag("none") == "raw"
    assert normalization_leaf_tag("raw") == "raw"
    assert roi_leaf_tag("noise_ceiling_hull") == "NChull"
    assert roi_leaf_tag("disk") == "disk"
    assert roi_leaf_tag("none") == "full"
    assert roi_leaf_tag("box_union") == "boxunion"
    assert roi_leaf_tag("roi") == "boxroi"
    assert roi_leaf_tag("path", mask_path=Path("union_of_boxes__mask.npy")) == (
        "union_of_boxes"
    )


def test_cleanliness_leaf_tag():
    assert cleanliness_leaf_tag(trial_cleanliness_csv=None) == "all"
    assert (
        cleanliness_leaf_tag(
            trial_cleanliness_csv="qc.csv", keep=["good"]
        )
        == "clean"
    )
    assert (
        cleanliness_leaf_tag(
            trial_cleanliness_csv="qc.csv", keep=["good", "pattern_outlier"]
        )
        == "clean_good_pattern_outlier"
    )


def test_flat_run_root_and_leaf_names():
    root = flat_run_root_name(
        run_date=date(2026, 8, 6),
        start_frame=35,
        end_frame=46,
        model_slug="resnet18_imagenet",
        feature_layer="layer3",
    )
    assert root == "2026-08-06_35-46_resnet18_l3"

    leaf = flat_leaf_name(
        protocol="A",
        normalization="baseline_zscore",
        target_mask_mode="noise_ceiling_hull",
        cleanliness="clean",
    )
    assert leaf == "protocol_A_zscore_NChull_clean"

    leaf_raw = flat_leaf_name(
        protocol="B",
        normalization="none",
        target_mask_mode="noise_ceiling_hull",
        cleanliness="clean",
    )
    assert leaf_raw == "protocol_B_raw_NChull_clean"


def test_uniquify_leaf_dir(tmp_path: Path):
    base = uniquify_leaf_dir(tmp_path, "protocol_A_zscore_NChull_clean")
    assert base == tmp_path / "protocol_A_zscore_NChull_clean"
    base.mkdir()
    when = datetime(2026, 8, 6, 14, 30)
    timed = uniquify_leaf_dir(
        tmp_path, "protocol_A_zscore_NChull_clean", when=when
    )
    assert timed.name == "protocol_A_zscore_NChull_clean_1430"
    timed.mkdir()
    v2 = uniquify_leaf_dir(
        tmp_path, "protocol_A_zscore_NChull_clean", when=when
    )
    assert v2.name == "protocol_A_zscore_NChull_clean_v2"


def test_resolve_flat_out_dir(tmp_path: Path):
    root, leaf = resolve_flat_out_dir(
        tmp_path,
        run_date="2026-08-06",
        start_frame=35,
        end_frame=46,
        model_slug="resnet18_imagenet",
        feature_layer="layer3",
        protocol="A",
        normalization="baseline_zscore",
        target_mask_mode="noise_ceiling_hull",
        cleanliness="clean",
    )
    assert root.name == "2026-08-06_35-46_resnet18_l3"
    assert leaf.name == "protocol_A_zscore_NChull_clean"
    assert leaf.parent == root
