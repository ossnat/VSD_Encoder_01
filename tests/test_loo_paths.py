"""Unit tests for flat LOO path naming helpers."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np

from src.loo.paths import (
    cleanliness_leaf_tag,
    flat_leaf_name,
    flat_run_root_name,
    list_loo_fold_dirs,
    normalization_leaf_tag,
    resolve_flat_out_dir,
    roi_leaf_tag,
    short_layer_slug,
    short_model_slug,
    uniquify_leaf_dir,
)


def test_list_loo_fold_dirs_includes_protocol_c(tmp_path: Path):
    (tmp_path / "A__stim__date_cond").mkdir()
    (tmp_path / "B__stim").mkdir()
    (tmp_path / "C__stim__date_cond").mkdir()
    (tmp_path / "overview").mkdir()
    names = [p.name for p in list_loo_fold_dirs(tmp_path)]
    assert names == ["A__stim__date_cond", "B__stim", "C__stim__date_cond"]


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
        normalization="baseline_zscore",
    )
    assert root == "2026-08-06_35-46_resnet18_l3_zscore"

    root_raw = flat_run_root_name(
        run_date=date(2026, 8, 6),
        start_frame=35,
        end_frame=46,
        model_slug="resnet18_imagenet",
        feature_layer="layer3",
        normalization="none",
    )
    assert root_raw == "2026-08-06_35-46_resnet18_l3_raw"

    shared = flat_run_root_name(
        run_date=date(2026, 8, 6),
        start_frame=35,
        end_frame=46,
        model_slug="resnet18_imagenet",
        feature_layer="layer3",
    )
    assert shared == "2026-08-06_35-46_resnet18_l3"

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
    assert root.name == "2026-08-06_35-46_resnet18_l3_zscore"
    assert leaf.name == "protocol_A_zscore_NChull_clean"
    assert leaf.parent == root

    root_raw, leaf_raw_dir = resolve_flat_out_dir(
        tmp_path,
        run_date="2026-08-06",
        start_frame=35,
        end_frame=46,
        model_slug="resnet18_imagenet",
        feature_layer="layer3",
        protocol="B",
        normalization="none",
        target_mask_mode="noise_ceiling_hull",
        cleanliness="all",
    )
    assert root_raw.name == "2026-08-06_35-46_resnet18_l3_raw"
    assert leaf_raw_dir.name == "protocol_B_raw_NChull_all"

    shared, shared_leaf = resolve_flat_out_dir(
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
        run_root="2026-08-06_35-46_resnet18_l3",
    )
    assert shared.name == "2026-08-06_35-46_resnet18_l3"
    assert shared_leaf.name == "protocol_A_zscore_NChull_clean"


def test_overview_collage_from_fold_mean_arrays(tmp_path: Path):
    from experiments.loo_encoding.make_loo_triplet_overview import (
        write_overview_batches,
    )

    protocol = tmp_path / "protocol_A_raw_NChull_CM"
    fold = protocol / "A__letter_A_white_1__201118c_condAN2"
    fold.mkdir(parents=True)
    orig = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    recon = orig * 0.4
    np.save(fold / "fold_mean_orig.npy", orig)
    np.save(fold / "fold_mean_recon.npy", recon)

    written = write_overview_batches(
        protocol, per_page=12, alias="all_folds_triplets.png"
    )
    alias = protocol / "overview" / "all_folds_triplets.png"
    assert alias.is_file() and alias.stat().st_size > 0
    assert alias in written
    index = (protocol / "overview" / "overview_index.txt").read_text()
    assert "fold_mean_orig.npy" in index
    assert not (fold / "sanity_orig_recon_residual.png").exists()


def test_overview_inclusive_puts_all_folds_on_one_page(tmp_path: Path):
    from experiments.loo_encoding.make_loo_triplet_overview import (
        write_overview_batches,
    )

    protocol = tmp_path / "protocol_B_zscore_NChull_all"
    orig = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    for i in range(5):
        fold = protocol / f"B__stim_{i}"
        fold.mkdir(parents=True)
        np.save(fold / "fold_mean_orig.npy", orig)
        np.save(fold / "fold_mean_recon.npy", orig * 0.5)

    written = write_overview_batches(
        protocol,
        per_page=2,
        alias="all_folds_triplets.png",
        inclusive=True,
    )
    names = {p.name for p in written}
    assert "triplet_overview__batch01_of_01.png" in names
    assert "all_folds_triplets.png" in names
    assert not any("batch02" in p.name for p in written)
    index = (protocol / "overview" / "overview_index.txt").read_text()
    assert "n_folds=5" in index
    assert "batch 1/1:" in index


def test_overview_name_suffix_and_shared_clim_from_fold_means(tmp_path: Path):
    from experiments.loo_encoding.make_loo_triplet_overview import (
        write_overview_batches,
    )

    protocol = tmp_path / "protocol_B_raw_NChull_all"
    orig = np.full((4, 4), 1.0, dtype=np.float32)
    orig[1, 1] = 1.002
    for i in range(3):
        fold = protocol / f"B__stim_{i}"
        fold.mkdir(parents=True)
        np.save(fold / "fold_mean_orig.npy", orig)
        np.save(fold / "fold_mean_recon.npy", orig * 0.999)
        (fold / "sanity_orig_recon_residual.png").write_bytes(b"not-a-real-png")

    written = write_overview_batches(
        protocol,
        inclusive=True,
        from_fold_means=True,
        cmap="mapgeog_gray",
        orig_recon_vmin=0.998,
        orig_recon_vmax=1.003,
        residual_vmin=-0.0015,
        residual_vmax=0.0015,
        name_suffix="__gray_shared",
        alias="all_folds_triplets__gray_shared.png",
    )
    names = {p.name for p in written}
    assert "triplet_overview__gray_shared__batch01_of_01.png" in names
    assert "all_folds_triplets__gray_shared.png" in names
    assert "all_folds_triplets.png" not in names
    assert "triplet_overview__batch01_of_01.png" not in names
    index = (protocol / "overview" / "overview_index__gray_shared.txt").read_text()
    assert "cmap=mapgeog_gray" in index
    assert "orig_recon_vmin=0.998" in index
    assert "residual_vmin=-0.0015" in index
    assert not (protocol / "overview" / "overview_index.txt").exists()


def test_protocol_b_pooled_leaf_order():
    from experiments.loo_encoding.assemble_protocol_B_pooled_maps import (
        PROTOCOL_B_LEAF_ORDER,
    )

    names = [leaf for _, _, leaf in PROTOCOL_B_LEAF_ORDER]
    assert "protocol_B_zscore_NChull_all" in names
    assert "protocol_B_raw_NChull_all" in names
    assert all(n.startswith("protocol_B_") for n in names)


def test_run_loo_skip_post_encode_figures_flag():
    from experiments.loo_encoding.run_loo_encoding import parse_args

    args = parse_args(
        [
            "--window",
            "configs/windows/evoked_35_46.yaml",
            "--protocol",
            "B",
            "--skip-post-encode-figures",
        ]
    )
    assert args.skip_post_encode_figures is True


def test_cm_loo_parse_skip_fold_plots_and_train_only():
    from experiments.CM_model.run_cm_loo import parse_args

    args = parse_args(
        [
            "--window",
            "configs/windows/evoked_35_46.yaml",
            "--protocol",
            "A",
            "--heldout",
            "experiments/loo_encoding/heldout_letters.yaml",
            "--train-only-dates",
            "201118a",
            "--skip-fold-plots",
            "--no-save-model",
        ]
    )
    assert args.skip_fold_plots is True
    assert args.no_save_model is True
    assert args.train_only_dates == ["201118a"]
    assert args.heldout.as_posix().endswith("heldout_letters.yaml")
