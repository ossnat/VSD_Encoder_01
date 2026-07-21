from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.evaluation.layer_sweep import (
    plot_layer_mean_pixel_r,
    select_best_layer,
    validate_layer_artifacts,
)
from src.evaluation.layer_sweep_report import (
    build_pdf_report,
    comparative_analysis_text,
)


def _write_test_png(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    fig.savefig(path, dpi=50, bbox_inches="tight")
    plt.close(fig)


def test_select_best_layer_ignores_missing():
    df = pd.DataFrame(
        {
            "feature_layer": ["layer2", "layer3", "layer4"],
            "eval_mean_r_masked": [np.nan, 0.21, 0.19],
        }
    )
    winner = select_best_layer(df)
    assert winner["feature_layer"] == "layer3"
    assert winner["metric_value"] == pytest.approx(0.21)


def test_select_best_layer_raises_when_all_missing():
    df = pd.DataFrame(
        {
            "feature_layer": ["layer2"],
            "eval_mean_r_masked": [np.nan],
        }
    )
    with pytest.raises(ValueError, match="No finite values"):
        select_best_layer(df)


def test_plot_layer_mean_pixel_r_marks_missing_as_na(tmp_path: Path):
    df = pd.DataFrame(
        {
            "feature_layer": ["layer2", "layer3"],
            "eval_mean_r_masked": [np.nan, 0.25],
        }
    )
    out = plot_layer_mean_pixel_r(
        df,
        tmp_path / "layers.png",
        title="test",
        layer_order=["layer2", "layer3"],
    )
    assert out.exists()


def test_validate_layer_artifacts_reports_missing(tmp_path: Path):
    cfg = {
        "monkey": "gandalf",
        "paths": {
            "ridge_encode_root": "Data/VSD_Encoder_01/ridge_encode",
            "evaluation_plots_root": "plots/evaluation",
        },
    }
    missing = validate_layer_artifacts(
        repo=tmp_path,
        cfg=cfg,
        window_id="win_0035_0042",
        model_slug_str="resnet18_imagenet",
        feature_layer="layer3",
        split="val",
    )
    assert missing == ["ridge model/metrics", "pixel evaluation (val)"]


def test_build_pdf_report_creates_multipage_pdf(tmp_path: Path):
    cfg = {
        "monkey": "gandalf",
        "start_frame": 35,
        "end_frame": 42,
        "paths": {"evaluation_plots_root": "plots/evaluation"},
    }
    resnet_df = pd.DataFrame(
        {
            "feature_layer": ["layer2", "layer3"],
            "feature_shape": ["(128, 28, 28)", "(256, 14, 14)"],
            "eval_mean_r_masked": [0.18, 0.22],
            "r_mean_val_masked": [0.17, 0.20],
            "eval_n_trials": [270, 270],
        }
    )
    vgg_df = pd.DataFrame(
        {
            "feature_layer": ["block3", "block4"],
            "feature_shape": ["(256, 28, 28)", "(512, 14, 14)"],
            "eval_mean_r_masked": [0.16, 0.20],
            "r_mean_val_masked": [0.15, 0.19],
            "eval_n_trials": [270, 270],
        }
    )
    test_df = pd.DataFrame(
        {
            "model_slug": ["resnet18_imagenet", "vgg16_imagenet"],
            "feature_layer": ["layer3", "block4"],
            "eval_mean_r_masked": [0.24, 0.21],
            "eval_mean_r2_masked": [0.05, 0.04],
            "r_mean_test_masked": [0.23, 0.20],
            "eval_n_trials": [152, 152],
        }
    )
    figure_paths = {
        "winner_comparison": tmp_path / "winner.png",
        "resnet_pixel_corr": tmp_path / "resnet_corr.png",
        "resnet_mean_maps": tmp_path / "resnet_maps.png",
        "vgg_pixel_corr": tmp_path / "vgg_corr.png",
        "vgg_mean_maps": tmp_path / "vgg_maps.png",
    }
    for path in figure_paths.values():
        _write_test_png(path)

    pdf_path = tmp_path / "report.pdf"
    build_pdf_report(
        repo=tmp_path,
        pdf_path=pdf_path,
        cfg=cfg,
        window_id="win_0035_0042",
        selection_split="val",
        test_split="test",
        resnet_slug="resnet18_imagenet",
        vgg_slug="vgg16_imagenet",
        resnet_val_df=resnet_df,
        vgg_val_df=vgg_df,
        resnet_winner=select_best_layer(resnet_df),
        vgg_winner=select_best_layer(vgg_df),
        test_winners_df=test_df,
        figure_paths=figure_paths,
    )
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_comparative_analysis_mentions_winners():
    resnet_winner = select_best_layer(
        pd.DataFrame(
            {"feature_layer": ["layer3"], "eval_mean_r_masked": [0.22]}
        )
    )
    vgg_winner = select_best_layer(
        pd.DataFrame(
            {"feature_layer": ["block4"], "eval_mean_r_masked": [0.20]}
        )
    )
    test_df = pd.DataFrame(
        {
            "model_slug": ["resnet18_imagenet", "vgg16_imagenet"],
            "feature_layer": ["layer3", "block4"],
            "eval_mean_r_masked": [0.24, 0.21],
        }
    )
    text = comparative_analysis_text(
        resnet_winner=resnet_winner,
        vgg_winner=vgg_winner,
        test_winners_df=test_df,
        selection_split="val",
        test_split="test",
    )
    assert "layer3" in text
    assert "block4" in text
    assert "validation split" in text.lower()
