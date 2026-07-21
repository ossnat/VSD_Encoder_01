"""PDF and plot helpers for cross-backbone layer-sweep reports."""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.evaluation.compare import comparison_output_dir
from src.evaluation.layer_sweep import layer_sweep_dir


def cross_model_report_dir(repo: Path, cfg: dict, window_id: str) -> Path:
    return comparison_output_dir(repo, cfg, window_id) / "cross_model_report"


def plot_winner_comparison(
    winners_df: pd.DataFrame,
    output_path: Path,
    *,
    split: str,
    metric_col: str = "eval_mean_r_masked",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{row.model_slug}\n{row.feature_layer}" for row in winners_df.itertuples(index=False)
    ]
    vals = pd.to_numeric(winners_df[metric_col], errors="coerce").to_numpy(dtype=float)
    colors = ["lightgray" if not np.isfinite(v) else "steelblue" for v in vals]
    heights = np.where(np.isfinite(vals), vals, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.2), 4))
    bars = ax.bar(range(len(labels)), heights, width=0.55, color=colors)
    ax.set_xticks(range(len(labels)), labels, rotation=0)
    ax.set_ylabel("Mean pixel r (masked)")
    ax.set_title(f"Winner comparison | split={split}")
    ax.axhline(0.0, color="k", linewidth=0.5)
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size:
        ymax = float(np.nanmax(finite_vals))
        ax.set_ylim(bottom=min(0.0, float(np.nanmin(finite_vals)) * 1.1), top=ymax * 1.15)
    for bar, val in zip(bars, vals):
        label = f"{val:.3f}" if np.isfinite(val) else "N/A"
        y = bar.get_height() if np.isfinite(val) else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_validation_layers_overlay(
    resnet_val_df: pd.DataFrame,
    vgg_val_df: pd.DataFrame,
    output_path: Path,
    *,
    selection_split: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, df, title in [
        (axes[0], resnet_val_df, f"ResNet18 ({selection_split})"),
        (axes[1], vgg_val_df, f"VGG16 ({selection_split})"),
    ]:
        vals = pd.to_numeric(df["eval_mean_r_masked"], errors="coerce").to_numpy(dtype=float)
        labels = df["feature_layer"].tolist()
        colors = ["lightgray" if not np.isfinite(v) else "steelblue" for v in vals]
        heights = np.where(np.isfinite(vals), vals, 0.0)
        ax.bar(range(len(labels)), heights, color=colors)
        ax.set_xticks(range(len(labels)), labels, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel("eval_mean_r_masked")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _df_to_table_text(df: pd.DataFrame, columns: list[str]) -> str:
    present = [c for c in columns if c in df.columns]
    if not present:
        return "(no data)"
    return df[present].to_string(index=False)


def _add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.text(0.08, 0.95, title, fontsize=16, fontweight="bold", va="top")
    wrapped = "\n\n".join(textwrap.fill(p, width=95) for p in body.split("\n\n"))
    fig.text(0.08, 0.90, wrapped, fontsize=10, va="top", family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    if not image_path.exists():
        _add_text_page(pdf, title, f"Missing figure: {image_path}")
        return
    try:
        img = plt.imread(image_path)
    except Exception:
        _add_text_page(pdf, title, f"Could not read figure: {image_path}")
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def comparative_analysis_text(
    *,
    resnet_winner: dict[str, Any],
    vgg_winner: dict[str, Any],
    test_winners_df: pd.DataFrame,
    selection_split: str,
    test_split: str,
) -> str:
    lines = [
        "Selection criterion: highest eval_mean_r_masked on the validation split only.",
        f"Validation winners were chosen from split={selection_split}; "
        f"test metrics below use split={test_split}.",
        "",
        f"ResNet18 winner: {resnet_winner['feature_layer']} "
        f"(val eval_mean_r_masked={resnet_winner['metric_value']:.3f})",
        f"VGG16 winner: {vgg_winner['feature_layer']} "
        f"(val eval_mean_r_masked={vgg_winner['metric_value']:.3f})",
        "",
    ]
    if not test_winners_df.empty and "eval_mean_r_masked" in test_winners_df.columns:
        best_test = test_winners_df.sort_values(
            "eval_mean_r_masked", ascending=False, na_position="last"
        ).iloc[0]
        lines.append(
            f"On {test_split}, the higher masked pixel correlation is "
            f"{best_test.model_slug}/{best_test.feature_layer} "
            f"(eval_mean_r_masked={best_test.eval_mean_r_masked:.3f})."
        )
        if len(test_winners_df) == 2:
            other = test_winners_df[
                ~(
                    (test_winners_df["model_slug"] == best_test.model_slug)
                    & (test_winners_df["feature_layer"] == best_test.feature_layer)
                )
            ].iloc[0]
            delta = float(best_test.eval_mean_r_masked) - float(
                other.eval_mean_r_masked
            )
            lines.append(
                f"Margin over the other winner: {delta:+.3f} masked pixel r."
            )
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "- Validation layer choice avoids peeking at the test split.",
            "- Shallow VGG blocks were omitted from the sweep because dense Ridge "
            "matrices are memory-intensive on the expanded dataset.",
            "- Compare trial-wise Ridge metrics (r_mean_*_masked) alongside pixel-wise "
            "eval metrics when judging generalization.",
        ]
    )
    return "\n".join(lines)


def build_pdf_report(
    *,
    repo: Path,
    pdf_path: Path,
    cfg: dict,
    window_id: str,
    selection_split: str,
    test_split: str,
    resnet_slug: str,
    vgg_slug: str,
    resnet_val_df: pd.DataFrame,
    vgg_val_df: pd.DataFrame,
    resnet_winner: dict[str, Any],
    vgg_winner: dict[str, Any],
    test_winners_df: pd.DataFrame,
    figure_paths: dict[str, Path],
) -> Path:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    title_body = "\n".join(
        [
            f"Monkey: {cfg['monkey']}",
            f"Window: {window_id} [{cfg['start_frame']}, {cfg['end_frame']})",
            f"Selection split: {selection_split}",
            f"Test evaluation split: {test_split}",
            f"ResNet18 sweep layers: {', '.join(resnet_val_df['feature_layer'].tolist())}",
            f"VGG16 sweep layers: {', '.join(vgg_val_df['feature_layer'].tolist())}",
            f"Generated: {created}",
        ]
    )

    with PdfPages(pdf_path) as pdf:
        _add_text_page(pdf, "Backbone Layer Sweep Report", title_body)

        resnet_plot = layer_sweep_dir(repo, cfg, window_id, resnet_slug) / "layer_mean_pixel_r.png"
        _add_image_page(
            pdf,
            resnet_plot,
            f"ResNet18 validation layer sweep ({selection_split})",
        )
        _add_text_page(
            pdf,
            "ResNet18 validation metrics",
            _df_to_table_text(
                resnet_val_df,
                [
                    "feature_layer",
                    "feature_shape",
                    "eval_mean_r_masked",
                    "r_mean_val_masked",
                    "eval_n_trials",
                ],
            ),
        )

        vgg_plot = layer_sweep_dir(repo, cfg, window_id, vgg_slug) / "layer_mean_pixel_r.png"
        _add_image_page(
            pdf,
            vgg_plot,
            f"VGG16 validation layer sweep ({selection_split})",
        )
        _add_text_page(
            pdf,
            "VGG16 validation metrics",
            _df_to_table_text(
                vgg_val_df,
                [
                    "feature_layer",
                    "feature_shape",
                    "eval_mean_r_masked",
                    "r_mean_val_masked",
                    "eval_n_trials",
                ],
            ),
        )

        winner_body = "\n".join(
            [
                f"ResNet18 winner: {resnet_winner['feature_layer']} "
                f"(eval_mean_r_masked={resnet_winner['metric_value']:.3f})",
                f"VGG16 winner: {vgg_winner['feature_layer']} "
                f"(eval_mean_r_masked={vgg_winner['metric_value']:.3f})",
            ]
        )
        _add_text_page(pdf, "Validation winners", winner_body)

        _add_image_page(
            pdf,
            figure_paths["winner_comparison"],
            f"Test-set winner comparison ({test_split})",
        )
        _add_text_page(
            pdf,
            f"Test metrics ({test_split})",
            _df_to_table_text(
                test_winners_df,
                [
                    "model_slug",
                    "feature_layer",
                    "eval_mean_r_masked",
                    "eval_mean_r2_masked",
                    "r_mean_test_masked",
                    "eval_n_trials",
                ],
            ),
        )

        for key, title in [
            ("resnet_pixel_corr", "ResNet18 winner: pixel correlation"),
            ("resnet_mean_maps", "ResNet18 winner: trial-mean maps"),
            ("vgg_pixel_corr", "VGG16 winner: pixel correlation"),
            ("vgg_mean_maps", "VGG16 winner: trial-mean maps"),
        ]:
            _add_image_page(pdf, figure_paths[key], title)

        analysis = comparative_analysis_text(
            resnet_winner=resnet_winner,
            vgg_winner=vgg_winner,
            test_winners_df=test_winners_df,
            selection_split=selection_split,
            test_split=test_split,
        )
        _add_text_page(pdf, "Comparative analysis", analysis)

    return pdf_path
