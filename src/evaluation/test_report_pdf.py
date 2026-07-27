"""Shared helpers for multi-backbone test-set PDF reports."""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.evaluation.condition_report import (
    build_parameter_table,
    compute_split_corr_map,
    list_split_conditions,
    plot_condition_orig_recon_corr,
    plot_parameter_table,
    predict_condition_split,
)
from src.evaluation.cornet_compare import (
    plot_center_periphery_bars,
    plot_center_periphery_definition,
    plot_weight_alpha_grid,
    summarize_layer_spatial_maps,
)
from src.evaluation.mask import apply_mask_nan, mask_from_eval_cfg
from src.evaluation.plotting import plot_pixel_correlation_heatmap


def add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.text(0.08, 0.95, title, fontsize=16, fontweight="bold", va="top")
    wrapped = "\n\n".join(textwrap.fill(p, width=95) for p in body.split("\n\n"))
    fig.text(0.08, 0.90, wrapped, fontsize=10, va="top", family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    if not image_path.exists():
        add_text_page(pdf, title, f"Missing figure: {image_path}")
        return
    try:
        img = plt.imread(image_path)
    except Exception:
        add_text_page(pdf, title, f"Could not read figure: {image_path}")
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(
    df: pd.DataFrame,
    output_path: Path,
    *,
    split: str,
    title: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{row.model_slug}\n{row.feature_layer}" for row in df.itertuples(index=False)
    ]
    vals = pd.to_numeric(df["eval_mean_r_masked"], errors="coerce").to_numpy(dtype=float)
    colors = []
    for slug in df["model_slug"]:
        s = str(slug)
        if "resnet" in s:
            colors.append("darkorange")
        elif "vgg" in s:
            colors.append("seagreen")
        else:
            colors.append("steelblue")
    heights = np.where(np.isfinite(vals), vals, 0.0)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.35), 4.5))
    bars = ax.bar(range(len(labels)), heights, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylabel("Mean pixel r (masked)")
    ax.set_title(title or f"Masked pixel correlation | split={split}")
    ax.axhline(0.0, color="k", linewidth=0.5)
    finite = vals[np.isfinite(vals)]
    if finite.size:
        ax.set_ylim(
            bottom=min(0.0, float(np.nanmin(finite)) * 1.1),
            top=float(np.nanmax(finite)) * 1.15,
        )
    for bar, val in zip(bars, vals):
        if np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def analysis_text(
    df: pd.DataFrame,
    weight_df: pd.DataFrame,
    *,
    split: str,
    intro: str | None = None,
) -> str:
    lines = [
        intro
        or (
            f"All metrics below are from stage 04 on split={split}. "
            "Backbones are ImageNet-pretrained; features → RidgeCV → VSD maps."
        ),
        "Pixel r is Pearson correlation across trials at each pixel, averaged "
        "inside the eval disk (radius 50).",
        "",
    ]
    if not df.empty and "eval_mean_r_masked" in df.columns:
        best = df.sort_values(
            "eval_mean_r_masked", ascending=False, na_position="last"
        ).iloc[0]
        lines.append(
            f"Best {split} masked pixel r: {best.model_slug}/{best.feature_layer} "
            f"= {best.eval_mean_r_masked:.3f}."
        )
        lines.append(f"{split} ranking (masked pixel r):")
        for row in df.sort_values(
            "eval_mean_r_masked", ascending=False, na_position="last"
        ).itertuples():
            lines.append(
                f"  - {row.model_slug}/{row.feature_layer}: {row.eval_mean_r_masked:.3f}"
            )
    lines.extend(
        [
            "",
            "Center/periphery note:",
            "Center = inner disk at center_frac×eval radius; periphery = remaining annulus.",
            "Geometric proxy only (not anatomical V1/V2/V4 ROIs).",
        ]
    )
    if not weight_df.empty and "weight_center_over_periphery" in weight_df.columns:
        lines.append("")
        lines.append("Weight center/periphery ratios:")
        for row in weight_df.itertuples():
            slug = getattr(row, "model_slug", "")
            lines.append(
                f"  - {slug}/{row.feature_layer}: "
                f"||w||₂ ratio={row.weight_center_over_periphery:.2f}"
            )
    return "\n".join(lines)


def build_condition_and_heatmap_figs(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    out_dir: Path,
    model_layers: Sequence[tuple[str, str]],
    split: str,
    eval_mask: np.ndarray | None,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]], list[Any]]:
    """Return (exemplar_figs, heatmap_figs, conditions)."""
    heatmap_figs: list[tuple[str, Path]] = []
    for slug, layer in model_layers:
        corr, mean_r = compute_split_corr_map(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_slug_str=slug,
            feature_layer=layer,
            split=split,
            eval_mask=eval_mask,
        )
        heat_path = out_dir / f"pixel_r_{split}_{slug}_{layer}.png"
        plot_pixel_correlation_heatmap(
            apply_mask_nan(corr.astype(float), eval_mask)
            if eval_mask is not None
            else corr,
            heat_path,
            title=f"Test-set per-pixel r | {slug}/{layer} | mean={mean_r:.3f}",
        )
        heatmap_figs.append((f"{slug}/{layer}", heat_path))

    exemplar_figs: list[tuple[str, Path]] = []
    conditions = list_split_conditions(
        repo=repo, cfg=cfg, window_id=window_id, split=split
    )
    for date, condition, shape_type, _stim_text in conditions:
        tag = f"{date}_{condition}"
        for slug, layer in model_layers:
            try:
                payload = predict_condition_split(
                    repo=repo,
                    cfg=cfg,
                    window_id=window_id,
                    model_slug_str=slug,
                    feature_layer=layer,
                    date=date,
                    condition=condition,
                    split=split,
                    eval_mask=eval_mask,
                )
            except FileNotFoundError as exc:
                print(
                    f"WARNING: condition skipped {slug}/{layer} {tag}: {exc}",
                    flush=True,
                )
                continue
            fig_path = out_dir / f"condition_{tag}_{slug}_{layer}.png"
            plot_condition_orig_recon_corr(
                payload,
                fig_path,
                title_prefix=f"{slug}/{layer}",
                eval_mask=eval_mask,
            )
            exemplar_figs.append(
                (f"{date}/{condition} ({shape_type}): {slug}/{layer}", fig_path)
            )
    return exemplar_figs, heatmap_figs, conditions


def write_backbone_test_pdf(
    *,
    repo: Path,
    cfg: dict,
    ridge_cfg: dict,
    window_id: str,
    out_dir: Path,
    pdf_name: str,
    report_title: str,
    df: pd.DataFrame,
    jobs: Sequence[tuple[Path, str, str]],
    split: str = "test",
    center_frac: float = 0.5,
    method_extra_lines: Sequence[str] | None = None,
) -> Path:
    """
    Build figures + PDF for a list of (model_cfg_path, slug, layer) jobs.

    Layout matches the CORnet vs ResNet test report:
    metrics → params → conditions (orig|recon|residual) → final pixel-r heatmaps
    → weight maps → analysis.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    spatial_size = tuple(int(x) for x in cfg.get("spatial_size", [100, 100]))
    eval_mask = mask_from_eval_cfg(ridge_cfg.get("evaluation", {}), spatial_size)

    bars_path = out_dir / f"{split}_masked_pixel_r.png"
    plot_metric_bars(df, bars_path, split=split)

    def_path = out_dir / "center_periphery_definition.png"
    plot_center_periphery_definition(
        spatial_size,
        def_path,
        eval_mask=eval_mask,
        center_frac=center_frac,
    )

    # Weight summaries: group by slug
    weight_frames: list[pd.DataFrame] = []
    weight_grid_paths: list[tuple[str, Path]] = []
    by_slug: dict[str, list[str]] = {}
    for _model_path, slug, layer in jobs:
        by_slug.setdefault(slug, []).append(layer)
    try:
        for slug, layers in by_slug.items():
            w = summarize_layer_spatial_maps(
                repo=repo,
                cfg=cfg,
                window_id=window_id,
                model_slug=slug,
                feature_layers=layers,
                ridge_cfg=ridge_cfg,
                center_frac=center_frac,
            )
            weight_frames.append(w)
            grid_path = out_dir / f"{slug}_weight_alpha_grid.png"
            plot_weight_alpha_grid(
                repo=repo,
                cfg=cfg,
                window_id=window_id,
                model_slug=slug,
                feature_layers=layers,
                output_path=grid_path,
                ridge_cfg=ridge_cfg,
            )
            weight_grid_paths.append((slug, grid_path))
    except FileNotFoundError as exc:
        print(f"WARNING: weight maps skipped: {exc}", flush=True)

    weight_df = (
        pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    )
    if not weight_df.empty:
        weight_df.to_csv(out_dir / "weight_center_periphery.csv", index=False)
        plot_center_periphery_bars(
            weight_df,
            out_dir / "weight_center_over_periphery.png",
            title=f"||w||₂ center/periphery | {window_id}",
        )

    param_entries = [(model_path, layer) for model_path, _slug, layer in jobs]
    param_df = build_parameter_table(repo=repo, cfg=cfg, entries=param_entries)
    param_df.to_csv(out_dir / "parameter_counts.csv", index=False)
    param_fig = out_dir / "parameter_counts.png"
    plot_parameter_table(param_df, param_fig)

    model_layers = [(slug, layer) for _mp, slug, layer in jobs]
    exemplar_figs, heatmap_figs, conditions = build_condition_and_heatmap_figs(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        out_dir=out_dir,
        model_layers=model_layers,
        split=split,
        eval_mask=eval_mask,
    )

    method_lines = [
        f"Monkey: {cfg['monkey']}",
        f"Window: {window_id} [{cfg['start_frame']}, {cfg['end_frame']})",
        f"Eval split: {split}",
        f"Entries: {', '.join(f'{s}/{l}' for _, s, l in jobs)}",
        f"Test conditions plotted: {len(conditions)}",
        "Condition panels: mean original | mean reconstruction | residual.",
        "Per-pixel r heatmaps (across all test trials) are final figures.",
        "n_ridge_features = C×H×W; n_backbone_params = CNN weight count.",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
    ]
    if method_extra_lines:
        method_lines = list(method_extra_lines) + method_lines

    pdf_path = out_dir / pdf_name
    analysis = analysis_text(df, weight_df, split=split)
    param_text = (
        param_df.to_string(index=False) if not param_df.empty else "(no parameter table)"
    )

    with PdfPages(pdf_path) as pdf:
        add_text_page(pdf, report_title, "\n".join(method_lines))
        add_image_page(pdf, bars_path, f"Masked pixel r ({split})")
        cols = [
            "model_slug",
            "feature_layer",
            "feature_shape",
            "eval_mean_r_masked",
            "eval_mean_r2_masked",
            "r_mean_test_masked",
            "r_mean_val_masked",
            "eval_n_trials",
        ]
        present = [c for c in cols if c in df.columns]
        add_text_page(
            pdf,
            f"Metrics table ({split})",
            df[present].to_string(index=False) if present else "(no data)",
        )
        add_image_page(pdf, param_fig, "Parameter / feature counts")
        add_text_page(
            pdf,
            "Parameter / feature counts (text)",
            "n_ridge_features = flattened feature-map size used by Ridge.\n"
            "n_backbone_params = CNN architecture parameter count.\n\n"
            + param_text,
        )
        add_image_page(pdf, def_path, "Center vs periphery definition")

        for title, fig_path in exemplar_figs:
            add_image_page(pdf, fig_path, title)

        for title, heat_path in heatmap_figs:
            add_image_page(pdf, heat_path, f"Final: test-set per-pixel r — {title}")

        for slug, grid_path in weight_grid_paths:
            add_image_page(pdf, grid_path, f"{slug} Ridge weight-norm and log α")

        add_image_page(
            pdf,
            out_dir / "weight_center_over_periphery.png",
            "Center/periphery ||w||₂ ratios",
        )
        add_text_page(pdf, "Comparative analysis", analysis)

    return pdf_path
