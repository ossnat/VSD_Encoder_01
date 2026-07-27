#!/usr/bin/env python3
"""Test-set comparison: CORnet-S layers vs ResNet18, with PDF report.

Runs stage 04 on --test-split for each configured model/layer (skips if JSON
exists unless --force-eval), then writes a multi-page PDF with:

  - method summary + test metrics / parameter table
  - all test conditions: orig | recon | residual (per model/layer)
  - final full test-set per-pixel r heatmaps (one per model/layer)
  - center/periphery definition, weight/alpha grids, analysis text

Example:
  scripts/py scripts/12_report_cornet_vs_resnet.py \\
    --window configs/windows/evoked_35_42.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from src.DL_features.schema import model_slug
from src.evaluation.compare import collect_backbone_metrics, comparison_output_dir
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
from src.evaluation.layer_sweep import pixel_eval_json_path
from src.evaluation.mask import apply_mask_nan, mask_from_eval_cfg
from src.evaluation.plotting import plot_pixel_correlation_heatmap
from src.paths import project_root


# Default CORnet comparison taps for the test report PDF.
DEFAULT_CORNET_LAYERS = ["V1_pool7", "V1_pool14", "V2", "V4"]
DEFAULT_RESNET_LAYER = "layer3"


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(default_path: Path, window_path: Path) -> dict:
    cfg = _load_yaml(default_path)
    cfg.update(_load_yaml(window_path))
    return cfg


def _run(cmd: list[str], *, repo: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo, check=True)


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


def _plot_test_metric_bars(df: pd.DataFrame, output_path: Path, *, split: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{row.model_slug}\n{row.feature_layer}" for row in df.itertuples(index=False)
    ]
    vals = pd.to_numeric(df["eval_mean_r_masked"], errors="coerce").to_numpy(dtype=float)
    colors = []
    for slug in df["model_slug"]:
        colors.append("darkorange" if "resnet" in str(slug) else "steelblue")
    heights = np.where(np.isfinite(vals), vals, 0.0)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.6), 4.5))
    bars = ax.bar(range(len(labels)), heights, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)), labels, rotation=15, ha="right")
    ax.set_ylabel("Mean pixel r (masked)")
    ax.set_title(f"Test-set masked pixel correlation | split={split}")
    ax.axhline(0.0, color="k", linewidth=0.5)
    finite = vals[np.isfinite(vals)]
    if finite.size:
        ax.set_ylim(bottom=min(0.0, float(np.nanmin(finite)) * 1.1), top=float(np.nanmax(finite)) * 1.15)
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


def _analysis_text(
    df: pd.DataFrame,
    weight_df: pd.DataFrame,
    *,
    split: str,
) -> str:
    lines = [
        f"All metrics below are from stage 04 on split={split}.",
        "Backbones are ImageNet-pretrained (ResNet18 via torchvision; CORnet-S via DiCarlo Lab weights).",
        "Feature maps → RidgeCV → VSD reconstructions; pixel r is Pearson correlation across trials at each pixel, averaged inside the eval disk (radius 50).",
        "",
    ]
    if not df.empty and "eval_mean_r_masked" in df.columns:
        best = df.sort_values("eval_mean_r_masked", ascending=False, na_position="last").iloc[0]
        lines.append(
            f"Best {split} masked pixel r: {best.model_slug}/{best.feature_layer} "
            f"= {best.eval_mean_r_masked:.3f}."
        )
        cornet = df[df["model_slug"].astype(str).str.contains("cornet")]
        resnet = df[df["model_slug"].astype(str).str.contains("resnet")]
        if not cornet.empty and not resnet.empty:
            cbest = cornet.sort_values("eval_mean_r_masked", ascending=False).iloc[0]
            rbest = resnet.sort_values("eval_mean_r_masked", ascending=False).iloc[0]
            delta = float(cbest.eval_mean_r_masked) - float(rbest.eval_mean_r_masked)
            lines.append(
                f"Best CORnet ({cbest.feature_layer}={cbest.eval_mean_r_masked:.3f}) vs "
                f"ResNet ({rbest.feature_layer}={rbest.eval_mean_r_masked:.3f}): "
                f"delta={delta:+.3f}."
            )
        if not cornet.empty:
            lines.append("CORnet layer ranking (masked pixel r):")
            for row in cornet.sort_values("eval_mean_r_masked", ascending=False).itertuples():
                lines.append(f"  - {row.feature_layer}: {row.eval_mean_r_masked:.3f}")
    lines.extend(
        [
            "",
            "Center/periphery note:",
            "Center = inner disk at center_frac×eval radius; periphery = remaining eval annulus.",
            "This is a geometric proxy only (not anatomical V1/V2/V4 ROIs).",
            "Higher ||w||₂ center/periphery means Ridge puts more weight magnitude in the map center.",
            "Lower α in the center means less regularization there (more flexible fit).",
        ]
    )
    if not weight_df.empty and "weight_center_over_periphery" in weight_df.columns:
        lines.append("")
        lines.append("Weight center/periphery ratios:")
        for row in weight_df.itertuples():
            lines.append(
                f"  - {row.feature_layer}: ||w||₂ ratio={row.weight_center_over_periphery:.2f}"
            )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=project_root() / "configs/default.yaml")
    p.add_argument(
        "--window",
        type=Path,
        default=project_root() / "configs/windows/evoked_35_42.yaml",
    )
    p.add_argument(
        "--ridge-config",
        type=Path,
        default=project_root() / "configs/ridge/default.yaml",
    )
    p.add_argument(
        "--cornet-model",
        type=Path,
        default=project_root() / "configs/models/cornet_s.yaml",
    )
    p.add_argument(
        "--resnet-model",
        type=Path,
        default=project_root() / "configs/models/resnet18.yaml",
    )
    p.add_argument("--cornet-layers", nargs="+", default=DEFAULT_CORNET_LAYERS)
    p.add_argument("--resnet-layer", type=str, default=DEFAULT_RESNET_LAYER)
    p.add_argument("--test-split", type=str, default="test")
    p.add_argument("--monkey", type=str, default=None)
    p.add_argument("--center-frac", type=float, default=0.5)
    p.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run stage 04 even if pixel_evaluation_*.json exists",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only aggregate existing test artifacts into the PDF",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    py = str(repo / "scripts" / "py")
    cornet_model = (
        args.cornet_model if args.cornet_model.is_absolute() else repo / args.cornet_model
    )
    resnet_model = (
        args.resnet_model if args.resnet_model.is_absolute() else repo / args.resnet_model
    )
    cfg = _merge_config(args.config, args.window)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey
    ridge_cfg = _load_yaml(args.ridge_config)
    window_id = cfg.get("window_id") or (
        f"win_{int(cfg['start_frame']):04d}_{int(cfg['end_frame']):04d}"
    )
    cornet_slug = model_slug(_load_yaml(cornet_model))
    resnet_slug = model_slug(_load_yaml(resnet_model))
    split = args.test_split

    jobs: list[tuple[Path, str, str]] = [
        (cornet_model, cornet_slug, layer) for layer in args.cornet_layers
    ]
    jobs.append((resnet_model, resnet_slug, args.resnet_layer))

    if not args.skip_eval:
        for model_path, slug, layer in jobs:
            out_json = pixel_eval_json_path(
                repo=repo,
                cfg=cfg,
                window_id=window_id,
                model_slug_str=slug,
                feature_layer=layer,
                split=split,
            )
            if out_json.exists() and not args.force_eval:
                print(f"Skip eval (exists): {slug}/{layer} {split}", flush=True)
                continue
            cmd = [
                py,
                "scripts/04_evaluate_pixel_correlation.py",
                "--config",
                str(args.config),
                "--window",
                str(args.window),
                "--ridge-config",
                str(args.ridge_config),
                "--model",
                str(model_path),
                "--feature-layer",
                layer,
                "--split",
                split,
            ]
            if args.monkey:
                cmd.extend(["--monkey", args.monkey])
            _run(cmd, repo=repo)

    # Metrics table: collect per model with its layers
    frames = []
    frames.append(
        collect_backbone_metrics(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_cfg_paths=[cornet_model],
            split=split,
            feature_layers=list(args.cornet_layers),
        )
    )
    frames.append(
        collect_backbone_metrics(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_cfg_paths=[resnet_model],
            split=split,
            feature_layers=[args.resnet_layer],
        )
    )
    df = pd.concat(frames, ignore_index=True)

    out_dir = comparison_output_dir(repo, cfg, window_id) / "cornet_vs_resnet_test_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "test_metrics.csv", index=False)
    with (out_dir / "test_metrics.json").open("w") as f:
        json.dump(
            {
                "window_id": window_id,
                "monkey": cfg["monkey"],
                "split": split,
                "rows": df.to_dict(orient="records"),
                "created": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    bars_path = out_dir / f"test_masked_pixel_r.png"
    _plot_test_metric_bars(df, bars_path, split=split)

    spatial_size = tuple(int(x) for x in cfg.get("spatial_size", [100, 100]))
    eval_mask = mask_from_eval_cfg(ridge_cfg.get("evaluation", {}), spatial_size)
    def_path = out_dir / "center_periphery_definition.png"
    plot_center_periphery_definition(
        spatial_size,
        def_path,
        eval_mask=eval_mask,
        center_frac=args.center_frac,
    )

    weight_df = pd.DataFrame()
    try:
        cornet_w = summarize_layer_spatial_maps(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_slug=cornet_slug,
            feature_layers=list(args.cornet_layers),
            ridge_cfg=ridge_cfg,
            center_frac=args.center_frac,
        )
        resnet_w = summarize_layer_spatial_maps(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_slug=resnet_slug,
            feature_layers=[args.resnet_layer],
            ridge_cfg=ridge_cfg,
            center_frac=args.center_frac,
        )
        weight_df = pd.concat([cornet_w, resnet_w], ignore_index=True)
        weight_df.to_csv(out_dir / "weight_center_periphery.csv", index=False)
        plot_center_periphery_bars(
            weight_df,
            out_dir / "weight_center_over_periphery.png",
            title=f"||w||₂ center/periphery (CORnet + ResNet) | {window_id}",
        )
        plot_weight_alpha_grid(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_slug=cornet_slug,
            feature_layers=list(args.cornet_layers),
            output_path=out_dir / "cornet_weight_alpha_grid.png",
            ridge_cfg=ridge_cfg,
        )
        plot_weight_alpha_grid(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_slug=resnet_slug,
            feature_layers=[args.resnet_layer],
            output_path=out_dir / "resnet_weight_alpha_grid.png",
            ridge_cfg=ridge_cfg,
        )
    except FileNotFoundError as exc:
        print(f"WARNING: weight maps skipped: {exc}", flush=True)

    # Parameter / feature-count table (CSV + figure for PDF)
    param_entries = [(cornet_model, layer) for layer in args.cornet_layers]
    param_entries.append((resnet_model, args.resnet_layer))
    param_df = build_parameter_table(repo=repo, cfg=cfg, entries=param_entries)
    param_df.to_csv(out_dir / "parameter_counts.csv", index=False)
    param_fig = out_dir / "parameter_counts.png"
    plot_parameter_table(param_df, param_fig)

    model_layers = [(cornet_slug, layer) for layer in args.cornet_layers] + [
        (resnet_slug, args.resnet_layer)
    ]

    # Full test-set per-pixel r heatmaps (final comparison figures)
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
            apply_mask_nan(corr.astype(float), eval_mask) if eval_mask is not None else corr,
            heat_path,
            title=f"Test-set per-pixel r | {slug}/{layer} | mean={mean_r:.3f}",
        )
        heatmap_figs.append((f"{slug}/{layer}", heat_path))

    # All test conditions: orig / recon / residual for each model
    exemplar_figs: list[tuple[str, Path]] = []
    conditions = list_split_conditions(
        repo=repo, cfg=cfg, window_id=window_id, split=split
    )
    for date, condition, shape_type, stim_text in conditions:
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

    pdf_path = out_dir / "cornet_vs_resnet_test_report.pdf"
    analysis = _analysis_text(df, weight_df, split=split)
    method = "\n".join(
        [
            f"Monkey: {cfg['monkey']}",
            f"Window: {window_id} [{cfg['start_frame']}, {cfg['end_frame']})",
            f"Eval split: {split}",
            f"Models: CORnet {', '.join(args.cornet_layers)} + ResNet {args.resnet_layer}",
            f"Test conditions plotted: {len(conditions)}",
            "Condition panels: mean original | mean reconstruction | residual.",
            "Per-pixel r heatmaps (across all test trials) are final figures.",
            "n_ridge_features = C×H×W; n_backbone_params = CNN weight count.",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
        ]
    )

    param_text = (
        param_df.to_string(index=False) if not param_df.empty else "(no parameter table)"
    )

    with PdfPages(pdf_path) as pdf:
        _add_text_page(pdf, "CORnet-S vs ResNet18 — Test Report", method)
        _add_image_page(pdf, bars_path, f"Masked pixel r ({split})")
        cols = [
            "model_slug",
            "feature_layer",
            "feature_shape",
            "eval_mean_r_masked",
            "eval_mean_r2_masked",
            "r_mean_test_masked",
            "eval_n_trials",
        ]
        present = [c for c in cols if c in df.columns]
        _add_text_page(
            pdf,
            f"Test metrics table ({split})",
            df[present].to_string(index=False) if present else "(no data)",
        )
        _add_image_page(pdf, param_fig, "Parameter / feature counts")
        _add_text_page(
            pdf,
            "Parameter / feature counts (text)",
            "n_ridge_features = flattened feature-map size used by Ridge.\n"
            "n_backbone_params = CNN architecture parameter count "
            "(same across CORnet taps).\n\n"
            + param_text,
        )
        _add_image_page(pdf, def_path, "Center vs periphery definition")

        for title, fig_path in exemplar_figs:
            _add_image_page(pdf, fig_path, title)

        for title, heat_path in heatmap_figs:
            _add_image_page(
                pdf,
                heat_path,
                f"Final: test-set per-pixel r — {title}",
            )

        _add_image_page(
            pdf,
            out_dir / "cornet_weight_alpha_grid.png",
            "CORnet Ridge weight-norm and log α",
        )
        _add_image_page(
            pdf,
            out_dir / "resnet_weight_alpha_grid.png",
            "ResNet18 Ridge weight-norm and log α",
        )
        _add_image_page(
            pdf,
            out_dir / "weight_center_over_periphery.png",
            "Center/periphery ||w||₂ ratios (CORnet + ResNet)",
        )
        _add_text_page(pdf, "Comparative analysis", analysis)

    print("\nTest report complete")
    if not df.empty:
        print(
            df[
                [
                    c
                    for c in [
                        "model_slug",
                        "feature_layer",
                        "eval_mean_r_masked",
                        "r_mean_test_masked",
                    ]
                    if c in df.columns
                ]
            ].to_string(index=False)
        )
    print(f"Report dir: {out_dir.relative_to(repo)}")
    print(f"PDF: {pdf_path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
