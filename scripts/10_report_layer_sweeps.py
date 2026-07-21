#!/usr/bin/env python3
"""Select validation winners from layer sweeps, evaluate on test, and build a PDF report."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.evaluation.compare import collect_backbone_metrics
from src.evaluation.layer_sweep import (
    layer_comparison_csv_path,
    layer_sweep_dir,
    load_layer_comparison,
    model_slug_from_yaml,
    select_best_layer,
)
from src.evaluation.layer_sweep_report import (
    build_pdf_report,
    cross_model_report_dir,
    plot_validation_layers_overlay,
    plot_winner_comparison,
)
from src.paths import project_root


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


def _winner_rows(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    model_cfg_path: Path,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    slug = model_slug_from_yaml(model_cfg_path)
    csv_path = layer_comparison_csv_path(repo, cfg, window_id, slug)
    df = load_layer_comparison(csv_path)
    winner = select_best_layer(df, metric_col="eval_mean_r_masked")
    return slug, df, winner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/default.yaml",
    )
    parser.add_argument(
        "--window",
        type=Path,
        default=project_root() / "configs/windows/evoked_35_42.yaml",
    )
    parser.add_argument(
        "--ridge-config",
        type=Path,
        default=project_root() / "configs/ridge/default.yaml",
    )
    parser.add_argument(
        "--resnet-model",
        type=Path,
        default=project_root() / "configs/models/resnet18.yaml",
    )
    parser.add_argument(
        "--vgg-model",
        type=Path,
        default=project_root() / "configs/models/vgg16.yaml",
    )
    parser.add_argument(
        "--selection-split",
        type=str,
        default="val",
        help="Split used to choose the best layer (default: val)",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="test",
        help="Split used for final winner evaluation (default: test)",
    )
    parser.add_argument("--monkey", type=str, default=None)
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip stage 04 test evaluation; aggregate existing artifacts only",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Alias for --compare-only",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.skip_test_eval:
        args.compare_only = True

    repo = project_root()
    py = str(repo / "scripts" / "py")
    resnet_model = (
        args.resnet_model if args.resnet_model.is_absolute() else repo / args.resnet_model
    )
    vgg_model = args.vgg_model if args.vgg_model.is_absolute() else repo / args.vgg_model

    cfg = _merge_config(args.config, args.window)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey
    window_id = cfg.get("window_id") or (
        f"win_{int(cfg['start_frame']):04d}_{int(cfg['end_frame']):04d}"
    )

    resnet_slug, resnet_val_df, resnet_winner = _winner_rows(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        model_cfg_path=resnet_model,
    )
    vgg_slug, vgg_val_df, vgg_winner = _winner_rows(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        model_cfg_path=vgg_model,
    )

    winners = [
        ("resnet", resnet_slug, resnet_model, resnet_winner["feature_layer"]),
        ("vgg", vgg_slug, vgg_model, vgg_winner["feature_layer"]),
    ]

    if not args.compare_only:
        for _label, slug, model_path, layer in winners:
            print(f"\n=== Test evaluation: {slug}/{layer} ===", flush=True)
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
                args.test_split,
            ]
            if args.monkey:
                cmd.extend(["--monkey", args.monkey])
            _run(cmd, repo=repo)

    test_winners_df = collect_backbone_metrics(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        model_cfg_paths=[resnet_model, vgg_model],
        split=args.test_split,
        feature_layers=[resnet_winner["feature_layer"], vgg_winner["feature_layer"]],
    )

    out_dir = cross_model_report_dir(repo, cfg, window_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    winners_summary = {
        "window_id": window_id,
        "monkey": cfg["monkey"],
        "selection_split": args.selection_split,
        "test_split": args.test_split,
        "selection_metric": "eval_mean_r_masked",
        "resnet18": {
            "model_slug": resnet_slug,
            "winner_layer": resnet_winner["feature_layer"],
            "validation_metric": resnet_winner["metric_value"],
            "validation_row": resnet_winner["row"],
        },
        "vgg16": {
            "model_slug": vgg_slug,
            "winner_layer": vgg_winner["feature_layer"],
            "validation_metric": vgg_winner["metric_value"],
            "validation_row": vgg_winner["row"],
        },
        "validation_sweeps": {
            resnet_slug: str(
                layer_sweep_dir(repo, cfg, window_id, resnet_slug).relative_to(repo)
            ),
            vgg_slug: str(layer_sweep_dir(repo, cfg, window_id, vgg_slug).relative_to(repo)),
        },
    }
    with (out_dir / "winner_summary.json").open("w") as f:
        json.dump(winners_summary, f, indent=2)

    combined_val = pd.concat(
        [
            resnet_val_df.assign(backbone="resnet18"),
            vgg_val_df.assign(backbone="vgg16"),
        ],
        ignore_index=True,
    )
    combined_val.to_csv(out_dir / "validation_layer_comparison.csv", index=False)
    test_winners_df.to_csv(out_dir / "test_winner_metrics.csv", index=False)

    winner_plot = out_dir / f"winner_comparison_{args.test_split}.png"
    plot_winner_comparison(
        test_winners_df,
        winner_plot,
        split=args.test_split,
        metric_col="eval_mean_r_masked",
    )
    val_overlay = out_dir / f"validation_layers_{args.selection_split}.png"
    plot_validation_layers_overlay(
        resnet_val_df,
        vgg_val_df,
        val_overlay,
        selection_split=args.selection_split,
    )

    def _eval_figure(slug: str, layer: str, stem: str) -> Path:
        eval_root = repo / cfg["paths"].get("evaluation_plots_root", "plots/evaluation")
        return eval_root / cfg["monkey"] / window_id / slug / layer / f"{stem}_{args.test_split}.png"

    figure_paths = {
        "winner_comparison": winner_plot,
        "resnet_pixel_corr": _eval_figure(
            resnet_slug, resnet_winner["feature_layer"], "pixel_correlation"
        ),
        "resnet_mean_maps": _eval_figure(
            resnet_slug, resnet_winner["feature_layer"], "pixel_mean_maps"
        ),
        "vgg_pixel_corr": _eval_figure(
            vgg_slug, vgg_winner["feature_layer"], "pixel_correlation"
        ),
        "vgg_mean_maps": _eval_figure(
            vgg_slug, vgg_winner["feature_layer"], "pixel_mean_maps"
        ),
    }

    pdf_path = out_dir / "backbone_layer_sweep_report.pdf"
    build_pdf_report(
        repo=repo,
        pdf_path=pdf_path,
        cfg=cfg,
        window_id=window_id,
        selection_split=args.selection_split,
        test_split=args.test_split,
        resnet_slug=resnet_slug,
        vgg_slug=vgg_slug,
        resnet_val_df=resnet_val_df,
        vgg_val_df=vgg_val_df,
        resnet_winner=resnet_winner,
        vgg_winner=vgg_winner,
        test_winners_df=test_winners_df,
        figure_paths=figure_paths,
    )

    print("\nCross-model layer-sweep report complete")
    print(
        f"ResNet18 winner: {resnet_winner['feature_layer']} "
        f"(val eval_mean_r_masked={resnet_winner['metric_value']:.3f})"
    )
    print(
        f"VGG16 winner: {vgg_winner['feature_layer']} "
        f"(val eval_mean_r_masked={vgg_winner['metric_value']:.3f})"
    )
    if not test_winners_df.empty:
        cols = [
            "model_slug",
            "feature_layer",
            "eval_mean_r_masked",
            "r_mean_test_masked",
        ]
        present = [c for c in cols if c in test_winners_df.columns]
        print(test_winners_df[present].to_string(index=False))
    print(f"Report dir: {out_dir.relative_to(repo)}")
    print(f"PDF: {pdf_path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
