#!/usr/bin/env python3
"""Sweep VGG16 layers (with early-block pooling) and summarize metrics.

Default layers:
  block1_pool7 block1_pool14 block2_pool7 block2_pool14 block3_pool14 block4 block5

Example:
  scripts/py scripts/13_sweep_vgg16_layers.py \\
    --window configs/windows/evoked_35_42.yaml \\
    --device cpu
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from src.DL_features.schema import model_slug
from src.evaluation.compare import collect_backbone_metrics, comparison_output_dir
from src.evaluation.cornet_compare import (
    plot_center_periphery_bars,
    plot_weight_alpha_grid,
    summarize_layer_spatial_maps,
)
from src.evaluation.layer_sweep import (
    layer_sweep_dir,
    plot_layer_mean_pixel_r,
    validate_layer_artifacts,
)
from src.paths import project_root


DEFAULT_VGG_LAYERS = [
    "block1_pool7",
    "block1_pool14",
    "block2_pool7",
    "block2_pool14",
    "block3_pool14",
    "block4",
    "block5",
]


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
        "--model",
        type=Path,
        default=project_root() / "configs/models/vgg16.yaml",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=DEFAULT_VGG_LAYERS,
        help="VGG taps to sweep",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--monkey", type=str, default=None)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only aggregate existing results / weight maps",
    )
    parser.add_argument("--center-frac", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    py = str(repo / "scripts" / "py")
    model_path = args.model if args.model.is_absolute() else repo / args.model
    model_cfg = _load_yaml(model_path)
    slug = model_slug(model_cfg)
    layers = list(args.layers)

    if args.compare_only:
        args.skip_extract = True
        args.skip_train = True
        args.skip_eval = True

    cfg = _merge_config(args.config, args.window)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey
    ridge_cfg = _load_yaml(args.ridge_config)
    window_id = cfg.get("window_id") or (
        f"win_{int(cfg['start_frame']):04d}_{int(cfg['end_frame']):04d}"
    )

    if not (args.skip_extract and args.skip_train and args.skip_eval):
        cmd = [
            py,
            "scripts/09_sweep_feature_layers.py",
            "--config",
            str(args.config),
            "--window",
            str(args.window),
            "--ridge-config",
            str(args.ridge_config),
            "--model",
            str(model_path),
            "--split",
            args.split,
            "--device",
            args.device,
            "--layers",
            *layers,
        ]
        if args.monkey:
            cmd.extend(["--monkey", args.monkey])
        if args.skip_extract:
            cmd.append("--skip-extract")
        if args.skip_train:
            cmd.append("--skip-train")
        if args.skip_eval:
            cmd.append("--skip-eval")
        _run(cmd, repo=repo)

    for layer in layers:
        missing = validate_layer_artifacts(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_slug_str=slug,
            feature_layer=layer,
            split=args.split,
            require_ridge=True,
            require_eval=True,
        )
        if missing:
            print(
                f"WARNING: {slug}/{layer} missing: " + ", ".join(missing),
                flush=True,
            )

    df = collect_backbone_metrics(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        model_cfg_paths=[model_path],
        split=args.split,
        feature_layers=layers,
    )

    out_dir = layer_sweep_dir(repo, cfg, window_id, slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = comparison_output_dir(repo, cfg, window_id) / "vgg16_comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "layer_comparison.csv"
    df.to_csv(csv_path, index=False)
    with (report_dir / "layer_comparison.json").open("w") as f:
        json.dump(
            {
                "window_id": window_id,
                "monkey": cfg["monkey"],
                "model_slug": slug,
                "split": args.split,
                "layers": layers,
                "rows": df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    plot_layer_mean_pixel_r(
        df,
        report_dir / "layer_mean_pixel_r.png",
        title=(
            f"{slug} | mean pixel r (masked) by layer | {window_id} | "
            f"split={args.split}"
        ),
        layer_order=layers,
    )

    try:
        stats_df = summarize_layer_spatial_maps(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_slug=slug,
            feature_layers=layers,
            ridge_cfg=ridge_cfg,
            center_frac=args.center_frac,
        )
        stats_df.to_csv(report_dir / "weight_center_periphery.csv", index=False)
        plot_center_periphery_bars(
            stats_df,
            report_dir / "weight_center_over_periphery.png",
            title=(
                f"{slug} | ||w||₂ center/periphery "
                f"(center_frac={args.center_frac}) | {window_id}"
            ),
        )
        plot_weight_alpha_grid(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_slug=slug,
            feature_layers=layers,
            output_path=report_dir / "weight_alpha_grid.png",
            ridge_cfg=ridge_cfg,
        )
        print("\nCenter/periphery weight ratios:")
        print(
            stats_df[
                [
                    "feature_layer",
                    "weight_center_over_periphery",
                    "weight_center_mean",
                    "weight_periphery_mean",
                ]
            ].to_string(index=False)
        )
    except FileNotFoundError as exc:
        print(f"WARNING: skipped weight/alpha maps: {exc}", flush=True)

    df.to_csv(out_dir / "layer_comparison.csv", index=False)

    print(f"\nVGG16 layer comparison complete for {slug}")
    if not df.empty:
        cols = [
            "feature_layer",
            "feature_shape",
            "eval_mean_r_masked",
            "r_mean_val_masked",
            "eval_n_trials",
        ]
        present = [c for c in cols if c in df.columns]
        print(df[present].to_string(index=False))
    print(f"Report dir: {report_dir.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
