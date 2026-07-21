#!/usr/bin/env python3
"""Sweep feature layers for one backbone: extract → ridge → pixel eval → bar plot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.DL_features.backbone import feature_layers_for_type
from src.DL_features.schema import model_slug
from src.evaluation.compare import collect_backbone_metrics
from src.evaluation.layer_sweep import (
    layer_sweep_dir,
    plot_layer_mean_pixel_r,
    validate_layer_artifacts,
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
        default=project_root() / "configs/windows/evoked_32_42.yaml",
    )
    parser.add_argument(
        "--ridge-config",
        type=Path,
        default=project_root() / "configs/ridge/default.yaml",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root() / "configs/models/resnet18.yaml",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Feature layers to sweep (default: all supported for this model type)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split passed to stage 04 during the sweep (default: val for layer selection)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--monkey", type=str, default=None)
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip stage 02b (reuse existing feature maps)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip stage 03 (reuse existing ridge models)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip stage 04 (reuse existing pixel evaluation)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only aggregate existing results and plot (implies all --skip-*)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    py = str(repo / "scripts" / "py")
    model_path = args.model if args.model.is_absolute() else repo / args.model
    model_cfg = _load_yaml(model_path)
    backbone_type = str(model_cfg.get("type", "resnet"))
    slug = model_slug(model_cfg)

    layers = args.layers or list(feature_layers_for_type(backbone_type))
    allowed = set(feature_layers_for_type(backbone_type))
    bad = [layer for layer in layers if layer not in allowed]
    if bad:
        raise ValueError(
            f"Unsupported layers for type={backbone_type!r}: {bad}. "
            f"Allowed: {sorted(allowed)}"
        )

    if args.compare_only:
        args.skip_extract = True
        args.skip_train = True
        args.skip_eval = True

    cfg = _merge_config(args.config, args.window)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey
    window_id = cfg.get("window_id") or (
        f"win_{int(cfg['start_frame']):04d}_{int(cfg['end_frame']):04d}"
    )

    for layer in layers:
        print(f"\n=== {slug} / {layer} ===", flush=True)
        if not args.skip_extract:
            cmd = [
                py,
                "scripts/02b_extract_stimulus_features.py",
                "--config",
                str(args.config),
                "--model",
                str(model_path),
                "--feature-layer",
                layer,
                "--device",
                args.device,
            ]
            if args.monkey:
                cmd.extend(["--monkey", args.monkey])
            _run(cmd, repo=repo)

        if not args.skip_train:
            cmd = [
                py,
                "scripts/03_train_ridge_encoder.py",
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
            ]
            if args.monkey:
                cmd.extend(["--monkey", args.monkey])
            _run(cmd, repo=repo)

        if not args.skip_eval:
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
                args.split,
            ]
            if args.monkey:
                cmd.extend(["--monkey", args.monkey])
            _run(cmd, repo=repo)

        missing = validate_layer_artifacts(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_slug_str=slug,
            feature_layer=layer,
            split=args.split,
            require_ridge=not args.skip_train,
            require_eval=not args.skip_eval,
        )
        if missing:
            print(
                f"WARNING: {slug}/{layer} missing artifacts after sweep: "
                + ", ".join(missing),
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
    csv_path = out_dir / "layer_comparison.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "window_id": window_id,
        "monkey": cfg["monkey"],
        "model_slug": slug,
        "split": args.split,
        "layers": layers,
        "rows": df.to_dict(orient="records"),
    }
    with (out_dir / "layer_comparison.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot_path = out_dir / "layer_mean_pixel_r.png"
    plot_layer_mean_pixel_r(
        df,
        plot_path,
        title=(
            f"{slug} | mean pixel r (masked) by layer | {window_id} | "
            f"split={args.split}"
        ),
        layer_order=layers,
    )

    print(f"\nLayer sweep complete for {slug}")
    cols = [
        "feature_layer",
        "feature_shape",
        "eval_mean_r_masked",
        "r_mean_val_masked",
        "r_mean_test_masked",
    ]
    present = [c for c in cols if c in df.columns]
    if not df.empty:
        print(df[present].to_string(index=False))
    print(f"CSV: {csv_path.relative_to(repo)}")
    print(f"Plot: {plot_path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
