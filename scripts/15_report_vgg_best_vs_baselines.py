#!/usr/bin/env python3
"""Compare best VGG16 layer vs ResNet18 and CORnet-S on the test set.

Selects the best VGG tap on the validation split (masked pixel r), then builds
a test-set PDF against ResNet18 layer3 and the best CORnet layer from a
candidate set (default: V1_pool7 V1_pool14 V2 V4).

Example:
  scripts/py scripts/15_report_vgg_best_vs_baselines.py \\
    --window configs/windows/evoked_35_42.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.DL_features.schema import model_slug
from src.evaluation.compare import collect_backbone_metrics, comparison_output_dir
from src.evaluation.layer_sweep import pixel_eval_json_path, select_best_layer
from src.evaluation.test_report_pdf import write_backbone_test_pdf
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
        "--vgg-model",
        type=Path,
        default=project_root() / "configs/models/vgg16.yaml",
    )
    p.add_argument(
        "--resnet-model",
        type=Path,
        default=project_root() / "configs/models/resnet18.yaml",
    )
    p.add_argument(
        "--cornet-model",
        type=Path,
        default=project_root() / "configs/models/cornet_s.yaml",
    )
    p.add_argument("--vgg-layers", nargs="+", default=DEFAULT_VGG_LAYERS)
    p.add_argument("--cornet-layers", nargs="+", default=DEFAULT_CORNET_LAYERS)
    p.add_argument("--resnet-layer", type=str, default=DEFAULT_RESNET_LAYER)
    p.add_argument(
        "--vgg-layer",
        type=str,
        default=None,
        help="Force VGG layer (skip val-based selection)",
    )
    p.add_argument(
        "--cornet-layer",
        type=str,
        default=None,
        help="Force CORnet layer (skip val-based selection)",
    )
    p.add_argument("--selection-split", type=str, default="val")
    p.add_argument("--test-split", type=str, default="test")
    p.add_argument("--monkey", type=str, default=None)
    p.add_argument("--center-frac", type=float, default=0.5)
    p.add_argument("--force-eval", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    py = str(repo / "scripts" / "py")

    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else repo / p

    vgg_model = _abs(args.vgg_model)
    resnet_model = _abs(args.resnet_model)
    cornet_model = _abs(args.cornet_model)
    cfg = _merge_config(args.config, args.window)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey
    ridge_cfg = _load_yaml(args.ridge_config)
    window_id = cfg.get("window_id") or (
        f"win_{int(cfg['start_frame']):04d}_{int(cfg['end_frame']):04d}"
    )
    vgg_slug = model_slug(_load_yaml(vgg_model))
    resnet_slug = model_slug(_load_yaml(resnet_model))
    cornet_slug = model_slug(_load_yaml(cornet_model))
    sel_split = args.selection_split
    test_split = args.test_split

    # --- select winners on validation ---
    if args.vgg_layer:
        vgg_best = args.vgg_layer
        vgg_sel_val = float("nan")
    else:
        vgg_val = collect_backbone_metrics(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_cfg_paths=[vgg_model],
            split=sel_split,
            feature_layers=list(args.vgg_layers),
        )
        pick = select_best_layer(vgg_val)
        vgg_best = pick["feature_layer"]
        vgg_sel_val = pick["metric_value"]
        print(
            f"Best VGG on {sel_split}: {vgg_best} "
            f"(masked r={vgg_sel_val:.3f})",
            flush=True,
        )

    if args.cornet_layer:
        cornet_best = args.cornet_layer
        cornet_sel_val = float("nan")
    else:
        cornet_val = collect_backbone_metrics(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_cfg_paths=[cornet_model],
            split=sel_split,
            feature_layers=list(args.cornet_layers),
        )
        pick = select_best_layer(cornet_val)
        cornet_best = pick["feature_layer"]
        cornet_sel_val = pick["metric_value"]
        print(
            f"Best CORnet on {sel_split}: {cornet_best} "
            f"(masked r={cornet_sel_val:.3f})",
            flush=True,
        )

    jobs = [
        (vgg_model, vgg_slug, vgg_best),
        (resnet_model, resnet_slug, args.resnet_layer),
        (cornet_model, cornet_slug, cornet_best),
    ]

    if not args.skip_eval:
        for model_path, slug, layer in jobs:
            out_json = pixel_eval_json_path(
                repo=repo,
                cfg=cfg,
                window_id=window_id,
                model_slug_str=slug,
                feature_layer=layer,
                split=test_split,
            )
            if out_json.exists() and not args.force_eval:
                print(f"Skip eval (exists): {slug}/{layer} {test_split}", flush=True)
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
                test_split,
            ]
            if args.monkey:
                cmd.extend(["--monkey", args.monkey])
            _run(cmd, repo=repo)

    frames = []
    for model_path, _slug, layer in jobs:
        frames.append(
            collect_backbone_metrics(
                repo=repo,
                cfg=cfg,
                window_id=window_id,
                model_cfg_paths=[model_path],
                split=test_split,
                feature_layers=[layer],
            )
        )
    df = pd.concat(frames, ignore_index=True)

    out_dir = (
        comparison_output_dir(repo, cfg, window_id) / "vgg_best_vs_baselines_test_report"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "test_metrics.csv", index=False)
    meta = {
        "window_id": window_id,
        "monkey": cfg["monkey"],
        "selection_split": sel_split,
        "test_split": test_split,
        "vgg_best_layer": vgg_best,
        "vgg_selection_metric": vgg_sel_val,
        "cornet_best_layer": cornet_best,
        "cornet_selection_metric": cornet_sel_val,
        "resnet_layer": args.resnet_layer,
        "rows": df.to_dict(orient="records"),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with (out_dir / "test_metrics.json").open("w") as f:
        json.dump(meta, f, indent=2)

    pdf_path = write_backbone_test_pdf(
        repo=repo,
        cfg=cfg,
        ridge_cfg=ridge_cfg,
        window_id=window_id,
        out_dir=out_dir,
        pdf_name="vgg_best_vs_baselines_test_report.pdf",
        report_title="Best VGG16 vs ResNet18 vs CORnet-S — Test Report",
        df=df,
        jobs=jobs,
        split=test_split,
        center_frac=args.center_frac,
        method_extra_lines=[
            f"VGG winner selected on {sel_split}: {vgg_best} "
            f"(masked r={vgg_sel_val:.3f}).",
            f"CORnet winner selected on {sel_split}: {cornet_best} "
            f"(masked r={cornet_sel_val:.3f}).",
            f"ResNet fixed reference: {args.resnet_layer}.",
        ],
    )

    print("\nCross-backbone test report complete")
    if not df.empty:
        cols = [
            c
            for c in [
                "model_slug",
                "feature_layer",
                "eval_mean_r_masked",
                "r_mean_test_masked",
            ]
            if c in df.columns
        ]
        print(df[cols].to_string(index=False))
    print(f"Report dir: {out_dir.relative_to(repo)}")
    print(f"PDF: {pdf_path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
