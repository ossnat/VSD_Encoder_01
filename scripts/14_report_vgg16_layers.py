#!/usr/bin/env python3
"""Test-set PDF report for all swept VGG16 layers.

Runs stage 04 on --test-split for each VGG layer (unless --skip-eval), then
writes a multi-page PDF analogous to the CORnet vs ResNet report:

  - metrics / parameter table
  - all test conditions: orig | recon | residual
  - final full test-set per-pixel r heatmaps
  - weight/alpha grids + analysis

Example:
  scripts/py scripts/14_report_vgg16_layers.py \\
    --window configs/windows/evoked_35_42.yaml --skip-eval
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.DL_features.schema import model_slug
from src.evaluation.compare import collect_backbone_metrics, comparison_output_dir
from src.evaluation.layer_sweep import pixel_eval_json_path
from src.evaluation.test_report_pdf import write_backbone_test_pdf
from src.paths import project_root

# Keep layer list in sync with scripts/13_sweep_vgg16_layers.py
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
        "--model",
        type=Path,
        default=project_root() / "configs/models/vgg16.yaml",
    )
    p.add_argument("--layers", nargs="+", default=DEFAULT_VGG_LAYERS)
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
    model_path = args.model if args.model.is_absolute() else repo / args.model
    cfg = _merge_config(args.config, args.window)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey
    ridge_cfg = _load_yaml(args.ridge_config)
    window_id = cfg.get("window_id") or (
        f"win_{int(cfg['start_frame']):04d}_{int(cfg['end_frame']):04d}"
    )
    slug = model_slug(_load_yaml(model_path))
    layers = list(args.layers)
    split = args.test_split

    if not args.skip_eval:
        for layer in layers:
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

    df = collect_backbone_metrics(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        model_cfg_paths=[model_path],
        split=split,
        feature_layers=layers,
    )
    out_dir = comparison_output_dir(repo, cfg, window_id) / "vgg16_test_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "test_metrics.csv", index=False)
    with (out_dir / "test_metrics.json").open("w") as f:
        json.dump(
            {
                "window_id": window_id,
                "monkey": cfg["monkey"],
                "split": split,
                "layers": layers,
                "rows": df.to_dict(orient="records"),
                "created": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    jobs = [(model_path, slug, layer) for layer in layers]
    pdf_path = write_backbone_test_pdf(
        repo=repo,
        cfg=cfg,
        ridge_cfg=ridge_cfg,
        window_id=window_id,
        out_dir=out_dir,
        pdf_name="vgg16_test_report.pdf",
        report_title="VGG16 layers — Test Report",
        df=df,
        jobs=jobs,
        split=split,
        center_frac=args.center_frac,
        method_extra_lines=[
            "Within-backbone comparison of ImageNet VGG16 taps.",
            "Early blocks use adaptive avg-pool mega-pixels (block*_pool7/14).",
        ],
    )

    print("\nVGG16 test report complete")
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
