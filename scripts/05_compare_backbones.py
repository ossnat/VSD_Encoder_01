#!/usr/bin/env python3
"""Compare Ridge encoding metrics across stimulus backbones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.evaluation.compare import collect_backbone_metrics, comparison_output_dir
from src.paths import project_root


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(default_path: Path, window_path: Path) -> dict:
    cfg = _load_yaml(default_path)
    cfg.update(_load_yaml(window_path))
    return cfg


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
        "--model",
        type=Path,
        action="append",
        dest="models",
        help="Model YAML (repeatable). Default: all configs/models/*.yaml",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _plot_summary(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    labels = [
        f"{row.model_slug}\n{row.feature_layer}"
        for row in df.itertuples(index=False)
    ]
    x = range(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(labels) * 2.5), 4))

    r_test = df["r_mean_test"].fillna(0.0).to_numpy()
    axes[0].bar(x, r_test, width=0.6, color="steelblue")
    axes[0].set_xticks(list(x), labels, rotation=15, ha="right")
    axes[0].set_ylabel("Pearson r (trial mean)")
    axes[0].set_title("Ridge test-set correlation")
    axes[0].axhline(0.0, color="k", linewidth=0.5)

    eval_r = df["eval_mean_r"].fillna(0.0).to_numpy()
    axes[1].bar(x, eval_r, width=0.6, color="darkorange")
    axes[1].set_xticks(list(x), labels, rotation=15, ha="right")
    axes[1].set_ylabel("Mean pixel r")
    axes[1].set_title("Pixel-wise test correlation")
    axes[1].axhline(0.0, color="k", linewidth=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    cfg = _merge_config(args.config, args.window)
    window_id = cfg.get("window_id") or (
        f"win_{int(cfg['start_frame']):04d}_{int(cfg['end_frame']):04d}"
    )

    if args.models:
        model_paths = [p if p.is_absolute() else repo / p for p in args.models]
    else:
        model_paths = sorted((repo / "configs/models").glob("*.yaml"))

    df = collect_backbone_metrics(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        model_cfg_paths=model_paths,
        split=args.split,
    )

    out_dir = args.output_dir or comparison_output_dir(repo, cfg, window_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "backbone_comparison.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "window_id": window_id,
        "monkey": cfg["monkey"],
        "split": args.split,
        "n_models": int(len(df)),
        "models": df.to_dict(orient="records"),
    }
    with (out_dir / "backbone_comparison.json").open("w") as f:
        json.dump(summary, f, indent=2)

    _plot_summary(df, out_dir / "backbone_comparison.png")

    print(f"Compared {len(df)} backbone(s) for window {window_id}")
    if not df.empty:
        cols = [
            "model_slug",
            "feature_layer",
            "r_mean_test",
            "eval_mean_r",
            "eval_mean_r2",
            "feature_shape",
        ]
        print(df[cols].to_string(index=False))
    print(f"CSV: {csv_path.relative_to(repo)}")
    print(f"Plot: {(out_dir / 'backbone_comparison.png').relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
