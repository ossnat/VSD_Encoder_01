#!/usr/bin/env python3
"""Extract DNN feature maps from averaged trial NetCDF files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.DL_features.backbone import (
    DEFAULT_FEATURE_LAYER,
    FEATURE_LAYERS,
    build_feature_extractor,
    default_device,
)
from src.DL_features.extract import extract_features
from src.DL_features.schema import feature_dir, map_path, model_slug
from src.paths import project_root, resolve_data_path, workspace_root


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_cfg(default_cfg: Path, window_cfg: Path, model_cfg: Path) -> dict:
    cfg = _load_yaml(default_cfg)
    cfg.update(_load_yaml(window_cfg))
    cfg["model"] = _load_yaml(model_cfg)
    return cfg


def _portable_path(abs_path: Path, repo_root: Path) -> str:
    ws = workspace_root(repo_root)
    resolved = abs_path.resolve()
    try:
        return str(resolved.relative_to(ws))
    except ValueError:
        return str(resolved)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/default.yaml",
    )
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root() / "configs/models/resnet18.yaml",
    )
    parser.add_argument("--monkey", type=str, default=None)
    parser.add_argument(
        "--feature-layer",
        type=str,
        default=None,
        choices=FEATURE_LAYERS,
        help=f"Override model config (default: {DEFAULT_FEATURE_LAYER})",
    )
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    cfg = _merge_cfg(args.config, args.window, args.model)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey

    monkey = cfg["monkey"]
    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])
    window_id = cfg.get("window_id") or f"win_{start_frame:04d}_{end_frame:04d}"

    averaged_root = resolve_data_path(cfg["paths"]["averaged_root"], repo)
    features_root = resolve_data_path(cfg["paths"]["dl_features_root"], repo)

    averaged_manifest = averaged_root / monkey / window_id / "manifest.parquet"
    if not averaged_manifest.exists():
        raise FileNotFoundError(f"Averaged manifest not found: {averaged_manifest}")
    manifest = pd.read_parquet(averaged_manifest)
    if args.max_trials is not None:
        manifest = manifest.head(args.max_trials).copy()
    if manifest.empty:
        raise RuntimeError("No trials to process in averaged manifest")

    model_cfg = cfg["model"]
    backbone_name = model_cfg["name"]
    pretrained = bool(model_cfg.get("pretrained", True))
    feature_layer = args.feature_layer or model_cfg.get(
        "feature_layer", DEFAULT_FEATURE_LAYER
    )

    model = build_feature_extractor(
        backbone_name,
        pretrained=pretrained,
        feature_layer=feature_layer,
    )

    if args.device == "auto":
        device = default_device()
    else:
        import torch

        device = torch.device(args.device)

    model_name = model_slug(backbone_name, pretrained)
    out_dir = feature_dir(features_root, monkey, window_id, model_name, feature_layer)
    out_dir.mkdir(parents=True, exist_ok=True)

    path_fn = lambda tid: map_path(
        features_root, monkey, window_id, model_name, feature_layer, tid
    )
    feat_df, n_written, n_skipped = extract_features(
        model,
        manifest,
        repo_root=repo,
        map_path_fn=path_fn,
        feature_layer=feature_layer,
        input_size=int(model_cfg.get("input_size", 224)),
        input_channels=int(model_cfg.get("input_channels", 3)),
        input_scaling=str(model_cfg.get("input_scaling", "none")),
        imagenet_normalize=bool(model_cfg.get("imagenet_normalize", True)),
        batch_size=int(model_cfg.get("batch_size", 32)),
        device=device,
        overwrite=args.overwrite,
    )

    feat_manifest = out_dir / "manifest.parquet"
    feat_df.to_parquet(feat_manifest, index=False)

    sample_shape = feat_df.iloc[0]["feature_shape"] if not feat_df.empty else None
    run_cfg = {
        "monkey": monkey,
        "window_id": window_id,
        "model_name": backbone_name,
        "model_slug": model_name,
        "pretrained": pretrained,
        "feature_layer": feature_layer,
        "feature_shape": sample_shape,
        "input_size": int(model_cfg.get("input_size", 224)),
        "input_channels": int(model_cfg.get("input_channels", 3)),
        "input_scaling": str(model_cfg.get("input_scaling", "none")),
        "imagenet_normalize": bool(model_cfg.get("imagenet_normalize", True)),
        "batch_size": int(model_cfg.get("batch_size", 32)),
        "device": str(device),
        "n_trials_manifest": int(len(manifest)),
        "n_maps_written": int(n_written),
        "n_maps_skipped_existing": int(n_skipped),
        "averaged_manifest": _portable_path(averaged_manifest, repo),
        "feature_manifest": _portable_path(feat_manifest, repo),
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"Monkey: {monkey}")
    print(f"Window: {window_id}")
    print(f"Backbone: {backbone_name} ({model_name})")
    print(f"Feature layer: {feature_layer}")
    if sample_shape:
        print(f"Map shape (C, H, W): {sample_shape}")
    print(f"Device: {device}")
    print(f"Trials in input manifest: {len(manifest)}")
    print(f"Maps written: {n_written} (skipped existing: {n_skipped})")
    print(f"Feature manifest: {_portable_path(feat_manifest, repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
