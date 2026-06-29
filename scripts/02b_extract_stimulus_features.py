#!/usr/bin/env python3
"""Extract DNN feature maps from rendered stimulus images."""

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
from src.DL_features.extract_stimulus import extract_stimulus_features
from src.DL_features.schema import model_slug, stimulus_feature_dir, stimulus_map_path
from src.encoding.pairs import dedupe_stimulus_manifest
from src.paths import project_root, resolve_data_path, workspace_root
from src.stimuli.schema import manifest_path as stimulus_manifest_path


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


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
        help=f"Override ResNet feature layer (choices: {', '.join(FEATURE_LAYERS)})",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    base_cfg = _load_yaml(args.config)
    model_cfg = _load_yaml(args.model)
    if args.monkey is not None:
        base_cfg["monkey"] = args.monkey

    monkey = base_cfg["monkey"]
    stimuli_root = resolve_data_path(base_cfg["paths"]["stimuli_root"], repo)
    features_root = resolve_data_path(
        base_cfg["paths"]["dl_features_stimuli_root"], repo
    )

    stim_manifest_path = stimulus_manifest_path(stimuli_root, monkey)
    if not stim_manifest_path.exists():
        raise FileNotFoundError(
            f"Stimulus manifest not found: {stim_manifest_path}. Run stage 01b first."
        )
    manifest = dedupe_stimulus_manifest(pd.read_parquet(stim_manifest_path))
    if manifest.empty:
        raise RuntimeError("No stimuli to process in manifest")

    backbone_name = model_cfg["name"]
    feature_layer = args.feature_layer or model_cfg.get(
        "feature_layer", DEFAULT_FEATURE_LAYER
    )
    backbone_type = model_cfg.get("type", "resnet")
    if args.feature_layer and backbone_type != "resnet":
        raise ValueError("--feature-layer override is only supported for ResNet models")

    model = build_feature_extractor(model_cfg, feature_layer=feature_layer)

    if args.device == "auto":
        device = default_device()
    else:
        import torch

        device = torch.device(args.device)

    model_name = model_slug(model_cfg)
    out_dir = stimulus_feature_dir(features_root, monkey, model_name, feature_layer)
    out_dir.mkdir(parents=True, exist_ok=True)

    path_fn = lambda session, condition: stimulus_map_path(
        features_root, monkey, model_name, feature_layer, session, condition
    )
    feat_df, n_written, n_skipped = extract_stimulus_features(
        model,
        manifest,
        repo_root=repo,
        map_path_fn=path_fn,
        feature_layer=feature_layer,
        model_cfg=model_cfg,
        input_size=int(model_cfg.get("input_size", 224)),
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
        "model_name": backbone_name,
        "model_slug": model_name,
        "model_type": backbone_type,
        "pretrained": bool(model_cfg.get("pretrained", True)),
        "preprocess": model_cfg.get("preprocess", "imagenet_rgb"),
        "feature_layer": feature_layer,
        "feature_shape": sample_shape,
        "input_size": int(model_cfg.get("input_size", 224)),
        "imagenet_normalize": bool(model_cfg.get("imagenet_normalize", True)),
        "batch_size": int(model_cfg.get("batch_size", 32)),
        "device": str(device),
        "n_stimuli_manifest": int(len(manifest)),
        "n_maps_written": int(n_written),
        "n_maps_skipped_existing": int(n_skipped),
        "stimulus_manifest": _portable_path(stim_manifest_path, repo),
        "feature_manifest": _portable_path(feat_manifest, repo),
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"Monkey: {monkey}")
    print(f"Backbone: {backbone_name} ({model_name})")
    print(f"Feature layer: {feature_layer}")
    if sample_shape:
        print(f"Map shape (C, H, W): {sample_shape}")
    print(f"Device: {device}")
    print(f"Stimuli in manifest: {len(manifest)}")
    print(f"Maps written: {n_written} (skipped existing: {n_skipped})")
    print(f"Feature manifest: {_portable_path(feat_manifest, repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
