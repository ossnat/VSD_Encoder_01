#!/usr/bin/env python3
"""Warp CNN stimulus feature maps onto the anchor VSD grid via the Schira LUT."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.DL_features.schema import model_slug, stimulus_map_path
from src.encoding.schema import encoding_pairs_manifest_path
from src.paths import project_root, resolve_data_path
from src.schira_encoding.io import resolve_output_root
from src.schira_encoding.lut import SchiraLUT
from src.schira_encoding.schema import lut_path, warped_feature_map_path
from src.schira_encoding.warp_features import warp_feature_map


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    repo = project_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=repo / "configs/schira_encoding/default.yaml",
    )
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument(
        "--window-config",
        type=Path,
        default=None,
        help="Default: window_config from --config (raw evoked_35_46).",
    )
    p.add_argument("--feature-layer", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Max unique stimulus keys")
    args = p.parse_args()

    cfg = _load_yaml(args.config)
    win_path = args.window_config
    if win_path is None:
        win_path = repo / str(
            cfg.get("window_config", "configs/windows/evoked_35_46.yaml")
        )
    win = _load_yaml(win_path)
    model_cfg = _load_yaml(args.model_config)
    cfg = {**cfg, **{k: v for k, v in win.items() if k != "paths"}}
    if "paths" in win:
        cfg.setdefault("paths", {}).update(win["paths"])

    monkey = str(cfg["monkey"])
    schira_set = str(cfg["schira_set"])
    anchor = str(cfg["anchor_session"])
    spatial = tuple(int(x) for x in cfg["spatial_size"])
    input_size = int(cfg.get("input_size", model_cfg.get("input_size", 224)))
    layer = str(args.feature_layer or model_cfg.get("feature_layer", "layer3"))
    slug = model_slug(model_cfg)

    stimuli_cfg = _load_yaml(
        repo / str(cfg.get("stimuli_config", "configs/stimuli/default.yaml"))
    )
    canvas = int(stimuli_cfg.get("canvas_size", 210))

    out_root = resolve_output_root(cfg["paths"]["schira_encoding_root"], repo)
    lut_file = lut_path(
        out_root,
        monkey,
        schira_set=schira_set,
        anchor_session=anchor,
        spatial_h=spatial[0],
        spatial_w=spatial[1],
        canvas_size=canvas,
        input_size=input_size,
    )
    if not lut_file.exists():
        raise FileNotFoundError(
            f"Missing LUT {lut_file}. Run scripts/10_build_schira_lut.py first."
        )
    lut = SchiraLUT.load(lut_file)

    features_root = resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo)
    pairs_root = resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo)
    start = int(cfg["start_frame"])
    end = int(cfg["end_frame"])
    window_id = cfg.get("window_id") or f"win_{start:04d}_{end:04d}"
    manifest = encoding_pairs_manifest_path(pairs_root, monkey, window_id)
    pairs = pd.read_parquet(manifest)

    keys = (
        pairs[["date", "condition"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    keys_list = list(keys)
    if args.limit is not None:
        keys_list = keys_list[: int(args.limit)]

    n_wrote = 0
    n_skip = 0
    for h5_session, condition in keys_list:
        h5_session = str(h5_session)
        condition = str(condition)
        src = stimulus_map_path(
            features_root, monkey, slug, layer, h5_session, condition
        )
        dst = warped_feature_map_path(
            out_root,
            monkey,
            model_slug=slug,
            feature_layer=layer,
            schira_set=schira_set,
            anchor_session=anchor,
            h5_session=h5_session,
            condition=condition,
        )
        if dst.exists() and not args.overwrite:
            n_skip += 1
            continue
        if not src.exists():
            raise FileNotFoundError(f"Missing feature map: {src}")
        feat = np.load(src)
        warped = warp_feature_map(feat, lut, fill=0.0)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, warped.astype(np.float32))
        n_wrote += 1

    print(
        f"Warped features: wrote={n_wrote} skipped={n_skip} "
        f"layer={layer} slug={slug} lut_valid={int(lut.valid.sum())}"
    )


if __name__ == "__main__":
    main()
