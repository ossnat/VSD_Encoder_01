#!/usr/bin/env python3
"""Build the static Schira LUT on the anchor VSD camera grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.paths import project_root
from src.retinotopy.params import load_schira_set
from src.schira_encoding.io import resolve_output_root
from src.schira_encoding.lut import SchiraLUT, build_schira_lut
from src.schira_encoding.schema import lut_path
from src.stimuli.render import RenderConfig


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _render_config(stimuli_cfg: dict) -> RenderConfig:
    return RenderConfig(
        canvas_size=int(stimuli_cfg.get("canvas_size", 210)),
        pixels_per_deg=float(stimuli_cfg.get("pixels_per_deg", 35.0)),
        quadrant_extent_deg=float(stimuli_cfg.get("quadrant_extent_deg", 6.0)),
        background_gray=int(stimuli_cfg.get("background_gray", 128)),
    )


def main() -> None:
    repo = project_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=repo / "configs/schira_encoding/default.yaml",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = _load_yaml(args.config)
    monkey = str(cfg["monkey"])
    schira_set = str(cfg["schira_set"])
    anchor = str(cfg["anchor_session"])
    spatial = tuple(int(x) for x in cfg["spatial_size"])
    input_size = int(cfg.get("input_size", 224))
    max_ecc = cfg.get("max_ecc_deg")
    max_ecc_f = None if max_ecc is None else float(max_ecc)

    stimuli_cfg = _load_yaml(repo / str(cfg.get("stimuli_config", "configs/stimuli/default.yaml")))
    render_cfg = _render_config(stimuli_cfg)

    schira_path = repo / str(cfg.get("schira_config", "configs/schira/sets.yaml"))
    name, params, affine, _ = load_schira_set(schira_path, set_name=schira_set)

    out_root = resolve_output_root(cfg["paths"]["schira_encoding_root"], repo)
    out_path = lut_path(
        out_root,
        monkey,
        schira_set=name,
        anchor_session=anchor,
        spatial_h=spatial[0],
        spatial_w=spatial[1],
        canvas_size=int(render_cfg.canvas_size),
        input_size=input_size,
    )
    if out_path.exists() and not args.overwrite:
        print(f"Exists (skip): {out_path}")
        lut = SchiraLUT.load(out_path)
        print(f"valid pixels: {int(lut.valid.sum())} / {lut.valid.size}")
        return

    lut = build_schira_lut(
        params=params,
        affine=affine,
        schira_set=name,
        anchor_session=anchor,
        spatial_size=spatial,
        render_cfg=render_cfg,
        input_size=input_size,
        max_ecc_deg=max_ecc_f,
    )
    lut.save(out_path)
    print(f"Wrote {out_path}")
    print(
        f"set={name} anchor={anchor} spatial={spatial} "
        f"canvas={lut.canvas_size} input={lut.input_size} "
        f"valid={int(lut.valid.sum())}/{lut.valid.size}"
    )


if __name__ == "__main__":
    main()
