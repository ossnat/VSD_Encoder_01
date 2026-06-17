#!/usr/bin/env python3
"""Build rendered stimulus images from EncoderData CSV catalogs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.paths import project_root, resolve_data_path, workspace_root
from src.stimuli.catalog import (
    StimulusSpec,
    load_stimulus_catalog,
    monkey_catalog_path,
    parse_stimulus_rows,
)
from src.stimuli.plotting import plot_all_stimuli
from src.stimuli.render import RenderConfig, render_stimulus
from src.stimuli.schema import (
    manifest_path,
    parsed_catalog_path,
    stimulus_image_path,
)


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _portable_path(abs_path: Path, repo: Path) -> str:
    ws = workspace_root(repo)
    resolved = abs_path.resolve()
    try:
        return str(resolved.relative_to(ws))
    except ValueError:
        return str(resolved)


def _render_config(cfg: dict) -> RenderConfig:
    canvas_size = int(cfg.get("canvas_size", 224))
    quadrant_extent_deg = float(cfg.get("quadrant_extent_deg", 4.0))
    pixels_per_deg = cfg.get("pixels_per_deg")
    if pixels_per_deg is None:
        pixels_per_deg = canvas_size / quadrant_extent_deg
    return RenderConfig(
        canvas_size=canvas_size,
        pixels_per_deg=float(pixels_per_deg),
        quadrant_extent_deg=quadrant_extent_deg,
        background_gray=int(cfg.get("background_gray", 128)),
        bar_length_deg=float(cfg.get("bar_length_deg", 0.3)),
        bar_width_px=int(cfg.get("bar_width_px", 1)),
        contour_width_px=int(cfg.get("contour_width_px", 1)),
        assume_size_is_diameter=bool(cfg.get("assume_size_is_diameter", True)),
        draw_fixation=bool(cfg.get("draw_fixation", False)),
    )


def build_stimulus_images(
    *,
    monkey: str,
    encoder_data_root: Path,
    stimuli_root: Path,
    plots_root: Path,
    render_cfg: RenderConfig,
    repo: Path,
    overwrite: bool = False,
    only_shape_types: set[str] | None = None,
) -> pd.DataFrame:
    catalog_path = monkey_catalog_path(encoder_data_root, monkey)
    parsed_df = load_stimulus_catalog(
        catalog_path,
        monkey=monkey,
        bar_length_deg=render_cfg.bar_length_deg,
    )
    parsed_out = parsed_catalog_path(stimuli_root, monkey)
    parsed_out.parent.mkdir(parents=True, exist_ok=True)
    parsed_df.to_parquet(parsed_out, index=False)

    manifest_rows: list[dict] = []
    plot_entries: list[tuple[dict, object]] = []

    specs = [
        StimulusSpec(**row) for row in parsed_df.to_dict(orient="records")
    ]
    for spec in specs:
        out_path = stimulus_image_path(
            stimuli_root, spec.monkey, spec.h5_session, spec.condition
        )
        meta = {
            "monkey": spec.monkey,
            "csv_date": spec.csv_date,
            "session_letter": spec.session_letter,
            "h5_session": spec.h5_session,
            "condition": spec.condition,
            "condition_num": spec.condition_num,
            "stimulus_text": spec.stimulus_text,
            "color": spec.color,
            "shape_type": spec.shape_type,
            "size_deg": spec.size_deg,
            "pos_x_deg": spec.pos_x_deg,
            "pos_y_deg": spec.pos_y_deg,
            "is_blank": spec.is_blank,
            "cortex_file": spec.cortex_file,
            "image_path": _portable_path(out_path, repo),
            "catalog_path": _portable_path(catalog_path, repo),
        }
        should_render = (
            only_shape_types is None or spec.shape_type in only_shape_types
        )
        if out_path.exists() and not (should_render and overwrite):
            image = np.asarray(Image.open(out_path))
        else:
            image = render_stimulus(spec, render_cfg)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(out_path)

        manifest_rows.append(meta)
        plot_entries.append((meta, image))

    manifest_df = pd.DataFrame(manifest_rows).sort_values(
        ["h5_session", "condition_num"]
    ).reset_index(drop=True)
    dup_mask = manifest_df.duplicated(["h5_session", "condition"], keep=False)
    if dup_mask.any():
        dups = manifest_df.loc[dup_mask, ["h5_session", "condition", "stimulus_text"]]
        print("WARNING: duplicate (h5_session, condition) rows in catalog:")
        print(dups.to_string(index=False))
    out_manifest = manifest_path(stimuli_root, monkey)
    manifest_df.to_parquet(out_manifest, index=False)

    plot_dir = plots_root / monkey
    plot_paths = plot_all_stimuli(plot_entries, plot_dir)

    run_cfg = {
        "monkey": monkey,
        "catalog_path": _portable_path(catalog_path, repo),
        "manifest_path": _portable_path(out_manifest, repo),
        "n_stimuli": int(len(manifest_df)),
        "render_config": render_cfg.__dict__,
        "created": datetime.now(timezone.utc).isoformat(),
        "plot_dir": str(plot_dir.relative_to(repo)),
    }
    cfg_path = stimuli_root / monkey / "config.json"
    with cfg_path.open("w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"Monkey: {monkey}")
    print(f"Catalog: {_portable_path(catalog_path, repo)}")
    print(f"Stimuli rendered: {len(manifest_df)}")
    print(f"Manifest: {_portable_path(out_manifest, repo)}")
    print(f"Images root: {_portable_path(stimuli_root / monkey / 'images', repo)}")
    print("QC plots:")
    for p in plot_paths:
        print(f"  {p.relative_to(repo)}")
    return manifest_df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/default.yaml",
    )
    parser.add_argument(
        "--stimuli-config",
        type=Path,
        default=project_root() / "configs/stimuli/default.yaml",
    )
    parser.add_argument("--monkey", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only-shape-types",
        type=str,
        default=None,
        help="Comma-separated shape types to re-render (e.g. bar_vertical,bar_horizontal)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    base_cfg = _load_yaml(args.config)
    stim_cfg = _load_yaml(args.stimuli_config)
    monkey = args.monkey or base_cfg["monkey"]

    encoder_data_root = resolve_data_path(base_cfg["paths"]["encoder_data_root"], repo)
    stimuli_root = resolve_data_path(base_cfg["paths"]["stimuli_root"], repo)
    plots_root = repo / base_cfg["paths"]["stimuli_plots_root"]

    only_shape_types = None
    if args.only_shape_types:
        only_shape_types = {s.strip() for s in args.only_shape_types.split(",")}

    render_cfg = _render_config(stim_cfg)
    build_stimulus_images(
        monkey=monkey,
        encoder_data_root=encoder_data_root,
        stimuli_root=stimuli_root,
        plots_root=plots_root,
        render_cfg=render_cfg,
        repo=repo,
        overwrite=args.overwrite,
        only_shape_types=only_shape_types,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
