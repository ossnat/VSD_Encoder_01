#!/usr/bin/env python3
"""Build rendered stimulus images from EncoderData CSV catalogs.

Stimulus library layout (Data/VSD_Encoder_01/stimuli/<monkey>/):
  library/<stimulus_uid>.png   one image per distinct stimulus (shape, size, colour/RGB, position, background)
  library.parquet              one row per stimulus_uid
  lookup.parquet               (h5_session, condition) -> stimulus_uid; several conditions can share a stimulus
  manifest.parquet             one row per (h5_session, condition), as before, with stimulus_uid;
                               image_path points into library/
Conditions that resolve to the same uid must render to identical pixels (checked here).
"""

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
)
from src.stimuli.contrast_letters_catalog import (
    contrast_letters_catalog_path,
    load_contrast_letters_catalog,
)
from src.stimuli.plotting import plot_all_stimuli
from src.stimuli.render import RenderConfig, render_stimulus
from src.stimuli.identity import stimulus_uid_from_row
from src.stimuli.schema import (
    library_dir,
    library_image_path,
    library_table_path,
    lookup_path,
    manifest_path,
    parsed_catalog_path,
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
    quadrant_extent_deg = float(cfg.get("quadrant_extent_deg", 6.0))
    pixels_per_deg = cfg.get("pixels_per_deg")
    if pixels_per_deg is None:
        pixels_per_deg = canvas_size / quadrant_extent_deg
    return RenderConfig(
        canvas_size=canvas_size,
        pixels_per_deg=float(pixels_per_deg),
        quadrant_extent_deg=quadrant_extent_deg,
        background_gray=int(cfg.get("background_gray", 128)),
        bar_length_deg=float(cfg.get("bar_length_deg", 1.0)),
        bar_width_px=int(cfg.get("bar_width_px", 1)),
        contour_width_px=int(cfg.get("contour_width_px", 1)),
        assume_size_is_diameter=bool(cfg.get("assume_size_is_diameter", True)),
        draw_fixation=bool(cfg.get("draw_fixation", False)),
        letter_box_deg=float(cfg.get("letter_box_deg", 1.0)),
    )


def _coerce_spec(row: dict) -> StimulusSpec:
    data = dict(row)
    for key in ("rgb", "letter", "source_path", "background_gray", "cortex_file", "size_deg", "size_note"):
        val = data.get(key, None)
        if val is None:
            data[key] = None
        elif isinstance(val, float) and np.isnan(val):
            data[key] = None
        elif isinstance(val, str) and val.strip().lower() in {"", "nan", "none"}:
            data[key] = None
    rgb = data.get("rgb")
    if isinstance(rgb, (list, tuple, np.ndarray)):
        data["rgb"] = tuple(int(x) for x in rgb)
    if data.get("background_gray") is not None:
        data["background_gray"] = int(data["background_gray"])
    return StimulusSpec(**data)


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
    catalog_paths = [_portable_path(catalog_path, repo)]

    letters_root = encoder_data_root / "letters_stimuli"
    contrast_path = contrast_letters_catalog_path(encoder_data_root)
    if contrast_path is not None:
        contrast_df = load_contrast_letters_catalog(
            contrast_path,
            monkey=monkey,
            letters_root=letters_root,
        )
        parsed_df = pd.concat([parsed_df, contrast_df], ignore_index=True)
        catalog_paths.append(_portable_path(contrast_path, repo))

    parsed_out = parsed_catalog_path(stimuli_root, monkey)
    parsed_out.parent.mkdir(parents=True, exist_ok=True)
    parsed_df.to_parquet(parsed_out, index=False)

    manifest_rows: list[dict] = []
    plot_entries: list[tuple[dict, object]] = []

    specs = [_coerce_spec(row) for row in parsed_df.to_dict(orient="records")]
    rendered: dict[str, np.ndarray] = {}
    for spec in specs:
        uid = stimulus_uid_from_row(spec.__dict__, default_background=render_cfg.background_gray)
        out_path = library_image_path(stimuli_root, spec.monkey, uid)
        meta = {
            "stimulus_uid": uid,
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
            "rgb": spec.rgb,
            "letter": spec.letter,
            "source_path": spec.source_path,
            "background_gray": spec.background_gray,
            "size_note": spec.size_note,
            "image_path": _portable_path(out_path, repo),
            "catalog_path": (
                _portable_path(contrast_path, repo)
                if contrast_path is not None
                and (spec.rgb is not None or spec.shape_type == "letter")
                else _portable_path(catalog_path, repo)
            ),
        }

        should_render = (
            only_shape_types is None or spec.shape_type in only_shape_types
        )
        if uid in rendered:
            image = rendered[uid]
            # Same uid must mean the same picture.
            check = render_stimulus(spec, render_cfg)
            if not np.array_equal(check, image):
                raise RuntimeError(f"{spec.h5_session} {spec.condition}: same uid {uid} but different pixels")
        elif out_path.exists() and not (should_render and overwrite):
            image = np.asarray(Image.open(out_path).convert("RGB"))
        else:
            image = render_stimulus(spec, render_cfg)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(out_path)
        rendered[uid] = image

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

    # Library table (one row per stimulus) and lookup table (session/condition -> stimulus).
    lookup_df = manifest_df[["h5_session", "condition", "condition_num", "stimulus_uid"]].copy()
    lookup_df.to_parquet(lookup_path(stimuli_root, monkey), index=False)
    lib_cols = ["stimulus_uid", "shape_type", "color", "rgb", "size_deg", "pos_x_deg", "pos_y_deg",
                "background_gray", "letter", "source_path", "is_blank", "size_note", "image_path"]
    first = manifest_df.drop_duplicates("stimulus_uid")[lib_cols].copy()
    uses = manifest_df.groupby("stimulus_uid").apply(
        lambda g: ", ".join(f"{s}:{c}" for s, c in zip(g.h5_session, g.condition)), include_groups=False
    ).rename("used_in")
    texts = manifest_df.groupby("stimulus_uid").stimulus_text.apply(lambda t: " | ".join(sorted(set(t)))).rename("catalog_texts")
    library_df = first.merge(uses, on="stimulus_uid").merge(texts, on="stimulus_uid")
    library_df["n_uses"] = library_df.used_in.str.count(",") + 1
    library_df.sort_values("stimulus_uid").reset_index(drop=True).to_parquet(library_table_path(stimuli_root, monkey), index=False)
    stale = sorted(p.name for p in library_dir(stimuli_root, monkey).glob("*.png")
                   if p.stem not in set(library_df.stimulus_uid))
    if stale:
        print(f"WARNING: {len(stale)} library images are not used by any condition: {stale[:5]}...")

    plot_dir = plots_root / monkey
    plot_paths = plot_all_stimuli(plot_entries, plot_dir)

    run_cfg = {
        "monkey": monkey,
        "catalog_paths": catalog_paths,
        "manifest_path": _portable_path(out_manifest, repo),
        "n_stimuli": int(len(manifest_df)),
        "n_library_images": int(manifest_df.stimulus_uid.nunique()),
        "library_path": _portable_path(library_dir(stimuli_root, monkey), repo),
        "lookup_path": _portable_path(lookup_path(stimuli_root, monkey), repo),
        "render_config": render_cfg.__dict__,
        "created": datetime.now(timezone.utc).isoformat(),
        "plot_dir": str(plot_dir.relative_to(repo)),
    }
    cfg_path = stimuli_root / monkey / "config.json"
    with cfg_path.open("w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"Monkey: {monkey}")
    print(f"Catalogs: {catalog_paths}")
    print(f"Stimuli rendered: {len(manifest_df)}")
    print(f"Manifest: {_portable_path(out_manifest, repo)}")
    print(f"Library: {_portable_path(library_dir(stimuli_root, monkey), repo)} "
          f"({manifest_df.stimulus_uid.nunique()} images for {len(manifest_df)} session-conditions)")
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
