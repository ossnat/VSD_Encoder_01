#!/usr/bin/env python3
"""Step 1: Schira-only maps in model cortical ``(u, v)`` (no camera affine).

Each stimulus PNG has three panels:
  1. Visual-field render
  2. Forward ink scatter (stimulus pixels → Schira ``w``)
  3. Inverse warp on a ``(u, v)`` grid (paper / thesis style cortical image)

Camera affine / VSD overlays are step 2 (``run_phase_0_register.py`` / ``run_phase_0.py``).

Usage:
  scripts/py experiments/schira2007/run_schira_only.py --set 201118
  scripts/py experiments/schira2007/run_schira_only.py --set 201118 \\
      --stimuli letter_D_white_1 letter_F_white_1 --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.paths import project_root, resolve_data_path
from src.retinotopy.params import load_schira_set
from src.retinotopy.plotting import (
    plot_schira_cortex_comparison,
    plot_schira_only_montage,
)
from src.retinotopy.schira import SchiraParams
from src.retinotopy.visual_field import upsample_render_canvas
from src.retinotopy.warp import forward_ink_cloud_w, sample_stimulus_onto_cortical_w
from src.stimuli.catalog import (
    load_full_encoder_catalog,
    stimulus_spec_from_mapping,
)
from src.stimuli.exclusions import is_excluded_encoding_trial
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.render import RenderConfig, render_stimulus

HERE = Path(__file__).resolve().parent
DEFAULT_SCHIRA_CONFIG = project_root() / "configs/schira/sets.yaml"
DEFAULT_CONFIG = project_root() / "configs/default.yaml"
DEFAULT_STIMULI = project_root() / "configs/stimuli/default.yaml"
OUT_DIR = HERE / "phase_0_schira"


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _render_config(stimuli_cfg: dict, *, canvas_size: int | None = None) -> RenderConfig:
    base_canvas = int(stimuli_cfg.get("canvas_size", 210))
    canvas = int(canvas_size) if canvas_size is not None else base_canvas
    quadrant_extent_deg = float(stimuli_cfg.get("quadrant_extent_deg", 6.0))
    # Keep degrees/pixel consistent with the base catalog canvas.
    base_ppd = stimuli_cfg.get("pixels_per_deg")
    if base_ppd is None:
        base_ppd = base_canvas / quadrant_extent_deg
    pixels_per_deg = float(base_ppd) * (canvas / base_canvas)
    return RenderConfig(
        canvas_size=canvas,
        pixels_per_deg=pixels_per_deg,
        quadrant_extent_deg=quadrant_extent_deg,
        background_gray=int(stimuli_cfg.get("background_gray", 128)),
        bar_length_deg=float(stimuli_cfg.get("bar_length_deg", 1.0)),
        bar_width_px=max(
            1, int(round(int(stimuli_cfg.get("bar_width_px", 1)) * canvas / base_canvas))
        ),
        contour_width_px=max(
            1,
            int(round(int(stimuli_cfg.get("contour_width_px", 1)) * canvas / base_canvas)),
        ),
        assume_size_is_diameter=bool(stimuli_cfg.get("assume_size_is_diameter", True)),
        draw_fixation=bool(stimuli_cfg.get("draw_fixation", False)),
        letter_box_deg=float(stimuli_cfg.get("letter_box_deg", 1.0)),
    )


def _unique_stimuli_for_prefix(catalog, date_prefix: str):
    """One row per ``stimulus_id`` under the session date prefix."""
    sub = catalog[catalog["h5_session"].astype(str).str.startswith(date_prefix)].copy()
    sub = sub[~sub["is_blank"]].copy()
    sub = attach_stimulus_ids(sub)
    exclude = sub.apply(
        lambda r: is_excluded_encoding_trial(
            str(r["h5_session"]), str(r["condition"]), shape_type=str(r.get("shape_type"))
        ),
        axis=1,
    )
    sub = sub.loc[~exclude].reset_index(drop=True)
    sub = sub.sort_values(["stimulus_id", "h5_session", "condition"]).reset_index(
        drop=True
    )
    return sub.groupby("stimulus_id", sort=True).first().reset_index()


def run_schira_only(
    *,
    repo: Path | None = None,
    schira_config: Path = DEFAULT_SCHIRA_CONFIG,
    set_name: str | None = None,
    config_path: Path = DEFAULT_CONFIG,
    stimuli_yaml: Path = DEFAULT_STIMULI,
    output_dir: Path | None = None,
    canvas_size: int | None = None,
    overwrite: bool = False,
    skip_existing: bool = True,
    sech_amp: float | None = None,
    stimulus_ids: list[str] | None = None,
    inverse_grid_size: int = 200,
) -> dict:
    """Schira-only cortex views: VF render, forward scatter, inverse warp."""
    repo = repo or project_root()
    cfg = _load_yaml(config_path)
    stimuli_cfg = _load_yaml(stimuli_yaml)
    # 3× YAML canvas (210→630) so 1° = 105 px = 35×3. Degrees unchanged.
    base_canvas = int(stimuli_cfg.get("canvas_size", 210))
    canvas = (
        int(canvas_size)
        if canvas_size is not None
        else upsample_render_canvas(base_canvas)
    )
    render_cfg = _render_config(stimuli_cfg, canvas_size=canvas)
    set_id, params, _affine, session_raw = load_schira_set(
        schira_config, set_name=set_name
    )
    if sech_amp is not None:
        params = SchiraParams(
            a=params.a,
            alpha=params.alpha,
            k=params.k,
            shear=params.shear,
            sech_ecc_k=params.sech_ecc_k,
            sech_amp=float(sech_amp),
            fa_combine=params.fa_combine,
        )

    date_prefix = session_raw.get("date_prefix")
    if not date_prefix:
        raise ValueError(f"Schira set {set_id!r} must define date_prefix")
    date_prefix = str(date_prefix)

    encoder_root = resolve_data_path(cfg["paths"]["encoder_data_root"], repo)
    catalog = load_full_encoder_catalog(
        encoder_root,
        monkey=str(cfg["monkey"]),
        bar_length_deg=render_cfg.bar_length_deg,
    )
    rows = _unique_stimuli_for_prefix(catalog, date_prefix)
    if stimulus_ids:
        wanted = {str(s) for s in stimulus_ids}
        rows = rows[rows["stimulus_id"].astype(str).isin(wanted)].copy()
        # Allow geometry checks for stimuli shown on other days (e.g. bars
        # under 201118 Schira params): fall back to full-catalog uniques.
        missing = wanted - set(rows["stimulus_id"].astype(str))
        if missing:
            full = attach_stimulus_ids(catalog.copy())
            full = full[~full["is_blank"]].copy()
            extra = (
                full[full["stimulus_id"].astype(str).isin(missing)]
                .sort_values(["stimulus_id", "h5_session"])
                .groupby("stimulus_id", sort=True)
                .first()
                .reset_index()
            )
            if not extra.empty:
                print(
                    f"Note: {sorted(extra['stimulus_id'].astype(str))} not on "
                    f"{date_prefix}; using catalog geometry with this set's Schira params."
                )
                rows = (
                    extra
                    if rows.empty
                    else pd.concat([rows, extra], ignore_index=True)
                )
    if rows.empty:
        raise RuntimeError(
            f"No stimuli for date prefix {date_prefix!r}"
            + (f" matching {stimulus_ids!r}" if stimulus_ids else "")
        )

    out_dir = Path(output_dir) if output_dir is not None else OUT_DIR
    out_dir = out_dir if out_dir.is_absolute() else (repo / out_dir)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(
        f"Schira ink-only (no affine/VSD): set={set_id} a={params.a} "
        f"alpha={params.alpha} k={params.k} shear={params.shear} "
        f"sech_amp={params.sech_amp} sech_ecc_k={params.sech_ecc_k} "
        f"canvas={render_cfg.canvas_size} pixels_per_deg={render_cfg.pixels_per_deg:g}"
    )

    param_note = (
        f"set={set_id} · ink pixels only · Schira ONLY (no camera registration) · "
        f"a={params.a:g} α={params.alpha:g} k={params.k:g} "
        f"shear={params.shear} sech_ecc_k={params.sech_ecc_k:g} "
        f"sech_amp={params.sech_amp:g}"
    )
    index_rows: list[dict] = []
    montage_panels: list[tuple[np.ndarray, np.ndarray, str]] = []

    for _, row in rows.iterrows():
        row_dict = row.to_dict()
        spec = stimulus_spec_from_mapping(row_dict)
        stim_id = str(row_dict["stimulus_id"])
        fig_name = f"{date_prefix}__{stim_id}.png"
        fig_path = out_dir / fig_name
        if skip_existing and fig_path.exists() and not overwrite:
            index_rows.append(
                {
                    "stimulus_id": stim_id,
                    "figure": str(fig_path.relative_to(repo)),
                    "skipped": True,
                }
            )
            continue

        # Raw render: map only non-gray ink (no fill / no gray background warp).
        stimulus_rgb = render_stimulus(spec, render_cfg)
        fu, fv, _w = forward_ink_cloud_w(
            stimulus_rgb, params=params, render_cfg=render_cfg
        )
        if fu.size == 0:
            raise RuntimeError(f"No ink pixels found for {stim_id!r}")

        warped_rgb, _valid, u_range, v_range = sample_stimulus_onto_cortical_w(
            stimulus_rgb,
            params=params,
            render_cfg=render_cfg,
            grid_size=int(inverse_grid_size),
        )

        title = f"{stim_id} · Schira cortex views (no camera affine)"
        plot_schira_cortex_comparison(
            stimulus_rgb,
            fu,
            fv,
            warped_rgb,
            u_range,
            v_range,
            output_path=fig_path,
            title=title,
            param_note=param_note,
        )
        montage_panels.append((fu, fv, stim_id))
        index_rows.append(
            {
                "stimulus_id": stim_id,
                "stimulus_text": spec.stimulus_text,
                "pos_x_deg": float(spec.pos_x_deg),
                "pos_y_deg": float(spec.pos_y_deg),
                "n_ink_pixels": int(fu.size),
                "u_range": [float(fu.min()), float(fu.max())],
                "v_range": [float(fv.min()), float(fv.max())],
                "figure": str(fig_path.relative_to(repo)),
                "skipped": False,
            }
        )
        print(
            f"{stim_id}: ink={fu.size} "
            f"u∈[{fu.min():.3g},{fu.max():.3g}] "
            f"v∈[{fv.min():.3g},{fv.max():.3g}]"
        )

    montage_path = out_dir / "all_stimuli_schira_only_montage.png"
    if montage_panels:
        plot_schira_only_montage(
            montage_panels,
            montage_path,
            title=(
                f"Schira ink-only → Cartesian (u, v) · {date_prefix} · "
                f"a={params.a:g}, α={params.alpha:g}, k={params.k:g}, "
                f"sech_amp={params.sech_amp:g}"
            ),
        )

    index = {
        "set_name": set_id,
        "mode": "schira_ink_only",
        "schira_config": str(schira_config),
        "date_prefix": date_prefix,
        "csv_date": session_raw.get("csv_date"),
        "canvas_size": int(render_cfg.canvas_size),
        "pixels_per_deg": float(render_cfg.pixels_per_deg),
        "yaml_base_canvas": int(stimuli_cfg.get("canvas_size", 210)),
        "yaml_pixels_per_deg": float(stimuli_cfg.get("pixels_per_deg", 35.0)),
        "output_dir": str(out_dir.relative_to(repo)),
        "schira": {
            "a": params.a,
            "alpha": params.alpha,
            "k": params.k,
            "shear": params.shear,
            "sech_ecc_k": params.sech_ecc_k,
            "sech_amp": params.sech_amp,
        },
        "affine": None,
        "note": (
            "Per-stimulus PNG: VF render | forward ink scatter | inverse warp on "
            "(u,v) grid (Ayzenshtat / thesis cortical map style). No camera affine."
        ),
        "stimuli": index_rows,
        "montage": (
            str(montage_path.relative_to(repo)) if montage_panels else None
        ),
    }
    index_path = out_dir / "index.yaml"
    index_path.write_text(yaml.safe_dump(index, sort_keys=False))
    print(f"Wrote {index_path}")
    return index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--schira-config",
        type=Path,
        default=DEFAULT_SCHIRA_CONFIG,
        help="YAML registry or standalone session file",
    )
    p.add_argument("--set", dest="set_name", default=None, help="Named set key")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--stimuli-yaml", type=Path, default=DEFAULT_STIMULI)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--canvas-size",
        type=int,
        default=None,
        help="Render resolution (default 3× YAML canvas: 210→630, 1°=105 px)",
    )
    p.add_argument(
        "--stimuli",
        nargs="+",
        default=None,
        help="Optional stimulus_id filter, e.g. letter_A_white_1 letter_D_white_1",
    )
    p.add_argument(
        "--sech-amp",
        type=float,
        default=None,
        help="Override YAML sech_amp for this run only",
    )
    p.add_argument(
        "--inverse-grid-size",
        type=int,
        default=200,
        help="Resolution of inverse-warp (u,v) panel (default 200)",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute even if the PNG already exists",
    )
    args = p.parse_args()
    run_schira_only(
        schira_config=args.schira_config,
        set_name=args.set_name,
        config_path=args.config,
        stimuli_yaml=args.stimuli_yaml,
        output_dir=args.output_dir,
        canvas_size=args.canvas_size,
        overwrite=bool(args.overwrite or args.no_skip_existing),
        skip_existing=not bool(args.overwrite or args.no_skip_existing),
        sech_amp=args.sech_amp,
        stimulus_ids=args.stimuli,
        inverse_grid_size=args.inverse_grid_size,
    )


if __name__ == "__main__":
    main()
