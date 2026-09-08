#!/usr/bin/env python3
"""Fig. 13 caption positions: thesis VSD+model vs our Schira (u,v).

No pixel alignment — thesis crop (left) vs Schira ink rotated with the
session chamber affine (right) so overall letter pose matches the thesis VSD.

Caption (Fig. 13, page 31):
  20/11/18 — D, F, N at (1, −0.9); L at (1.4, −0.7)
  10/07/18 — vertical and horizontal bars at (0.6, −0.75)

Usage:
  scripts/py experiments/schira2007/make_fig13_caption_schira_compare.py
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from src.paths import project_root, resolve_data_path
from src.retinotopy.affine import CorticalAffine
from src.retinotopy.params import load_schira_set
from src.retinotopy.warp import forward_ink_cloud_w
from src.stimuli.catalog import (
    load_full_encoder_catalog,
    stimulus_spec_from_mapping,
)
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.render import RenderConfig, render_stimulus

HERE = Path(__file__).resolve().parent
DEFAULT_SCHIRA = project_root() / "configs/schira/sets.yaml"
DEFAULT_STIMULI = project_root() / "configs/stimuli/default.yaml"
THESIS_PAGE = HERE / "_throwaway_thesis_fig10_pdf_pages" / "page_031.png"
OUT_DIR = HERE / "phase_0_register"

# Per-panel crop on page_031 (1530×1980): center of VSD circle in page frac coords.
FIG13_ROWS: list[dict] = [
    {
        "stimulus_id": "letter_D_white_1",
        "panel": "c",
        "label": "D",
        "crop_x_frac": 0.452,
        "crop_y_frac": 0.308,
        "pos_x_deg": 1.0,
        "pos_y_deg": -0.9,
        "schira_set": "201118",
        "session_note": "20/11/18",
    },
    {
        "stimulus_id": "letter_F_white_1",
        "panel": "d",
        "label": "F",
        "crop_x_frac": 0.495,
        "crop_y_frac": 0.381,
        "pos_x_deg": 1.0,
        "pos_y_deg": -0.9,
        "schira_set": "201118",
        "session_note": "20/11/18",
    },
    {
        "stimulus_id": "letter_L_white_1",
        "panel": "e",
        "label": "L",
        "crop_x_frac": 0.463,
        "crop_y_frac": 0.437,
        "pos_x_deg": 1.4,
        "pos_y_deg": -0.7,
        "schira_set": "201118",
        "session_note": "20/11/18",
    },
    {
        "stimulus_id": "letter_N_white_1",
        "panel": "f",
        "label": "N",
        "crop_x_frac": 0.518,
        "crop_y_frac": 0.508,
        "pos_x_deg": 1.0,
        "pos_y_deg": -0.9,
        "schira_set": "201118",
        "session_note": "20/11/18",
    },
    {
        "stimulus_id": "black_bar_vertical_1",
        "panel": "g",
        "label": "bar V",
        "crop_x_frac": 0.463,
        "crop_y_frac": 0.551,
        "pos_x_deg": 0.6,
        "pos_y_deg": -0.75,
        "schira_set": "100718",
        "session_note": "10/07/18",
    },
    {
        "stimulus_id": "black_bar_horizontal_1",
        "panel": "h",
        "label": "bar H",
        "crop_x_frac": 0.820,
        "crop_y_frac": 0.551,
        "pos_x_deg": 0.6,
        "pos_y_deg": -0.75,
        "schira_set": "100718",
        "session_note": "10/07/18",
    },
]


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _render_config(stimuli_cfg: dict, *, canvas_size: int = 630) -> RenderConfig:
    base_canvas = int(stimuli_cfg.get("canvas_size", 224))
    quadrant_extent_deg = float(stimuli_cfg.get("quadrant_extent_deg", 6.0))
    base_ppd = stimuli_cfg.get("pixels_per_deg")
    if base_ppd is None:
        base_ppd = base_canvas / quadrant_extent_deg
    pixels_per_deg = float(base_ppd) * (canvas_size / base_canvas)
    return RenderConfig(
        canvas_size=canvas_size,
        pixels_per_deg=pixels_per_deg,
        quadrant_extent_deg=quadrant_extent_deg,
        background_gray=int(stimuli_cfg.get("background_gray", 128)),
        bar_length_deg=float(stimuli_cfg.get("bar_length_deg", 1.0)),
        bar_width_px=int(stimuli_cfg.get("bar_width_px", 1)),
        contour_width_px=int(stimuli_cfg.get("contour_width_px", 1)),
        assume_size_is_diameter=bool(stimuli_cfg.get("assume_size_is_diameter", True)),
        draw_fixation=bool(stimuli_cfg.get("draw_fixation", False)),
    )


def _crop_thesis_vsd_panel(
    page: Image.Image,
    *,
    x_frac: float,
    y_frac: float,
    half_px: int = 105,
) -> np.ndarray:
    w, h = page.size
    cx = int(float(x_frac) * w)
    cy = int(float(y_frac) * h)
    half = int(half_px)
    box = (cx - half, cy - half, cx + half, cy + half)
    arr = np.asarray(page.crop(box))
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float64) / 255.0
    return arr


def _catalog_row(catalog, stimulus_id: str) -> dict:
    hit = catalog[catalog["stimulus_id"].astype(str) == stimulus_id]
    if hit.empty:
        raise RuntimeError(f"No catalog row for {stimulus_id!r}")
    return hit.iloc[0].to_dict()


def _view_uv_for_chamber(
    u: np.ndarray,
    v: np.ndarray,
    affine: CorticalAffine,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate/flip (u,v) like the fitted chamber affine (no scale or translation)."""
    u = np.asarray(u, dtype=np.float64).ravel().copy()
    v = np.asarray(v, dtype=np.float64).ravel().copy()
    if affine.flip_u:
        u = -u
    if affine.flip_v:
        v = -v
    th = np.deg2rad(float(affine.rotation_deg))
    c, s = np.cos(th), np.sin(th)
    ur = c * u - s * v
    vr = s * u + c * v
    return ur, vr


def _plot_schira_ax(
    ax,
    fu: np.ndarray,
    fv: np.ndarray,
    *,
    title: str,
    invert_y: bool = True,
) -> None:
    fu = np.asarray(fu, dtype=np.float64).ravel()
    fv = np.asarray(fv, dtype=np.float64).ravel()
    ax.scatter(fu, fv, s=3, c="black", marker="s", linewidths=0)
    pad = 0.12 * max(float(fu.max() - fu.min()), float(fv.max() - fv.min()), 0.1)
    ax.set_xlim(float(fu.min()) - pad, float(fu.max()) + pad)
    ax.set_ylim(float(fv.min()) - pad, float(fv.max()) + pad)
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("u (chamber view)")
    ax.set_ylabel("v (chamber view)")
    ax.set_title(title, fontsize=9)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schira-config", type=Path, default=DEFAULT_SCHIRA)
    p.add_argument("--thesis-page", type=Path, default=THESIS_PAGE)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--canvas-size", type=int, default=630)
    args = p.parse_args(argv)

    repo = project_root()
    stimuli_cfg = _load_yaml(DEFAULT_STIMULI)
    render_cfg = _render_config(stimuli_cfg, canvas_size=args.canvas_size)
    cfg = _load_yaml(project_root() / "configs/default.yaml")
    encoder_root = resolve_data_path(cfg["paths"]["encoder_data_root"], repo)
    catalog = attach_stimulus_ids(
        load_full_encoder_catalog(
            encoder_root,
            monkey=str(cfg["monkey"]),
            bar_length_deg=render_cfg.bar_length_deg,
        )
    )

    thesis_path = args.thesis_page if args.thesis_page.is_absolute() else repo / args.thesis_page
    if not thesis_path.exists():
        raise FileNotFoundError(f"Missing thesis page: {thesis_path}")
    thesis_page = Image.open(thesis_path)

    schira_sets: dict[str, tuple] = {}
    for row in FIG13_ROWS:
        name = row["schira_set"]
        if name not in schira_sets:
            schira_sets[name] = load_schira_set(args.schira_config, set_name=name)

    # Chamber pose for thesis VSD maps (201118 landmark fit).
    _, _, chamber_affine, _ = schira_sets["201118"]

    out_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(FIG13_ROWS)
    fig, axes = plt.subplots(n, 2, figsize=(7.4, 2.55 * n), layout="constrained")
    meta_rows: list[dict] = []

    for i, row in enumerate(FIG13_ROWS):
        stim_id = row["stimulus_id"]
        set_name = row["schira_set"]
        _, params, _, _ = schira_sets[set_name]

        base_spec = stimulus_spec_from_mapping(_catalog_row(catalog, stim_id))
        spec = replace(
            base_spec,
            pos_x_deg=float(row["pos_x_deg"]),
            pos_y_deg=float(row["pos_y_deg"]),
        )
        stimulus_rgb = render_stimulus(spec, render_cfg)
        fu, fv, _ = forward_ink_cloud_w(stimulus_rgb, params=params, render_cfg=render_cfg)
        fu_v, fv_v = _view_uv_for_chamber(fu, fv, chamber_affine)

        ax_thesis, ax_ours = axes[i, 0], axes[i, 1]
        thesis_vsd = _crop_thesis_vsd_panel(
            thesis_page,
            x_frac=row["crop_x_frac"],
            y_frac=row["crop_y_frac"],
        )
        ax_thesis.imshow(thesis_vsd)
        ax_thesis.set_title(
            f"Thesis {row['panel']} · VSD + model overlay\n"
            f"{row['session_note']} · caption pos "
            f"({row['pos_x_deg']:g}, {row['pos_y_deg']:g})",
            fontsize=9,
        )
        ax_thesis.axis("off")

        _plot_schira_ax(
            ax_ours,
            fu_v,
            fv_v,
            title=(
                f"Ours · Schira · set {set_name}\n"
                f"pos=({row['pos_x_deg']:g}, {row['pos_y_deg']:g}) · "
                f"view rot={chamber_affine.rotation_deg:.0f}° "
                f"flip_u={chamber_affine.flip_u}"
            ),
        )
        axes[i, 0].set_ylabel(
            row["label"],
            fontsize=11,
            rotation=0,
            labelpad=32,
            va="center",
        )

        meta_rows.append(
            {
                "stimulus_id": stim_id,
                "thesis_panel": row["panel"],
                "schira_set": set_name,
                "session_note": row["session_note"],
                "pos_x_deg": row["pos_x_deg"],
                "pos_y_deg": row["pos_y_deg"],
                "crop_x_frac": row["crop_x_frac"],
                "crop_y_frac": row["crop_y_frac"],
                "n_schira_ink": int(fu.size),
                "chamber_view_rot_deg": float(chamber_affine.rotation_deg),
                "chamber_view_flip_u": bool(chamber_affine.flip_u),
            }
        )

    fig.suptitle(
        "Fig. 13 caption · thesis VSD+model (left) vs Schira ink in chamber pose (right)\n"
        "Same caption positions; right panel = rotate/flip only (201118 chamber affine)",
        fontsize=11,
    )
    fig_path = out_dir / "fig13_caption_schira_vs_thesis_vsd.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "figure": str(fig_path.relative_to(repo)),
        "thesis_page": str(thesis_path.relative_to(repo)),
        "caption_positions": FIG13_ROWS,
        "chamber_view_affine": {
            "rotation_deg": chamber_affine.rotation_deg,
            "flip_u": chamber_affine.flip_u,
            "flip_v": chamber_affine.flip_v,
            "source_set": "201118",
        },
        "rows": meta_rows,
    }
    meta_path = out_dir / "fig13_caption_schira_vs_thesis_vsd.yaml"
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    print(f"Wrote {fig_path}")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
