"""Overlay warped Schira ink on mean VSD for a few large shapes.

Compares YAML start vs the large-shapes Schira+camera fit (point-held-out fold,
which sees all 31 geometry examples).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.encoding.cm_plotting import load_stimulus_rgb
from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.loss_roi import NOISE_CEILING_HULL_MASK_RELPATH
from src.evaluation.mask import apply_mask_nan
from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import VSD_CMAP, register_mapgeog
from src.retinotopy.params import affine_from_mapping, load_schira_set, params_from_mapping
from src.retinotopy.plotting import _ink_rgba_overlay, plot_vsd_with_warped_stimulus
from src.retinotopy.warp import forward_ink_mask_on_vsd
from src.schira_encoding.geometry_fit import build_anchor_stimulus_means
from src.schira_encoding.io import resolve_output_root
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.render import RenderConfig

SAMPLES = (
    ("100718a", "black_triangle_contour_0.4"),
    ("100718a", "black_circle_contour_0.3"),
    ("130618d", "black_filled_circle_0.3"),
    ("230518b", "white_filled_circle_0.8"),
)
BEST_FIT = (
    Path("experiments/schira_encoding/gandalf/loo/win_0035_0046")
    / "vgg16_imagenet/block1_prepool/set-100718__anchor-100718a"
    / "global_channel_ridge/schira_opt_large_shapes_all"
    / "B__white_point_0.1/schira_fit.yaml"
)


def _render_cfg(repo: Path) -> RenderConfig:
    raw = yaml.safe_load((repo / "configs/stimuli/default.yaml").read_text()) or {}
    return RenderConfig(
        canvas_size=int(raw.get("canvas_size", 210)),
        pixels_per_deg=float(raw.get("pixels_per_deg", 35.0)),
        quadrant_extent_deg=float(raw.get("quadrant_extent_deg", 6.0)),
        background_gray=int(raw.get("background_gray", 128)),
    )


def main() -> None:
    repo = project_root()
    register_mapgeog()
    cfg = yaml.safe_load((repo / "configs/schira_encoding/default.yaml").read_text())
    spatial = tuple(int(x) for x in cfg["spatial_size"])
    monkey = str(cfg["monkey"])
    anchor = str(cfg["anchor_session"])
    render_cfg = _render_cfg(repo)
    _, yaml_params, yaml_affine, _ = load_schira_set(
        repo / str(cfg["schira_config"]), set_name=str(cfg["schira_set"])
    )
    fit_raw = yaml.safe_load((repo / BEST_FIT).read_text())
    fit_params = params_from_mapping(fit_raw["schira"])
    fit_affine = affine_from_mapping(fit_raw["affine"])

    pairs_root = resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo)
    register_root = resolve_output_root(cfg["paths"]["session_register_root"], repo)
    pairs = pd.read_parquet(
        encoding_pairs_manifest_path(pairs_root, monkey, "win_0035_0046")
    )
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    pairs = attach_stimulus_ids(pairs)
    means = build_anchor_stimulus_means(
        pairs,
        repo=repo,
        spatial_size=spatial,
        anchor_session=anchor,
        monkey=monkey,
        register_root=register_root,
        background_gray=float(render_cfg.background_gray),
        canvas_size=int(render_cfg.canvas_size),
        per_session=True,
    )
    hull = np.load(repo / NOISE_CEILING_HULL_MASK_RELPATH).astype(bool)
    out_dir = (repo / BEST_FIT).parents[1] / "overview" / "ink_vsd_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for date, sid in SAMPLES:
        key = f"{date}::{sid}"
        if key not in means:
            raise KeyError(f"missing {key}; have {sorted(means)[:8]}…")
        stim = means[key]
        rgb = load_stimulus_rgb(
            resolve_data_path(
                pairs.loc[
                    (pairs["date"].astype(str) == date)
                    & (pairs["stimulus_id"] == sid),
                    "image_path",
                ].iloc[0],
                repo,
            )
        )
        vsd = apply_mask_nan(stim.vsd_mean, hull)
        ink_yaml = forward_ink_mask_on_vsd(
            rgb,
            params=yaml_params,
            affine=yaml_affine,
            render_cfg=render_cfg,
            spatial_size=spatial,
        )
        ink_fit = forward_ink_mask_on_vsd(
            rgb,
            params=fit_params,
            affine=fit_affine,
            render_cfg=render_cfg,
            spatial_size=spatial,
        )
        label = f"{sid}\n{date}  n={stim.n_trials}"
        rows.append((rgb, vsd, ink_yaml, ink_fit, label, sid, date))
        for tag, ink in (("yaml", ink_yaml), ("fit", ink_fit)):
            plot_vsd_with_warped_stimulus(
                vsd,
                rgb,
                None,
                ink,
                out_dir / f"{date}__{sid}__{tag}.png",
                title=f"{sid} · {date} · {tag}",
                overlay_title=f"VSD + Schira ink ({tag})",
            )

    fig, axes = plt.subplots(len(rows), 4, figsize=(12.4, 3.15 * len(rows)), layout="constrained")
    for i, (rgb, vsd, ink_yaml, ink_fit, label, _sid, _date) in enumerate(rows):
        finite = vsd[np.isfinite(vsd)]
        lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_ylabel(label, fontsize=8)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 1].imshow(vsd, cmap=VSD_CMAP, vmin=lo, vmax=hi)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(vsd, cmap=VSD_CMAP, vmin=lo, vmax=hi)
        axes[i, 2].imshow(_ink_rgba_overlay(ink_yaml, outline_iters=0))
        axes[i, 2].axis("off")
        axes[i, 3].imshow(vsd, cmap=VSD_CMAP, vmin=lo, vmax=hi)
        axes[i, 3].imshow(_ink_rgba_overlay(ink_fit, outline_iters=0))
        axes[i, 3].axis("off")
    axes[0, 0].set_title("Stimulus", fontsize=10)
    axes[0, 1].set_title("Mean VSD (NC hull)", fontsize=10)
    axes[0, 2].set_title("Overlay · YAML Schira", fontsize=10)
    axes[0, 3].set_title("Overlay · fitted Schira+camera", fontsize=10)
    fig.suptitle(
        "Warped ink on mean VSD · is the mismatch cortical stretch?",
        fontsize=11,
    )
    grid = out_dir / "overlay_grid.png"
    fig.savefig(grid, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {grid}")


if __name__ == "__main__":
    main()
