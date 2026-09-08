#!/usr/bin/env python3
"""Schira LOO overview figures (flatten-ridge style).

1. One orig|recon grid: fold-mean maps, one row per held-out stimulus.
2. Pooled fold-level pixel-r (encoding) vs odd/even noise correlation.

Uses cached ``fold_mean_{orig,recon}.npy`` and ``fold_pairs.parquet`` — no
ridge re-fit. Odd/even halves are built from test trials via the same
anchor-aligned ``build_schira_xy`` path as encoding.

Example::

  scripts/py scripts/18_plot_schira_loo_figures.py \\
    --loo-dir experiments/schira_encoding/gandalf/loo/win_0035_0046/\\
vgg16_imagenet/block1_prepool/set-100718__anchor-100718a/protocol_B_noise_ceiling_hull
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.encoding.ridge_plotting import plot_reconstruction_grid
from src.evaluation.loss_roi import NOISE_CEILING_HULL_MASK_RELPATH
from src.evaluation.mask import apply_mask_nan, masked_map_summary
from src.evaluation.pixel_correlation import (
    pixel_correlation_across_trials,
    pixel_r2_across_trials,
)
from src.evaluation.plotting import plot_pixel_correlation_heatmap
from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import VSD_CMAP, register_mapgeog
from src.schira_encoding.io import build_schira_xy, resolve_output_root


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _discover_folds(loo_dir: Path) -> list[Path]:
    folds = sorted(
        p
        for p in loo_dir.iterdir()
        if p.is_dir()
        and (p.name.startswith("B__") or p.name.startswith("C__"))
        and (p / "fold_mean_orig.npy").is_file()
        and (p / "fold_mean_recon.npy").is_file()
    )
    return folds


def _write_folds_index(loo_dir: Path, fold_dirs: list[Path]) -> Path:
    folds_meta = []
    for fold_dir in fold_dirs:
        metrics = json.loads((fold_dir / "metrics.json").read_text())
        folds_meta.append(
            {
                "fold_id": fold_dir.name,
                "heldout_stimulus_id": metrics.get("heldout_stimulus_id"),
                "n_train": metrics.get("n_train"),
                "n_test": metrics.get("n_test"),
                "r_mean_test_masked": metrics.get("r_mean_test_masked"),
            }
        )
    path = loo_dir / "folds_index.yaml"
    path.write_text(
        yaml.safe_dump(
            {"n_folds": len(folds_meta), "folds": folds_meta},
            sort_keys=False,
        )
    )
    return path


def _hull_mask(repo: Path, spatial: tuple[int, int]) -> np.ndarray:
    mask = np.load(repo / NOISE_CEILING_HULL_MASK_RELPATH).astype(bool)
    if mask.shape != spatial:
        raise ValueError(f"Hull {mask.shape} != spatial {spatial}")
    return mask


def _plot_all_shapes_orig_recon(
    fold_dirs: list[Path],
    *,
    valid: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    samples = []
    for fold_dir in fold_dirs:
        metrics = json.loads((fold_dir / "metrics.json").read_text())
        sid = str(metrics.get("heldout_stimulus_id") or fold_dir.name)
        date = metrics.get("heldout_date")
        cond = metrics.get("heldout_condition")
        if (not date or not cond) and (fold_dir / "fold_pairs.parquet").is_file():
            pairs = pd.read_parquet(fold_dir / "fold_pairs.parquet")
            test = pairs[pairs["loo_split"] == "test"] if "loo_split" in pairs.columns else pairs
            if not test.empty:
                date = date or str(test["date"].iloc[0])
                if "condition" in test.columns:
                    cond = cond or str(test["condition"].iloc[0])
        if date:
            label = f"{sid}\n{date}" + (f"/{cond}" if cond else "")
        else:
            label = sid
        orig = np.load(fold_dir / "fold_mean_orig.npy").astype(np.float32)
        recon = np.load(fold_dir / "fold_mean_recon.npy").astype(np.float32)
        samples.append(
            (
                {
                    "date": str(date or "loo"),
                    "condition": str(cond or sid),
                    "stimulus_label": label,
                    "shape_type": "",
                    "trial_global_id": 0,
                    "split": "loo_test_mean",
                    "trial_dataset": "",
                },
                apply_mask_nan(orig, valid),
                apply_mask_nan(recon, valid),
            )
        )
    # Shared 1–99% clim across all orig|recon panels (easy fold comparison).
    plot_reconstruction_grid(
        samples,
        out_path,
        title=title,
    )


def _pooled_encoding_maps(
    fold_dirs: list[Path],
    *,
    hull: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    origs = []
    recons = []
    for fold_dir in fold_dirs:
        origs.append(np.load(fold_dir / "fold_mean_orig.npy").astype(np.float32))
        recons.append(np.load(fold_dir / "fold_mean_recon.npy").astype(np.float32))
    o = np.stack(origs, axis=0)
    r = np.stack(recons, axis=0)
    corr = pixel_correlation_across_trials(o, r)
    r2 = pixel_r2_across_trials(o, r)
    mean_r = float(masked_map_summary(corr, hull)["mean"])
    mean_r2 = float(masked_map_summary(r2, hull)["mean"])
    return (
        apply_mask_nan(corr, hull).astype(np.float32),
        apply_mask_nan(r2, hull).astype(np.float32),
        mean_r,
        mean_r2,
    )


def _odd_even_half_means(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Even indices 0::2, odd 1::2 (same convention as flatten OE)."""
    even = maps[0::2]
    odd = maps[1::2]
    if even.size == 0 or odd.size == 0:
        raise RuntimeError(
            f"Need both odd and even halves (n={maps.shape[0]})"
        )
    return (
        np.nanmean(even, axis=0).astype(np.float32),
        np.nanmean(odd, axis=0).astype(np.float32),
        int(odd.shape[0]),
        int(even.shape[0]),
    )


def _pooled_odd_even_maps(
    fold_dirs: list[Path],
    *,
    repo: Path,
    out_root: Path,
    monkey: str,
    model_slug: str,
    feature_layer: str,
    schira_set: str,
    anchor_session: str,
    spatial_size: tuple[int, int],
    register_root: Path,
    hull: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    odd_means: list[np.ndarray] = []
    even_means: list[np.ndarray] = []
    for fold_dir in fold_dirs:
        pairs_path = fold_dir / "fold_pairs.parquet"
        if not pairs_path.is_file():
            raise FileNotFoundError(pairs_path)
        fold_df = pd.read_parquet(pairs_path)
        test_df = fold_df[fold_df["loo_split"] == "test"].copy()
        test_df = test_df.sort_values("trial_global_id").reset_index(drop=True)
        if len(test_df) < 2:
            raise RuntimeError(
                f"{fold_dir.name}: need ≥2 test trials for odd/even, got {len(test_df)}"
            )
        _x, y = build_schira_xy(
            test_df,
            repo=repo,
            out_root=out_root,
            monkey=monkey,
            model_slug=model_slug,
            feature_layer=feature_layer,
            schira_set=schira_set,
            anchor_session=anchor_session,
            spatial_size=spatial_size,
            register_root=register_root,
        )
        odd_m, even_m, n_odd, n_even = _odd_even_half_means(y)
        odd_means.append(odd_m)
        even_means.append(even_m)
        print(
            f"  OE {fold_dir.name}: n_test={len(test_df)} "
            f"odd={n_odd} even={n_even}",
            flush=True,
        )
    odd_stack = np.stack(odd_means, axis=0)
    even_stack = np.stack(even_means, axis=0)
    corr = pixel_correlation_across_trials(odd_stack, even_stack)
    r2 = pixel_r2_across_trials(odd_stack, even_stack)
    mean_r = float(masked_map_summary(corr, hull)["mean"])
    mean_r2 = float(masked_map_summary(r2, hull)["mean"])
    return (
        apply_mask_nan(corr, hull).astype(np.float32),
        apply_mask_nan(r2, hull).astype(np.float32),
        mean_r,
        mean_r2,
    )


def _plot_enc_vs_oe(
    *,
    enc_r: np.ndarray,
    oe_r: np.ndarray,
    enc_r2: np.ndarray,
    oe_r2: np.ndarray,
    enc_mean_r: float,
    oe_mean_r: float,
    enc_mean_r2: float,
    oe_mean_r2: float,
    n_folds: int,
    out_dir: Path,
    title_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _one(left, right, left_t, right_t, path, label):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained")
        im = None
        for ax, title, arr in zip(axes, (left_t, right_t), (left, right)):
            im = ax.imshow(arr, cmap=VSD_CMAP, vmin=-1.0, vmax=1.0)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, label=label)
        fig.suptitle(f"{title_prefix} · n={n_folds} folds · NC hull", fontsize=11)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _one(
        enc_r,
        oe_r,
        f"Encoding fold-mean r\nmean={enc_mean_r:.3f}",
        f"Odd/even noise corr r\nmean={oe_mean_r:.3f}",
        out_dir / "encoding_vs_noise_corr_r.png",
        "Pearson r",
    )
    _one(
        enc_r2,
        oe_r2,
        f"Encoding fold-mean R²\nmean={enc_mean_r2:.3f}",
        f"Odd/even noise corr R²\nmean={oe_mean_r2:.3f}",
        out_dir / "encoding_vs_noise_corr_r2.png",
        "R²",
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 9.0), layout="constrained")
    panels = [
        (axes[0, 0], f"Encoding r\nmean={enc_mean_r:.3f}", enc_r),
        (axes[0, 1], f"Odd/even r\nmean={oe_mean_r:.3f}", oe_r),
        (axes[1, 0], f"Encoding R²\nmean={enc_mean_r2:.3f}", enc_r2),
        (axes[1, 1], f"Odd/even R²\nmean={oe_mean_r2:.3f}", oe_r2),
    ]
    for ax, title, arr in panels:
        im = ax.imshow(arr, cmap=VSD_CMAP, vmin=-1.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    fig.suptitle(
        f"{title_prefix} · encoding vs odd/even · n={n_folds}",
        fontsize=11,
    )
    fig.savefig(out_dir / "encoding_vs_noise_corr_r_and_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = project_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--loo-dir",
        type=Path,
        required=True,
        help="Schira Protocol B/C LOO leaf (…/protocol_B_… or protocol_C_…).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=repo / "configs/schira_encoding/default.yaml",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Use whatever fold_mean_*.npy folders exist (partial LOO).",
    )
    args = p.parse_args()
    register_mapgeog()

    loo_dir = args.loo_dir if args.loo_dir.is_absolute() else repo / args.loo_dir
    if not loo_dir.is_dir():
        raise FileNotFoundError(loo_dir)

    cfg = _load_yaml(args.config)
    params = _load_yaml(loo_dir / "params.yaml") if (loo_dir / "params.yaml").is_file() else {}
    monkey = str(params.get("monkey") or cfg.get("monkey") or "gandalf")
    schira_set = str(params.get("schira_set") or cfg["schira_set"])
    anchor = str(params.get("anchor_session") or cfg["anchor_session"])
    spatial = tuple(int(x) for x in cfg["spatial_size"])
    model_slug = str(params.get("model_slug") or "vgg16_imagenet")
    feature_layer = str(params.get("feature_layer") or "block1_prepool")
    window_id = str(params.get("window_id") or "win_0035_0046")

    fold_dirs = _discover_folds(loo_dir)
    if not fold_dirs:
        raise RuntimeError(f"No completed folds with fold_mean_*.npy under {loo_dir}")
    expected = params.get("n_folds") or (
        len(params["heldout_ids"]) if params.get("heldout_ids") else None
    )
    if expected and not args.allow_partial and len(fold_dirs) < int(expected):
        raise RuntimeError(
            f"Only {len(fold_dirs)}/{expected} folds complete; "
            "re-run later or pass --allow-partial"
        )

    _write_folds_index(loo_dir, fold_dirs)
    overview = loo_dir / "overview"
    overview.mkdir(parents=True, exist_ok=True)

    # Valid mask from first fold (LUT.valid ∩ NC hull).
    valid = np.load(fold_dirs[0] / "valid.npy").astype(bool)
    hull = _hull_mask(repo, spatial)
    plot_mask = valid & hull

    protocol = str(params.get("protocol") or "B")
    title = (
        f"Schira LOO Protocol {protocol} · fold-mean orig|recon · "
        f"{model_slug}/{feature_layer} · {window_id}"
    )
    grid_path = overview / "all_shapes_orig_recon.png"
    _plot_all_shapes_orig_recon(
        fold_dirs, valid=plot_mask, out_path=grid_path, title=title
    )
    print(f"Wrote {grid_path}")

    enc_r, enc_r2, enc_mean_r, enc_mean_r2 = _pooled_encoding_maps(
        fold_dirs, hull=plot_mask
    )
    np.save(overview / "pooled_fold_pixel_r__raw.npy", enc_r)
    np.save(overview / "pooled_fold_pixel_r2__raw.npy", enc_r2)
    plot_pixel_correlation_heatmap(
        enc_r,
        overview / "pooled_fold_pixel_r__raw.png",
        title=(
            f"Schira encoding pooled fold-mean r · mean={enc_mean_r:.3f} "
            f"(NC∩LUT) · n={len(fold_dirs)}"
        ),
    )
    plot_pixel_correlation_heatmap(
        enc_r2,
        overview / "pooled_fold_pixel_r2__raw.png",
        title=(
            f"Schira encoding pooled fold-mean R² · mean={enc_mean_r2:.3f} "
            f"(NC∩LUT) · n={len(fold_dirs)}"
        ),
    )
    print(
        f"Encoding pooled mean r={enc_mean_r:.3f} R²={enc_mean_r2:.3f} "
        f"(n={len(fold_dirs)})"
    )

    out_root = resolve_output_root(cfg["paths"]["schira_encoding_root"], repo)
    register_root = resolve_output_root(cfg["paths"]["session_register_root"], repo)
    print("Computing odd/even noise corr (anchor-aligned test trials)…", flush=True)
    oe_r, oe_r2, oe_mean_r, oe_mean_r2 = _pooled_odd_even_maps(
        fold_dirs,
        repo=repo,
        out_root=out_root,
        monkey=monkey,
        model_slug=model_slug,
        feature_layer=feature_layer,
        schira_set=schira_set,
        anchor_session=anchor,
        spatial_size=spatial,
        register_root=register_root,
        hull=plot_mask,
    )
    oe_dir = loo_dir / "noise_corr_odd_even"
    oe_dir.mkdir(parents=True, exist_ok=True)
    np.save(oe_dir / "r_map_pooled_folds_masked__raw.npy", oe_r)
    np.save(oe_dir / "r2_map_pooled_folds_masked__raw.npy", oe_r2)
    plot_pixel_correlation_heatmap(
        oe_r,
        oe_dir / "r_map_pooled_folds__raw.png",
        title=(
            f"Schira odd/even pooled r · mean={oe_mean_r:.3f} "
            f"(NC∩LUT) · n={len(fold_dirs)}"
        ),
    )
    (oe_dir / "summary.json").write_text(
        json.dumps(
            {
                "n_folds": len(fold_dirs),
                "mean_r_hull": oe_mean_r,
                "mean_r2_hull": oe_mean_r2,
                "encoding_mean_r_hull": enc_mean_r,
                "encoding_mean_r2_hull": enc_mean_r2,
                "window_id": window_id,
                "feature_layer": feature_layer,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"OE pooled mean r={oe_mean_r:.3f} R²={oe_mean_r2:.3f}")

    _plot_enc_vs_oe(
        enc_r=enc_r,
        oe_r=oe_r,
        enc_r2=enc_r2,
        oe_r2=oe_r2,
        enc_mean_r=enc_mean_r,
        oe_mean_r=oe_mean_r,
        enc_mean_r2=enc_mean_r2,
        oe_mean_r2=oe_mean_r2,
        n_folds=len(fold_dirs),
        out_dir=overview,
        title_prefix=f"Schira LOO B · {model_slug}/{feature_layer}",
    )
    print(f"Wrote compare figures under {overview}")

    (overview / "summary.json").write_text(
        json.dumps(
            {
                "n_folds": len(fold_dirs),
                "fold_ids": [p.name for p in fold_dirs],
                "encoding_mean_r_hull": enc_mean_r,
                "encoding_mean_r2_hull": enc_mean_r2,
                "oe_mean_r_hull": oe_mean_r,
                "oe_mean_r2_hull": oe_mean_r2,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
