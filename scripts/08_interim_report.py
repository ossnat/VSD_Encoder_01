#!/usr/bin/env python3
"""Interim report figures: masked test-condition maps + quantitative comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.DL_features.schema import model_slug
from src.encoding.ridge import attach_feature_paths
from src.encoding.schema import encoding_pairs_manifest_path, ridge_output_dir
from src.evaluation.compare import collect_backbone_metrics, comparison_output_dir
from src.evaluation.mask import mask_from_eval_cfg, masked_pearson_r
from src.evaluation.pixel_correlation import (
    load_reconstructed_maps,
    load_trial_mean_maps,
    pixel_correlation_across_conditions,
    pixel_correlation_across_trials,
    pixel_r2_across_conditions,
    stack_condition_mean_maps,
)
from src.evaluation.plotting import (
    plot_backbone_correlation_comparison,
    plot_metrics_bar_comparison,
    plot_per_condition_trial_r,
    plot_pixel_correlation_heatmap,
    plot_pixel_r2_heatmap,
    plot_test_conditions_grid,
)
from src.evaluation.mask import apply_mask_nan, masked_map_summary
from src.paths import project_root, resolve_data_path

GWP_BEST = (
    "../Data/VSD_Encoder_01/grid_search/gwp/gandalf/win_0032_0042/best_model.yaml"
)


def _comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    if "label" not in display.columns:
        display["label"] = display["model_slug"].map(
            lambda s: "ResNet18" if "resnet18" in s else "GWP (best, γ=0.3)"
        )
    cols = {
        "label": "Model",
        "r_mean_test_masked": "Trial r (test, masked)",
        "eval_mean_r_masked": "Mean pixel r across trials (masked)",
        "eval_mean_r2_masked": "Mean pixel R² across trials (masked)",
        "mean_r_across_conditions_masked": "Mean pixel r across conditions (masked)",
        "mean_r2_across_conditions_masked": "Mean pixel R² across conditions (masked)",
    }
    use = {k: v for k, v in cols.items() if k in display.columns}
    out = display[list(use.keys())].rename(columns=use)
    for col in out.columns[1:]:
        out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    return out


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(default_path: Path, window_path: Path, ridge_path: Path) -> dict:
    cfg = _load_yaml(default_path)
    cfg.update(_load_yaml(window_path))
    cfg["ridge"] = _load_yaml(ridge_path)
    return cfg


def _model_label(model_cfg: dict, slug: str) -> str:
    if "resnet18" in slug:
        return "ResNet18"
    gamma = model_cfg.get("gwp", {}).get("gamma")
    return f"GWP (best, γ={gamma})" if gamma is not None else "GWP (best)"


def _condition_balanced_spatial_r(
    conditions: list[dict[str, object]],
    mask: np.ndarray,
) -> float:
    """Equal-weight mean of masked spatial r(mean orig, recon) per condition."""
    rs = [
        masked_pearson_r(
            np.asarray(c["original"]),
            np.asarray(c["reconstruction"]),
            mask,
        )
        for c in conditions
    ]
    finite = [r for r in rs if np.isfinite(r)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def render_interim_report(
    cfg: dict,
    *,
    model_cfg_paths: list[Path],
    repo: Path | None = None,
    split: str = "test",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    repo = repo or project_root()
    window_id = cfg.get("window_id") or (
        f"win_{int(cfg['start_frame']):04d}_{int(cfg['end_frame']):04d}"
    )
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    ridge_cfg = cfg["ridge"]
    eval_mask = mask_from_eval_cfg(ridge_cfg.get("evaluation"), spatial_size)
    if eval_mask is None:
        raise ValueError("evaluation.use_mask must be true for interim report")
    mask_radius = int(ridge_cfg["evaluation"]["mask_radius"])

    out_dir = output_dir or (
        comparison_output_dir(repo, cfg, window_id) / "interim_report"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        cfg["monkey"],
        window_id,
    )
    pairs = pd.read_parquet(pairs_path)
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    split_df = pairs[pairs["split"] == split].copy().reset_index(drop=True)

    originals_all = load_trial_mean_maps(
        split_df,
        repo=repo,
        spatial_size=spatial_size,
        start_frame=int(cfg["start_frame"]),
        end_frame=int(cfg["end_frame"]),
        avg_method=cfg.get("avg_method", "mean"),
    )

    group_indices: list[np.ndarray] = []
    per_cond_trial_r: list[dict[str, object]] = []
    corr_panels: list[tuple[str, np.ndarray]] = []
    corr_across_cond_panels: list[tuple[str, np.ndarray]] = []
    r2_across_cond_panels: list[tuple[str, np.ndarray]] = []
    balanced_rows: list[dict[str, str]] = []
    across_cond_rows: list[dict[str, str]] = []

    metrics_df = collect_backbone_metrics(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        model_cfg_paths=model_cfg_paths,
        split=split,
    )
    metrics_df["label"] = [
        _model_label(_load_yaml(p), model_slug(_load_yaml(p)))
        for p in model_cfg_paths
    ]

    for model_path in model_cfg_paths:
        model_cfg = _load_yaml(model_path)
        slug = model_slug(model_cfg)
        feature_layer = str(model_cfg.get("feature_layer", "layer3"))
        label = _model_label(model_cfg, slug)

        features_root = resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo)
        pairs_feat = attach_feature_paths(
            pairs,
            features_root=features_root,
            monkey=cfg["monkey"],
            model_slug=slug,
            feature_layer=feature_layer,
            repo=repo,
        )
        model_split = pairs_feat[pairs_feat["split"] == split].copy().reset_index(drop=True)

        result = joblib.load(
            ridge_output_dir(
                resolve_data_path(cfg["paths"]["ridge_encode_root"], repo),
                cfg["monkey"],
                window_id,
                slug,
                feature_layer,
            )
            / "model.joblib"
        )["result"]
        result.spatial_size = spatial_size

        recon_all = load_reconstructed_maps(
            model_split,
            result=result,
            repo=repo,
            spatial_size=spatial_size,
        )

        if not group_indices:
            for (_, _), grp in model_split.groupby(["date", "condition"], sort=True):
                group_indices.append(grp.index.to_numpy())

        conditions: list[dict[str, object]] = []
        for (date, condition), grp in model_split.groupby(["date", "condition"], sort=True):
            idx = grp.index.to_numpy()
            mean_orig = np.nanmean(originals_all[idx], axis=0).astype(np.float32)
            recon = recon_all[idx[0]]
            trial_rs = [
                masked_pearson_r(originals_all[i], recon_all[i], eval_mask)
                for i in idx
            ]
            mean_trial_r = float(np.nanmean(trial_rs))
            conditions.append(
                {
                    "date": str(date),
                    "condition": str(condition),
                    "n_trials": int(len(grp)),
                    "original": mean_orig,
                    "reconstruction": recon,
                    "trial_r_masked": mean_trial_r,
                }
            )
            per_cond_trial_r.append(
                {
                    "model_label": label,
                    "model_slug": slug,
                    "date": str(date),
                    "condition": str(condition),
                    "n_trials": int(len(grp)),
                    "trial_r_masked": mean_trial_r,
                }
            )

        balanced_r = _condition_balanced_spatial_r(conditions, eval_mask)
        row_idx = metrics_df["model_slug"] == slug
        metrics_df.loc[row_idx, "spatial_r_cond_mean_masked"] = balanced_r
        balanced_rows.append(
            {
                "Model": label,
                "Spatial r (condition-mean, balanced)": f"{balanced_r:.4f}"
                if np.isfinite(balanced_r)
                else "—",
            }
        )

        cond_orig, cond_recon, cond_meta = stack_condition_mean_maps(
            model_split, originals_all, recon_all
        )
        corr_across = pixel_correlation_across_conditions(cond_orig, cond_recon)
        r2_across = pixel_r2_across_conditions(cond_orig, cond_recon)
        mean_r_ac = masked_map_summary(corr_across, eval_mask)["mean"]
        mean_r2_ac = masked_map_summary(r2_across, eval_mask)["mean"]
        metrics_df.loc[row_idx, "mean_r_across_conditions_masked"] = mean_r_ac
        metrics_df.loc[row_idx, "mean_r2_across_conditions_masked"] = mean_r2_ac
        across_cond_rows.append(
            {
                "Model": label,
                "Mean pixel r across conditions (masked)": (
                    f"{mean_r_ac:.4f}" if np.isfinite(mean_r_ac) else "—"
                ),
                "Mean pixel R² across conditions (masked)": (
                    f"{mean_r2_ac:.4f}" if np.isfinite(mean_r2_ac) else "—"
                ),
            }
        )
        corr_across_cond_panels.append(
            (f"{label}\nmean r = {mean_r_ac:.3f}", apply_mask_nan(corr_across, eval_mask))
        )
        r2_across_cond_panels.append(
            (f"{label}\nmean R² = {mean_r2_ac:.3f}", apply_mask_nan(r2_across, eval_mask))
        )

        corr_map = pixel_correlation_across_trials(originals_all, recon_all)
        corr_map = apply_mask_nan(corr_map, eval_mask)
        mr = float(metrics_df.loc[row_idx, "eval_mean_r_masked"].iloc[0])
        corr_panels.append((f"{label}\nmean r = {mr:.3f}", corr_map))

        grid_path = out_dir / f"{slug}_{split}_conditions_masked.png"
        plot_test_conditions_grid(
            conditions,
            grid_path,
            mask=eval_mask,
            spatial_size=spatial_size,
            mask_radius=mask_radius,
            model_label=label,
            split=split,
        )

        # Per-model heatmaps for across-conditions metrics
        mean_orig_underlay = np.nanmean(originals_all, axis=0).astype(np.float32)
        plot_pixel_correlation_heatmap(
            apply_mask_nan(corr_across, eval_mask),
            out_dir / f"{slug}_{split}_pixel_r_across_conditions.png",
            title=(
                f"{label} | pixel r across conditions ({split}) | "
                f"C={len(cond_meta)} | mean r={mean_r_ac:.3f} | mask r={mask_radius}"
            ),
            underlay=mean_orig_underlay,
        )
        plot_pixel_r2_heatmap(
            apply_mask_nan(r2_across, eval_mask),
            out_dir / f"{slug}_{split}_pixel_r2_across_conditions.png",
            title=(
                f"{label} | pixel R² across conditions ({split}) | "
                f"C={len(cond_meta)} | mean R²={mean_r2_ac:.3f} | mask r={mask_radius}"
            ),
        )

    table = _comparison_table(metrics_df)
    balanced_df = pd.DataFrame(balanced_rows)
    table = table.merge(balanced_df, on="Model", how="left")
    table_path = out_dir / f"comparison_table_{split}.csv"
    table.to_csv(table_path, index=False)

    bar_path = out_dir / f"comparison_bars_{split}.png"
    plot_metrics_bar_comparison(
        metrics_df,
        bar_path,
        title=f"Masked test metrics (r={mask_radius}) | {window_id}",
    )

    cond_bar_path = out_dir / f"per_condition_trial_r_{split}.png"
    plot_per_condition_trial_r(
        pd.DataFrame(per_cond_trial_r),
        cond_bar_path,
        title=f"Mean trial r per test condition (masked r={mask_radius})",
    )

    mean_original = np.nanmean(originals_all, axis=0).astype(np.float32)

    heatmap_path = out_dir / f"pixel_correlation_side_by_side_{split}.png"
    plot_backbone_correlation_comparison(
        corr_panels,
        mean_original,
        heatmap_path,
        title=(
            f"Pixel r across trials ({split}, masked r={mask_radius}) | "
            f"gray = trial-mean VSD"
        ),
    )

    heatmap_ac_path = out_dir / f"pixel_r_across_conditions_side_by_side_{split}.png"
    plot_backbone_correlation_comparison(
        corr_across_cond_panels,
        mean_original,
        heatmap_ac_path,
        title=(
            f"Pixel r across conditions ({split}, masked r={mask_radius}) | "
            f"one sample per condition | gray = trial-mean VSD"
        ),
    )

    heatmap_r2_ac_path = out_dir / f"pixel_r2_across_conditions_side_by_side_{split}.png"
    plot_backbone_correlation_comparison(
        r2_across_cond_panels,
        mean_original,
        heatmap_r2_ac_path,
        title=(
            f"Pixel R² across conditions ({split}, masked r={mask_radius}) | "
            f"one sample per condition | gray = trial-mean VSD"
        ),
    )

    summary = {
        "window_id": window_id,
        "split": split,
        "mask_radius": mask_radius,
        "n_conditions": int(
            split_df.groupby(["date", "condition"]).ngroups
        ),
        "table": table.to_dict(orient="records"),
        "per_condition_trial_r": per_cond_trial_r,
        "outputs": {
            "table": str(table_path.relative_to(repo)),
            "bars": str(bar_path.relative_to(repo)),
            "per_condition_bars": str(cond_bar_path.relative_to(repo)),
            "heatmap_across_trials": str(heatmap_path.relative_to(repo)),
            "heatmap_r_across_conditions": str(heatmap_ac_path.relative_to(repo)),
            "heatmap_r2_across_conditions": str(heatmap_r2_ac_path.relative_to(repo)),
        },
    }
    with (out_dir / f"interim_report_{split}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return {
        "table": table_path,
        "bars": bar_path,
        "per_condition_bars": cond_bar_path,
        "heatmap": heatmap_path,
        "heatmap_r_across_conditions": heatmap_ac_path,
        "heatmap_r2_across_conditions": heatmap_r2_ac_path,
        "out_dir": out_dir,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/default.yaml",
    )
    parser.add_argument(
        "--window",
        type=Path,
        default=project_root() / "configs/windows/evoked_32_42.yaml",
    )
    parser.add_argument(
        "--ridge-config",
        type=Path,
        default=project_root() / "configs/ridge/default.yaml",
    )
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        dest="models",
        default=None,
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    cfg = _merge_config(args.config, args.window, args.ridge_config)

    if args.models:
        model_paths = [p if p.is_absolute() else repo / p for p in args.models]
    else:
        model_paths = [
            repo / "configs/models/resnet18.yaml",
            repo / GWP_BEST,
        ]

    paths = render_interim_report(
        cfg,
        model_cfg_paths=model_paths,
        repo=repo,
        split=args.split,
        output_dir=args.output_dir,
    )

    table = pd.read_csv(paths["table"])
    print(table.to_string(index=False))
    print(f"\nOutput dir: {paths['out_dir'].relative_to(repo)}")
    for key, path in paths.items():
        if key != "out_dir":
            print(f"  {key}: {path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
