"""Helpers for backbone feature-layer sweeps and cross-model reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.DL_features.schema import model_slug
from src.encoding.schema import ridge_output_dir
from src.evaluation.compare import comparison_output_dir
from src.paths import resolve_data_path


def layer_sweep_dir(repo: Path, cfg: dict, window_id: str, model_slug_str: str) -> Path:
    return comparison_output_dir(repo, cfg, window_id) / f"layer_sweep_{model_slug_str}"


def layer_comparison_csv_path(
    repo: Path, cfg: dict, window_id: str, model_slug_str: str
) -> Path:
    return layer_sweep_dir(repo, cfg, window_id, model_slug_str) / "layer_comparison.csv"


def ridge_artifacts_exist(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    model_slug_str: str,
    feature_layer: str,
) -> bool:
    ridge_root = resolve_data_path(cfg["paths"]["ridge_encode_root"], repo)
    ridge_dir = ridge_output_dir(
        ridge_root, cfg["monkey"], window_id, model_slug_str, feature_layer
    )
    return (ridge_dir / "model.joblib").exists() and (
        ridge_dir / "metrics.json"
    ).exists()


def pixel_eval_json_path(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    model_slug_str: str,
    feature_layer: str,
    split: str,
) -> Path:
    eval_root = repo / cfg["paths"].get("evaluation_plots_root", "plots/evaluation")
    return (
        eval_root
        / cfg["monkey"]
        / window_id
        / model_slug_str
        / feature_layer
        / f"pixel_evaluation_{split}.json"
    )


def pixel_eval_exists(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    model_slug_str: str,
    feature_layer: str,
    split: str,
) -> bool:
    return pixel_eval_json_path(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        model_slug_str=model_slug_str,
        feature_layer=feature_layer,
        split=split,
    ).exists()


def validate_layer_artifacts(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    model_slug_str: str,
    feature_layer: str,
    split: str,
    require_ridge: bool = True,
    require_eval: bool = True,
) -> list[str]:
    """Return human-readable missing artifact messages."""
    missing: list[str] = []
    if require_ridge and not ridge_artifacts_exist(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        model_slug_str=model_slug_str,
        feature_layer=feature_layer,
    ):
        missing.append("ridge model/metrics")
    if require_eval and not pixel_eval_exists(
        repo=repo,
        cfg=cfg,
        window_id=window_id,
        model_slug_str=model_slug_str,
        feature_layer=feature_layer,
        split=split,
    ):
        missing.append(f"pixel evaluation ({split})")
    return missing


def plot_layer_mean_pixel_r(
    df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    layer_order: list[str] | None = None,
    metric_col: str = "eval_mean_r_masked",
    ylabel: str = "Mean pixel r (masked)",
) -> Path:
    """Bar chart of masked mean pixel r across feature layers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return output_path

    plot_df = df.copy()
    if layer_order:
        order = [
            layer for layer in layer_order if layer in set(plot_df["feature_layer"])
        ]
        plot_df["feature_layer"] = pd.Categorical(
            plot_df["feature_layer"], categories=order, ordered=True
        )
        plot_df = plot_df.sort_values("feature_layer").reset_index(drop=True)
    else:
        plot_df = plot_df.sort_values("feature_layer").reset_index(drop=True)

    labels = plot_df["feature_layer"].tolist()
    vals = pd.to_numeric(plot_df[metric_col], errors="coerce").to_numpy(dtype=float)
    colors = ["lightgray" if not np.isfinite(v) else "steelblue" for v in vals]
    heights = np.where(np.isfinite(vals), vals, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    bars = ax.bar(range(len(labels)), heights, width=0.65, color=colors)
    ax.set_xticks(range(len(labels)), labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, color="k", linewidth=0.5)
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size:
        ymax = float(np.nanmax(finite_vals))
        ax.set_ylim(bottom=min(0.0, float(np.nanmin(finite_vals)) * 1.1), top=ymax * 1.15)

    for bar, val in zip(bars, vals):
        label = f"{val:.3f}" if np.isfinite(val) else "N/A"
        y = bar.get_height() if np.isfinite(val) else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def select_best_layer(
    df: pd.DataFrame,
    *,
    metric_col: str = "eval_mean_r_masked",
    layer_col: str = "feature_layer",
) -> dict[str, Any]:
    """Pick the layer with the highest finite metric value."""
    if df.empty:
        raise ValueError("Cannot select best layer from empty comparison table")
    work = df.copy()
    work["_metric"] = pd.to_numeric(work[metric_col], errors="coerce")
    valid = work[np.isfinite(work["_metric"])]
    if valid.empty:
        raise ValueError(f"No finite values in column {metric_col!r}")
    row = valid.sort_values("_metric", ascending=False).iloc[0]
    return {
        "feature_layer": str(row[layer_col]),
        "metric_col": metric_col,
        "metric_value": float(row["_metric"]),
        "row": row.to_dict(),
    }


def load_layer_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Layer comparison CSV not found: {path}")
    return pd.read_csv(path)


def load_pixel_eval_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        payload = json.load(f)
    return dict(payload.get("metrics", {}))


def model_slug_from_yaml(model_cfg_path: Path) -> str:
    import yaml

    with model_cfg_path.open() as f:
        model_cfg = yaml.safe_load(f)
    return model_slug(model_cfg)
