"""Collect Ridge and pixel-evaluation metrics across backbone runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.DL_features.schema import model_slug
from src.encoding.schema import ridge_output_dir
from src.paths import resolve_data_path


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _feature_shape_from_stimulus_config(path: Path) -> str | None:
    cfg = _load_json(path)
    if not cfg:
        return None
    shape = cfg.get("feature_shape")
    if shape is None:
        return None
    return str(shape)


def collect_backbone_metrics(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    model_cfg_paths: list[Path],
    split: str = "test",
) -> pd.DataFrame:
    """
    Gather metrics for each model config from ridge + evaluation artifacts.

    Returns one row per (model_slug, feature_layer).
    """
    monkey = cfg["monkey"]
    ridge_root = resolve_data_path(cfg["paths"]["ridge_encode_root"], repo)
    stim_feat_root = resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo)
    eval_root = repo / cfg["paths"].get("evaluation_plots_root", "plots/evaluation")

    rows: list[dict[str, Any]] = []
    for model_path in model_cfg_paths:
        with model_path.open() as f:
            model_cfg = yaml.safe_load(f)
        slug = model_slug(model_cfg)
        feature_layer = str(model_cfg.get("feature_layer", "layer3"))

        ridge_dir = ridge_output_dir(ridge_root, monkey, window_id, slug, feature_layer)
        ridge_metrics = _load_json(ridge_dir / "metrics.json") or {}

        eval_json = _load_json(
            eval_root / monkey / window_id / slug / feature_layer / f"pixel_evaluation_{split}.json"
        )
        eval_metrics = (eval_json or {}).get("metrics", {})

        stim_cfg_path = (
            stim_feat_root / monkey / slug / feature_layer / "config.json"
        )
        feature_shape = _feature_shape_from_stimulus_config(stim_cfg_path)

        row: dict[str, Any] = {
            "model_config": str(model_path.relative_to(repo)),
            "model_slug": slug,
            "feature_layer": feature_layer,
            "model_type": model_cfg.get("type", "resnet"),
            "preprocess": model_cfg.get("preprocess", "imagenet_rgb"),
            "alpha": ridge_metrics.get("alpha"),
            "n_train": ridge_metrics.get("n_train"),
            "r_mean_train": ridge_metrics.get("r_mean_train"),
            "r_mean_val": ridge_metrics.get("r_mean_val"),
            "r_mean_test": ridge_metrics.get("r_mean_test"),
            "eval_mean_r": eval_metrics.get("mean_r"),
            "eval_mean_r2": eval_metrics.get("mean_r2"),
            "eval_n_trials": eval_metrics.get("n_test_trials"),
            "feature_shape": feature_shape,
            "ridge_dir": str(ridge_dir.relative_to(repo.parent))
            if ridge_dir.exists()
            else None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def comparison_output_dir(repo: Path, cfg: dict, window_id: str) -> Path:
    monkey = cfg["monkey"]
    root = repo / cfg["paths"].get("evaluation_plots_root", "plots/evaluation")
    return root / monkey / window_id / "backbone_comparison"
