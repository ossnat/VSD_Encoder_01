"""Grid search over Gabor GWP hyperparameters using the validation split."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.DL_features.backbone import build_feature_extractor, default_device
from src.DL_features.extract_stimulus import extract_stimulus_features
from src.DL_features.schema import model_slug, stimulus_feature_dir, stimulus_map_path
from src.encoding.pairs import dedupe_stimulus_manifest
from src.encoding.ridge import (
    attach_feature_paths,
    build_xy,
    fit_ridge_encoder,
    pearson_r,
    predict_maps,
)
from src.encoding.schema import encoding_pairs_manifest_path, ridge_output_dir
from src.evaluation.mask import mask_from_eval_cfg, masked_pearson_r
from src.paths import resolve_data_path
from src.stimuli.schema import manifest_path as stimulus_manifest_path


def _nested_set(cfg: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _nested_get(cfg: dict, dotted_key: str) -> Any:
    node: Any = cfg
    for part in dotted_key.split("."):
        node = node[part]
    return node


def expand_param_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos: list[dict[str, Any]] = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def gwp_variant_id(params: dict[str, Any]) -> str:
    """Short unique slug fragment for one hyperparameter combo."""
    parts: list[str] = []
    for key in sorted(params):
        val = params[key]
        short = key.split(".")[-1]
        if isinstance(val, float):
            parts.append(f"{short}{val:g}".replace(".", "p"))
        else:
            parts.append(f"{short}{val}")
    return "_".join(parts)


def build_model_cfg(base_cfg: dict, params: dict[str, Any]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for key, value in params.items():
        _nested_set(cfg, key, value)
    cfg["variant"] = gwp_variant_id(params)
    return cfg


def _mean_split_r(
    pairs: pd.DataFrame,
    *,
    split: str,
    result,
    repo: Path,
    spatial_size: tuple[int, int],
    eval_mask: np.ndarray | None = None,
) -> float:
    split_df = pairs[pairs["split"] == split]
    if split_df.empty:
        return float("nan")
    x_split, y_split = build_xy(split_df, repo=repo, spatial_size=spatial_size)
    preds = predict_maps(result, x_split, spatial_size)
    y_true = y_split.reshape(len(split_df), *spatial_size)
    if eval_mask is not None:
        rs = [
            masked_pearson_r(y_true[i], preds[i], eval_mask)
            for i in range(len(split_df))
        ]
    else:
        rs = [pearson_r(y_true[i], preds[i]) for i in range(len(split_df))]
    return float(np.nanmean(rs))


def run_gwp_trial(
    model_cfg: dict,
    *,
    cfg: dict,
    repo: Path,
    window_id: str,
    ridge_cfg: dict,
    overwrite_features: bool = False,
) -> dict[str, Any]:
    """
    Extract stimulus features, fit Ridge on train, return train/val metrics.

    Artifacts are written under model_slug directories (includes variant suffix).
    """
    monkey = cfg["monkey"]
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    feature_layer = str(model_cfg.get("feature_layer", "energy"))
    slug = model_slug(model_cfg)
    eval_mask = mask_from_eval_cfg(ridge_cfg.get("evaluation"), spatial_size)

    stimuli_root = resolve_data_path(cfg["paths"]["stimuli_root"], repo)
    features_root = resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo)
    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        monkey,
        window_id,
    )
    pairs = pd.read_parquet(pairs_path)
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()

    stim_manifest_path = stimulus_manifest_path(stimuli_root, monkey)
    manifest = dedupe_stimulus_manifest(pd.read_parquet(stim_manifest_path))

    model = build_feature_extractor(model_cfg, feature_layer=feature_layer)
    device = default_device()
    out_dir = stimulus_feature_dir(features_root, monkey, slug, feature_layer)
    path_fn = lambda session, condition: stimulus_map_path(
        features_root, monkey, slug, feature_layer, session, condition
    )
    extract_stimulus_features(
        model,
        manifest,
        repo_root=repo,
        map_path_fn=path_fn,
        feature_layer=feature_layer,
        model_cfg=model_cfg,
        input_size=int(model_cfg.get("input_size", 224)),
        imagenet_normalize=bool(model_cfg.get("imagenet_normalize", False)),
        batch_size=int(model_cfg.get("batch_size", 32)),
        device=device,
        overwrite=overwrite_features,
    )

    pairs = attach_feature_paths(
        pairs,
        features_root=features_root,
        monkey=monkey,
        model_slug=slug,
        feature_layer=feature_layer,
        repo=repo,
    )
    train_df = pairs[pairs["split"] == "train"].copy()
    if train_df.empty:
        raise RuntimeError("No training trials for grid search")

    x_train, y_train = build_xy(train_df, repo=repo, spatial_size=spatial_size)
    result = fit_ridge_encoder(
        x_train,
        y_train,
        alphas=np.asarray(ridge_cfg["alphas"], dtype=np.float64),
        cv_folds=int(ridge_cfg.get("cv_folds", 5)),
        standardize_features=bool(ridge_cfg.get("standardize_features", True)),
    )
    result.spatial_size = spatial_size
    result.feature_layer = feature_layer
    result.model_slug = slug

    ridge_dir = ridge_output_dir(
        resolve_data_path(cfg["paths"]["ridge_encode_root"], repo),
        monkey,
        window_id,
        slug,
        feature_layer,
    )
    ridge_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_slug": slug,
        "variant": model_cfg.get("variant"),
        "feature_layer": feature_layer,
        "alpha": float(result.alpha),
        "n_train": int(len(train_df)),
        "r_mean_train": _mean_split_r(
            pairs,
            split="train",
            result=result,
            repo=repo,
            spatial_size=spatial_size,
        ),
        "r_mean_val": _mean_split_r(
            pairs,
            split="val",
            result=result,
            repo=repo,
            spatial_size=spatial_size,
        ),
    }
    if eval_mask is not None:
        metrics["r_mean_train_masked"] = _mean_split_r(
            pairs,
            split="train",
            result=result,
            repo=repo,
            spatial_size=spatial_size,
            eval_mask=eval_mask,
        )
        metrics["r_mean_val_masked"] = _mean_split_r(
            pairs,
            split="val",
            result=result,
            repo=repo,
            spatial_size=spatial_size,
            eval_mask=eval_mask,
        )
    with (ridge_dir / "grid_search_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run_grid_search(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    base_model_cfg: dict,
    param_grid: dict[str, list[Any]],
    ridge_cfg: dict,
    objective: str = "r_mean_val",
    overwrite_features: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for params in expand_param_grid(param_grid):
        model_cfg = build_model_cfg(base_model_cfg, params)
        row = dict(params)
        try:
            metrics = run_gwp_trial(
                model_cfg,
                cfg=cfg,
                repo=repo,
                window_id=window_id,
                ridge_cfg=ridge_cfg,
                overwrite_features=overwrite_features,
            )
            row.update(metrics)
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 — collect failures per combo
            row["status"] = f"error: {exc}"
            row[objective] = float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    if objective in df.columns:
        df = df.sort_values(objective, ascending=False, na_position="last")
    return df


def _to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_grid_search_results(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    base_model_cfg: dict,
    param_grid: dict[str, list[Any]],
    objective: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "grid_search_results.csv"
    df.to_csv(csv_path, index=False)

    best_path = out_dir / "best_model.yaml"
    best_row = df[df["status"] == "ok"].head(1)
    if not best_row.empty:
        params = {
            key: _to_python(best_row.iloc[0][key])
            for key in param_grid
            if key in best_row.columns
        }
        best_cfg = build_model_cfg(base_model_cfg, params)
        with best_path.open("w") as f:
            yaml.safe_dump(best_cfg, f, sort_keys=False)

    summary = {
        "objective": objective,
        "n_trials": int(len(df)),
        "n_ok": int((df["status"] == "ok").sum()),
        "best": best_row.iloc[0].to_dict() if not best_row.empty else None,
    }
    with (out_dir / "grid_search_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    return csv_path
