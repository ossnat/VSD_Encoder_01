"""Feature-map extraction loop over averaged NetCDF trials."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.DL_features.preprocess import preprocess_image


def _load_image(nc_path: Path) -> np.ndarray:
    da = xr.open_dataarray(nc_path)
    arr = da.values.astype(np.float32)
    da.close()
    return arr


def _feature_shape_str(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(int(x)) for x in shape) + ")"


def extract_features(
    model: torch.nn.Module,
    manifest: pd.DataFrame,
    *,
    repo_root: Path,
    map_path_fn,
    feature_layer: str,
    input_size: int,
    input_channels: int,
    input_scaling: str,
    imagenet_normalize: bool,
    batch_size: int,
    device: torch.device,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    """
    Extract and save (C, H, W) maps per trial.

    Returns (feature_manifest, written_count, skipped_count).
    """
    rows: list[dict] = []
    pending_tensors: list[torch.Tensor] = []
    pending_meta: list[dict] = []
    n_written = 0
    n_skipped = 0

    model = model.to(device)
    model.eval()

    def flush_batch() -> None:
        nonlocal pending_tensors, pending_meta, n_written
        if not pending_tensors:
            return
        batch = torch.stack(pending_tensors, dim=0).to(device)
        with torch.no_grad():
            feats = model(batch).detach().cpu().numpy().astype(np.float32)
        # feats: (N, C, H, W)
        for meta, feat_map in zip(pending_meta, feats):
            out_path = meta["map_path_abs"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, feat_map)
            n_written += 1
            row = dict(meta["row"])
            row["feature_path"] = meta["feature_path"]
            row["feature_layer"] = feature_layer
            row["feature_shape"] = _feature_shape_str(feat_map.shape)
            row["feature_channels"] = int(feat_map.shape[0])
            row["feature_height"] = int(feat_map.shape[1])
            row["feature_width"] = int(feat_map.shape[2])
            row["feature_n_elements"] = int(feat_map.size)
            rows.append(row)
        pending_tensors = []
        pending_meta = []

    for row in manifest.itertuples(index=False):
        map_abs = Path(map_path_fn(int(row.trial_global_id)))
        map_rel = str(map_abs.resolve().relative_to(repo_root.resolve().parent))
        base = {
            "trial_global_id": int(row.trial_global_id),
            "monkey": row.monkey,
            "date": row.date,
            "condition": row.condition,
            "condition_code": int(row.condition_code),
            "trial_index_in_condition": int(row.trial_index_in_condition),
            "trial_dataset": row.trial_dataset,
            "split": row.split,
            "window_id": row.window_id,
            "start_frame": int(row.start_frame),
            "end_frame": int(row.end_frame),
            "avg_method": row.avg_method,
            "normalization": row.normalization,
            "nc_path": row.nc_path,
        }
        if map_abs.exists() and not overwrite:
            n_skipped += 1
            existing = np.load(map_abs)
            skipped_row = dict(base)
            skipped_row["feature_path"] = map_rel
            skipped_row["feature_layer"] = feature_layer
            skipped_row["feature_shape"] = _feature_shape_str(existing.shape)
            skipped_row["feature_channels"] = int(existing.shape[0])
            skipped_row["feature_height"] = int(existing.shape[1])
            skipped_row["feature_width"] = int(existing.shape[2])
            skipped_row["feature_n_elements"] = int(existing.size)
            rows.append(skipped_row)
            continue

        nc_abs = (repo_root.resolve().parent / row.nc_path).resolve()
        image = _load_image(nc_abs)
        tensor = preprocess_image(
            image,
            input_size=input_size,
            input_channels=input_channels,
            input_scaling=input_scaling,
            imagenet_normalize=imagenet_normalize,
        )
        pending_tensors.append(tensor)
        pending_meta.append(
            {
                "row": base,
                "map_path_abs": map_abs,
                "feature_path": map_rel,
            }
        )
        if len(pending_tensors) >= batch_size:
            flush_batch()

    flush_batch()
    out_df = pd.DataFrame(rows).sort_values("trial_global_id").reset_index(drop=True)
    return out_df, n_written, n_skipped
