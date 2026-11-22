"""Feature-map extraction from rendered stimulus images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.DL_features.preprocess import preprocess_stimulus
from src.paths import resolve_data_path


def _feature_shape_str(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(int(x)) for x in shape) + ")"


def plan_stimulus_extraction(manifest: pd.DataFrame, map_path_fn) -> tuple[pd.DataFrame, dict]:
    """Split the (session, condition) rows into rows to extract (first row per output file) and aliases.

    Returns (primary_rows, aliases) where aliases maps an output path to the list of extra rows that share it.
    Pure function (no torch) so the de-duplication can be tested anywhere.
    """
    rows = manifest.drop_duplicates(["h5_session", "condition"], keep="first").sort_values(["h5_session", "condition_num"])
    primary_idx, aliases, seen = [], {}, {}
    for idx, row in zip(rows.index, rows.itertuples(index=False)):
        path = Path(map_path_fn(str(row.h5_session), str(row.condition)))
        if path in seen:
            aliases.setdefault(path, []).append(idx)
        else:
            seen[path] = idx
            primary_idx.append(idx)
    return rows.loc[primary_idx], {p: rows.loc[ix] for p, ix in aliases.items()}


def extract_stimulus_features(
    model: torch.nn.Module,
    manifest: pd.DataFrame,
    *,
    repo_root: Path,
    map_path_fn,
    feature_layer: str,
    model_cfg: dict,
    input_size: int,
    imagenet_normalize: bool,
    batch_size: int,
    device: torch.device,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    """
    Extract and save (C, H, W) maps per unique stimulus (h5_session, condition).

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

    # One forward pass per output file: conditions that show the same stimulus share one feature map.
    stimulus_rows, alias_rows = plan_stimulus_extraction(manifest, map_path_fn)

    for row in stimulus_rows.itertuples(index=False):
        map_abs = Path(
            map_path_fn(str(row.h5_session), str(row.condition))
        )
        ws = repo_root.resolve().parent
        try:
            feature_path = str(map_abs.resolve().relative_to(ws))
        except ValueError:
            feature_path = str(map_abs.resolve())

        base = {
            "monkey": row.monkey,
            "h5_session": row.h5_session,
            "condition": row.condition,
            "condition_num": int(row.condition_num),
            "image_path": row.image_path,
            "stimulus_text": row.stimulus_text,
            "shape_type": row.shape_type,
            "color": row.color,
            "is_blank": bool(row.is_blank),
            "stimulus_uid": getattr(row, "stimulus_uid", None),
        }
        if map_abs.exists() and not overwrite:
            n_skipped += 1
            existing = np.load(map_abs)
            skipped_row = dict(base)
            skipped_row["feature_path"] = feature_path
            skipped_row["feature_layer"] = feature_layer
            skipped_row["feature_shape"] = _feature_shape_str(existing.shape)
            skipped_row["feature_channels"] = int(existing.shape[0])
            skipped_row["feature_height"] = int(existing.shape[1])
            skipped_row["feature_width"] = int(existing.shape[2])
            skipped_row["feature_n_elements"] = int(existing.size)
            rows.append(skipped_row)
            continue

        image_path = resolve_data_path(row.image_path, repo_root)
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        tensor = preprocess_stimulus(
            image,
            model_cfg=model_cfg,
            input_size=input_size,
            imagenet_normalize=imagenet_normalize,
        )
        pending_tensors.append(tensor)
        pending_meta.append(
            {
                "row": base,
                "map_path_abs": map_abs,
                "feature_path": feature_path,
            }
        )
        if len(pending_tensors) >= batch_size:
            flush_batch()

    flush_batch()
    # Alias rows: same feature file and shape as their primary row, own session / condition.
    by_path = {r["feature_path"]: r for r in rows}
    ws = repo_root.resolve().parent
    for path, extra in alias_rows.items():
        try:
            fp = str(Path(path).resolve().relative_to(ws))
        except ValueError:
            fp = str(Path(path).resolve())
        for row in extra.itertuples(index=False):
            r = dict(by_path[fp])
            r.update(h5_session=row.h5_session, condition=row.condition, condition_num=int(row.condition_num),
                     image_path=row.image_path, stimulus_text=row.stimulus_text, is_blank=bool(row.is_blank))
            rows.append(r)
    out_df = (
        pd.DataFrame(rows)
        .sort_values(["h5_session", "condition_num"])
        .reset_index(drop=True)
    )
    return out_df, n_written, n_skipped
