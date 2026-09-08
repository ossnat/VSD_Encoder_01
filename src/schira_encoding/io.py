"""Shared path / batch helpers for Schira encoding scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.paths import project_root, resolve_data_path
from src.schira_encoding.schema import warped_feature_map_path
from src.schira_encoding.targets import is_anchor_session, warp_target_to_anchor
from src.session_register.transform import SessionTransform


def resolve_output_root(path_str: str, repo: Path | None = None) -> Path:
    """Resolve ``Data/...`` via workspace; otherwise relative to the repo."""
    repo = repo or project_root()
    s = str(path_str)
    if s.startswith("Data/"):
        return resolve_data_path(s, repo)
    p = Path(s)
    if p.is_absolute():
        return p
    return (repo / p).resolve()


def load_target_map(nc_path: Path, spatial_size: tuple[int, int]) -> np.ndarray:
    da = xr.open_dataarray(nc_path)
    image = da.values.astype(np.float32)
    da.close()
    height, width = spatial_size
    if image.shape != (height, width):
        raise ValueError(f"Expected target shape {(height, width)}, got {image.shape}")
    return image


def load_session_transform(
    *,
    session: str,
    anchor_session: str,
    monkey: str,
    register_root: Path | None,
) -> SessionTransform | None:
    if is_anchor_session(session, anchor_session):
        return None
    if register_root is None:
        raise FileNotFoundError(
            f"Session {session} is not the anchor {anchor_session}; "
            "set paths.session_register_root and ensure transform.yaml exists."
        )
    yaml_path = (
        register_root / f"{monkey}__{anchor_session}__{session}" / "transform.yaml"
    )
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"Missing session registration for {session}: {yaml_path}"
        )
    transform, _ = SessionTransform.load_yaml(yaml_path)
    return transform


def stimulus_label(row) -> str:
    """Human-readable stimulus name for plot titles."""
    text = str(getattr(row, "stimulus_text", "") or "").strip()
    if text and text.lower() not in {"nan", "none"}:
        return text
    shape = str(getattr(row, "shape_type", "") or "").strip()
    color = str(getattr(row, "color", "") or "").strip()
    parts = [p for p in (shape, color) if p and p.lower() not in {"nan", "none"}]
    return " ".join(parts) if parts else str(getattr(row, "condition", ""))


def build_schira_xy(
    pairs: pd.DataFrame,
    *,
    repo: Path,
    out_root: Path,
    monkey: str,
    model_slug: str,
    feature_layer: str,
    schira_set: str,
    anchor_session: str,
    spatial_size: tuple[int, int],
    register_root: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack warped features ``(n,C,H,W)`` and anchor-aligned targets ``(n,H,W)``."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    cache: dict[str, SessionTransform | None] = {}
    for row in pairs.itertuples(index=False):
        session = str(row.date)
        condition = str(row.condition)
        feat_path = warped_feature_map_path(
            out_root,
            monkey,
            model_slug=model_slug,
            feature_layer=feature_layer,
            schira_set=schira_set,
            anchor_session=anchor_session,
            h5_session=session,
            condition=condition,
        )
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Missing warped features {feat_path}. "
                "Run scripts/11_warp_features_to_vsd.py first."
            )
        if session not in cache:
            cache[session] = load_session_transform(
                session=session,
                anchor_session=anchor_session,
                monkey=monkey,
                register_root=register_root,
            )
        x = np.load(feat_path)
        y = load_target_map(resolve_data_path(row.nc_path, repo), spatial_size)
        y = warp_target_to_anchor(y, cache[session])
        xs.append(x)
        ys.append(y)
    return np.stack(xs, axis=0), np.stack(ys, axis=0)
