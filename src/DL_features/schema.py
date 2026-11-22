"""Output schema helpers for extracted feature maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def model_slug(model_cfg_or_name: dict[str, Any] | str, pretrained: bool = True) -> str:
    """
    Stable directory slug for a model config or legacy ResNet name.

    Examples: ``resnet18_imagenet``, ``gabor_serre_gwp``.
    """
    if isinstance(model_cfg_or_name, dict):
        cfg = model_cfg_or_name
        name = str(cfg["name"]).lower()
        backbone_type = cfg.get("type", "resnet")
        if backbone_type == "resnet":
            pt = bool(cfg.get("pretrained", True))
            return f"{name}_{'imagenet' if pt else 'random'}"
        if backbone_type == "vgg":
            pt = bool(cfg.get("pretrained", True))
            return f"{name}_{'imagenet' if pt else 'random'}"
        if backbone_type == "cornet":
            pt = bool(cfg.get("pretrained", True))
            return f"{name}_{'imagenet' if pt else 'random'}"
        if backbone_type == "gabor_gwp":
            base = f"{name}_gwp"
            variant = cfg.get("variant")
            if variant:
                return f"{base}_{variant}"
            return base
        return name

    name = str(model_cfg_or_name).lower()
    return f"{name}_{'imagenet' if pretrained else 'random'}"


def feature_dir(
    features_root: Path,
    monkey: str,
    window_id: str,
    model_name: str,
    feature_layer: str,
) -> Path:
    return features_root / monkey / window_id / model_name / feature_layer


def map_path(
    features_root: Path,
    monkey: str,
    window_id: str,
    model_name: str,
    feature_layer: str,
    trial_global_id: int,
) -> Path:
    return (
        feature_dir(features_root, monkey, window_id, model_name, feature_layer)
        / "maps"
        / f"{trial_global_id:06d}.npy"
    )


def stimulus_key(h5_session: str, condition: str) -> str:
    return f"{h5_session}__{condition}"


def stimulus_feature_dir(
    features_root: Path,
    monkey: str,
    model_name: str,
    feature_layer: str,
) -> Path:
    return features_root / monkey / model_name / feature_layer


# Stimulus feature maps are stored once per library image (stimulus_uid):
#   <features_root>/<monkey>/<model>/<layer>/maps_by_uid/<stimulus_uid>.npy
# (session, condition) is resolved to its uid through the stimulus lookup table
#   <features_root>/../stimuli/<monkey>/lookup.parquet   (written by scripts/01b_build_stimulus_images.py)
UID_MAPS_DIRNAME = "maps_by_uid"
UID_FEATURE_MANIFEST = "manifest_by_uid.parquet"


def stimulus_lookup_file(features_root: Path, monkey: str) -> Path:
    return Path(features_root).parent / "stimuli" / monkey / "lookup.parquet"


_LOOKUP_CACHE: dict[tuple[str, float], dict[tuple[str, str], str]] = {}


def stimulus_uid_for(features_root: Path, monkey: str, h5_session: str, condition: str) -> str:
    """(session, condition) -> stimulus_uid via the stimulus lookup table (cached; reloaded if the file changes)."""
    import pandas as pd

    path = stimulus_lookup_file(features_root, monkey)
    if not path.is_file():
        raise FileNotFoundError(
            f"Stimulus lookup table not found: {path}. Run scripts/01b_build_stimulus_images.py first."
        )
    key = (str(path), path.stat().st_mtime)
    table = _LOOKUP_CACHE.get(key)
    if table is None:
        df = pd.read_parquet(path)
        table = {(str(s), str(c)): str(u) for s, c, u in zip(df.h5_session, df.condition, df.stimulus_uid)}
        _LOOKUP_CACHE.clear()
        _LOOKUP_CACHE[key] = table
    try:
        return table[(str(h5_session), str(condition))]
    except KeyError:
        raise KeyError(
            f"No stimulus for session {h5_session!r} condition {condition!r} in {path}"
        ) from None


def stimulus_uid_map_path(
    features_root: Path, monkey: str, model_name: str, feature_layer: str, stimulus_uid: str
) -> Path:
    return stimulus_feature_dir(features_root, monkey, model_name, feature_layer) / UID_MAPS_DIRNAME / f"{stimulus_uid}.npy"


def stimulus_map_path(
    features_root: Path,
    monkey: str,
    model_name: str,
    feature_layer: str,
    h5_session: str,
    condition: str,
) -> Path:
    """Feature map of the stimulus shown in (session, condition): one shared file per stimulus_uid."""
    uid = stimulus_uid_for(features_root, monkey, h5_session, condition)
    return stimulus_uid_map_path(features_root, monkey, model_name, feature_layer, uid)


def legacy_stimulus_map_path(
    features_root: Path,
    monkey: str,
    model_name: str,
    feature_layer: str,
    h5_session: str,
    condition: str,
) -> Path:
    """Old per-(session, condition) location (maps/<session>__<condition>.npy), kept for comparisons only."""
    return (
        stimulus_feature_dir(features_root, monkey, model_name, feature_layer)
        / "maps"
        / f"{stimulus_key(h5_session, condition)}.npy"
    )
