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


def stimulus_map_path(
    features_root: Path,
    monkey: str,
    model_name: str,
    feature_layer: str,
    h5_session: str,
    condition: str,
) -> Path:
    return (
        stimulus_feature_dir(features_root, monkey, model_name, feature_layer)
        / "maps"
        / f"{stimulus_key(h5_session, condition)}.npy"
    )
