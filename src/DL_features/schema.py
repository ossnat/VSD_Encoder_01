"""Output schema helpers for extracted feature maps."""

from __future__ import annotations

from pathlib import Path


def model_slug(name: str, pretrained: bool) -> str:
    return f"{name.lower()}_{'imagenet' if pretrained else 'random'}"


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
