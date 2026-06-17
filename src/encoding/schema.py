"""Path helpers for encoding pair manifests."""

from __future__ import annotations

from pathlib import Path


def encoding_pairs_dir(
    encoding_pairs_root: Path, monkey: str, window_id: str
) -> Path:
    return encoding_pairs_root / monkey / window_id


def encoding_pairs_manifest_path(
    encoding_pairs_root: Path, monkey: str, window_id: str
) -> Path:
    return encoding_pairs_dir(encoding_pairs_root, monkey, window_id) / "manifest.parquet"


def ridge_output_dir(
    ridge_root: Path,
    monkey: str,
    window_id: str,
    model_slug: str,
    feature_layer: str,
) -> Path:
    return ridge_root / monkey / window_id / model_slug / feature_layer
