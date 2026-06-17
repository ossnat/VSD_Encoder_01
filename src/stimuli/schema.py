"""Path helpers for rendered stimulus images."""

from __future__ import annotations

from pathlib import Path


def stimuli_root_path(stimuli_root: Path, monkey: str) -> Path:
    return stimuli_root / monkey


def session_images_dir(stimuli_root: Path, monkey: str, h5_session: str) -> Path:
    return stimuli_root_path(stimuli_root, monkey) / "images" / h5_session


def stimulus_image_path(
    stimuli_root: Path, monkey: str, h5_session: str, condition: str
) -> Path:
    return session_images_dir(stimuli_root, monkey, h5_session) / f"{condition}.png"


def manifest_path(stimuli_root: Path, monkey: str) -> Path:
    return stimuli_root_path(stimuli_root, monkey) / "manifest.parquet"


def parsed_catalog_path(stimuli_root: Path, monkey: str) -> Path:
    return stimuli_root_path(stimuli_root, monkey) / "parsed" / "conditions.parquet"
