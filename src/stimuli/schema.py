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


# ---- stimulus library: one image per distinct stimulus (see src/stimuli/identity.stimulus_uid_from_row)
def library_dir(stimuli_root: Path, monkey: str) -> Path:
    return stimuli_root_path(stimuli_root, monkey) / "library"


def library_image_path(stimuli_root: Path, monkey: str, stimulus_uid: str) -> Path:
    return library_dir(stimuli_root, monkey) / f"{stimulus_uid}.png"


def library_table_path(stimuli_root: Path, monkey: str) -> Path:
    """One row per stimulus_uid: shape, size, colour, position, background, notes, image path."""
    return stimuli_root_path(stimuli_root, monkey) / "library.parquet"


def lookup_path(stimuli_root: Path, monkey: str) -> Path:
    """(h5_session, condition) -> stimulus_uid. Several conditions can point to the same stimulus."""
    return stimuli_root_path(stimuli_root, monkey) / "lookup.parquet"
