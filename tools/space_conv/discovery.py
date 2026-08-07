"""Discover monkeys / sessions / conditions / trials from ProcessedData H5 files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import pandas as pd

from src.paths import project_root, resolve_data_path, workspace_root
from src.stimuli.schema import manifest_path, parsed_catalog_path

PROCESSED_REL = Path("Data/FoundationData/ProcessedData")
DEFAULT_STIMULI_ROOT = Path("Data/VSD_Encoder_01/stimuli")
_COND_TEXT_RE = re.compile(r"(?i)^cond\s*\d+\s*:\s*(.*)$")


@dataclass(frozen=True)
class TrialRef:
    trial_global_id: int
    trial_index_in_condition: int
    dataset_name: str
    condition: str
    date: str
    monkey: str


def processed_monkey_dir(monkey: str, *, repo: Path | None = None) -> Path:
    root = repo or project_root()
    return resolve_data_path(PROCESSED_REL / monkey, root)


def _is_blank_session_h5(path: Path) -> bool:
    """True for blank-trial extracts like ``session_*_blank.h5`` / ``*_blank.h5``."""
    return path.stem.endswith("_blank")


def list_monkeys(*, repo: Path | None = None) -> list[str]:
    """Monkey folders under ProcessedData that contain non-blank session_*.h5 files."""
    root = repo or project_root()
    base = resolve_data_path(PROCESSED_REL, root)
    if not base.is_dir():
        return []
    monkeys: list[str] = []
    for path in sorted(base.iterdir()):
        if path.is_dir() and any(
            not _is_blank_session_h5(p) for p in path.glob("session_*.h5")
        ):
            monkeys.append(path.name)
    return monkeys


def list_session_h5_files(monkey: str, *, repo: Path | None = None) -> list[Path]:
    """Sorted session H5 paths for a monkey (excludes ``*_blank.h5``)."""
    monkey_dir = processed_monkey_dir(monkey, repo=repo)
    if not monkey_dir.is_dir():
        return []
    return sorted(
        p for p in monkey_dir.glob("session_*.h5") if not _is_blank_session_h5(p)
    )


def session_date_from_h5(h5_path: Path) -> str:
    """Parse date tag from ``session_{date}_condsAN.h5`` (or similar)."""
    stem = h5_path.stem  # e.g. session_270618b_condsAN
    if stem.startswith("session_"):
        rest = stem[len("session_") :]
        # Drop trailing _condsAN / _conds* if present
        parts = rest.split("_")
        return parts[0]
    return stem


def resolve_session_h5_by_date(
    monkey: str,
    date: str,
    *,
    repo: Path | None = None,
) -> Path | None:
    """
    Resolve a session H5 by date tag (e.g. ``270618b``).

    Prefers ``*_condsAN.h5`` when more than one non-blank file matches the date.
    """
    matches = [
        p for p in list_session_h5_files(monkey, repo=repo) if session_date_from_h5(p) == date
    ]
    if not matches:
        return None
    preferred = [p for p in matches if p.stem.endswith("_condsAN")]
    return preferred[0] if preferred else matches[0]


def _read_trial_metadata(h5_path: Path) -> list[dict]:
    with h5py.File(h5_path, "r") as f:
        if "trial_metadata_json" not in f.attrs:
            raise KeyError(f"No trial_metadata_json in {h5_path}")
        return list(json.loads(f.attrs["trial_metadata_json"]))


def list_conditions(h5_path: Path) -> list[str]:
    """Unique condition labels in file order of first appearance."""
    meta = _read_trial_metadata(h5_path)
    seen: list[str] = []
    for entry in meta:
        cond = str(entry["condition"])
        if cond not in seen:
            seen.append(cond)
    return seen


def list_trials_for_condition(h5_path: Path, condition: str) -> list[TrialRef]:
    """Trials belonging to ``condition``, ordered by metadata index."""
    meta = _read_trial_metadata(h5_path)
    date = session_date_from_h5(h5_path)
    out: list[TrialRef] = []
    for idx, entry in enumerate(meta):
        if str(entry["condition"]) != condition:
            continue
        out.append(
            TrialRef(
                trial_global_id=int(entry["trial_global_id"]),
                trial_index_in_condition=int(entry.get("trial_index_in_condition", len(out))),
                dataset_name=f"trial_{idx:06d}",
                condition=condition,
                date=str(entry.get("date", date)),
                monkey=str(entry.get("monkey", "")),
            )
        )
    return out


def default_split_paths(monkey: str) -> tuple[str, str]:
    """Portable CSV paths used elsewhere for this monkey (optional fallback)."""
    split = (
        f"Data/FoundationData/ProcessedData/splits/"
        f"split_v3_seed17_session_condition_group_{monkey}.csv"
    )
    index = f"Data/FoundationData/ProcessedData/splits/all_trials_index_{monkey}.csv"
    return split, index


def describe_workspace(*, repo: Path | None = None) -> str:
    root = repo or project_root()
    return (
        f"project_root={root}\n"
        f"workspace_root={workspace_root(root)}\n"
        f"ProcessedData={resolve_data_path(PROCESSED_REL, root)}"
    )


def _stimuli_root(*, repo: Path | None = None) -> Path:
    return resolve_data_path(DEFAULT_STIMULI_ROOT, repo or project_root())


@lru_cache(maxsize=8)
def _load_stimulus_catalog_df(monkey: str, stimuli_root_str: str) -> pd.DataFrame | None:
    """Load parsed conditions.parquet (preferred) or manifest.parquet for a monkey."""
    stimuli_root = Path(stimuli_root_str)
    for path in (
        parsed_catalog_path(stimuli_root, monkey),
        manifest_path(stimuli_root, monkey),
    ):
        if path.is_file():
            df = pd.read_parquet(path)
            if {"h5_session", "condition", "stimulus_text"}.issubset(df.columns):
                return df
    return None


def format_stimulus_description(
    stimulus_text: str,
    *,
    shape_type: str | None = None,
    color: str | None = None,
    size_deg: float | None = None,
    letter: str | None = None,
) -> str:
    """
    Short human label for prompts/titles.

    Prefer cleaned catalog ``stimulus_text`` (strip leading ``condN:``).
    Fall back to color / shape / size fields when text is empty.
    """
    text = str(stimulus_text or "").strip()
    match = _COND_TEXT_RE.match(text)
    if match:
        text = match.group(1).strip()
    if text:
        return text
    shape = str(shape_type or "").strip()
    if not shape or shape == "blank":
        return "blank"
    col = str(color or "").strip()
    parts: list[str] = []
    if col and col.lower() not in {"", "none", "nan"}:
        parts.append(col)
    if shape == "letter" and letter and str(letter).lower() != "nan":
        parts.append(f"letter {str(letter).upper()}")
    else:
        parts.append(shape.replace("_", " "))
    if size_deg is not None and not (isinstance(size_deg, float) and pd.isna(size_deg)):
        parts.append(f"{float(size_deg):g}")
    return " ".join(parts) if parts else "unknown"


def stimulus_description_for_condition(
    monkey: str,
    h5_session: str,
    condition: str,
    *,
    repo: Path | None = None,
) -> str | None:
    """
    Resolve stimulus description for ``(monkey, h5_session, condition)``.

    Uses the same parsed stimulus catalog / manifest as the rest of the project
    (``Data/VSD_Encoder_01/stimuli/<monkey>/…``).
    """
    root = repo or project_root()
    df = _load_stimulus_catalog_df(monkey, str(_stimuli_root(repo=root)))
    if df is None:
        return None
    hits = df[(df["h5_session"] == h5_session) & (df["condition"] == condition)]
    if hits.empty:
        return None
    row = hits.iloc[0]
    letter = row["letter"] if "letter" in hits.columns else None
    size = row["size_deg"] if "size_deg" in hits.columns else None
    return format_stimulus_description(
        str(row["stimulus_text"]),
        shape_type=str(row["shape_type"]) if "shape_type" in hits.columns else None,
        color=str(row["color"]) if "color" in hits.columns else None,
        size_deg=None if size is None or (isinstance(size, float) and pd.isna(size)) else float(size),
        letter=None if letter is None or (isinstance(letter, float) and pd.isna(letter)) else str(letter),
    )


def condition_labels_with_stimuli(
    h5_path: Path,
    conditions: list[str],
    *,
    monkey: str,
    repo: Path | None = None,
) -> list[str]:
    """Labels like ``condAN1 — black point 0.1 diameter`` for selection prompts."""
    date = session_date_from_h5(h5_path)
    labels: list[str] = []
    for cond in conditions:
        desc = stimulus_description_for_condition(monkey, date, cond, repo=repo)
        labels.append(f"{cond} — {desc}" if desc else cond)
    return labels
