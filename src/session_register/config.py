"""Dataclass config for session-camera vasculature registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.paths import project_root


class MixedMonkeyError(ValueError):
    """Raised when a pair mixes animals. This module is within-animal only."""


class RegistrationBoundsError(RuntimeError):
    """Raised when the rigid search lands on the configured rotation/translation wall."""


def check_same_monkey(*monkeys: str | None) -> str:
    """Return the single monkey name, or raise ``MixedMonkeyError``."""
    names = {str(m).strip().lower() for m in monkeys if m is not None and str(m).strip()}
    if not names:
        raise MixedMonkeyError("monkey is required")
    if len(names) != 1:
        raise MixedMonkeyError(
            "session_register is within-animal only; "
            f"got monkeys={sorted(names)}"
        )
    return names.pop()


def _as_float_pair(value: Any, name: str) -> tuple[float, float]:
    if value is None:
        raise ValueError(f"{name} is required")
    seq = list(value)
    if len(seq) != 2:
        raise ValueError(f"{name} must be a length-2 sequence, got {value!r}")
    return float(seq[0]), float(seq[1])


def _as_int_pair(value: Any, name: str) -> tuple[int, int]:
    a, b = _as_float_pair(value, name)
    return int(a), int(b)


def _optional_float_pair(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    return _as_float_pair(value, name)


def _conditions(value: Any) -> str | tuple[str, ...]:
    if value is None or value == "all":
        return "all"
    if isinstance(value, str):
        return value
    return tuple(str(x) for x in value)


def _n_trials(value: Any) -> int | None:
    """``None`` means all trials. Integers are a cap."""
    if value is None or value == "all":
        return None
    return int(value)


@dataclass(frozen=True)
class SessionRegisterConfig:
    """Parameters for building vessel maps and fitting a 2D camera warp.

    Frame window ``[start_frame, end_frame)`` is half-open (Python slice),
    matching ``configs/windows/*.yaml``. Default ``35, 40`` is five frames.
    """

    monkey: str = "gandalf"
    fixed_session: str = "100718a"
    moving_session: str | None = None
    start_frame: int = 35
    end_frame: int = 40
    spatial_size: tuple[int, int] = (100, 100)
    normalization: str = "none"
    n_trials: int | None = None
    seed: int = 17
    conditions: str | tuple[str, ...] = "all"
    invert_intensity: bool = False
    vessel_method: str = "meijering"
    clip_percentiles: tuple[float, float] = (1.0, 99.0)
    hard_clip_percentiles: tuple[float, float] | None = None
    tophat_radius_px: float = 6.0
    gaussian_sigma: float = 0.6
    vessel_percentile: float = 75.0
    background_sigma: float = 8.0
    n_vessel_objects: int = 2
    n_landmarks: int = 6
    vessel_start_frame: int = 5
    vessel_end_frame: int = 25
    landmark_match_max_px: float = 12.0
    hysteresis_percentiles: tuple[float, float] = (74.0, 91.0)
    chamber_rim_px: int = 12
    roi_radius_px: int = 40
    vessel_trace_px: int = 1
    transform_type: str = "rigid"
    metric: str = "ncc"
    mask_border_px: int = 4
    max_rotation_deg: float = 25.0
    max_translation_px: float = 30.0
    output_dir: str = "experiments/session_register/runs"
    verbose: int = 0
    save_qc: bool = False
    split_csv: str = (
        "Data/FoundationData/ProcessedData/splits/"
        "split_v3_seed17_session_condition_group_gandalf.csv"
    )
    trials_index_csv: str | None = (
        "Data/FoundationData/ProcessedData/splits/all_trials_index_gandalf.csv"
    )
    avg_method: str = "mean"
    overwrite: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"Frame window must be half-open with end > start; "
                f"got [{self.start_frame}, {self.end_frame})"
            )
        if self.mask_border_px < 0:
            raise ValueError("mask_border_px must be >= 0")
        if int(self.n_landmarks) < 2:
            raise ValueError("n_landmarks must be >= 2")
        if int(self.vessel_end_frame) <= int(self.vessel_start_frame):
            raise ValueError(
                "vessel frame window must have end > start; "
                f"got [{self.vessel_start_frame}, {self.vessel_end_frame})"
            )
        if int(self.verbose) < 0:
            raise ValueError("verbose must be >= 0")
        t = str(self.transform_type).strip().lower()
        if t not in ("translation", "rigid", "similarity"):
            raise ValueError(
                f"transform_type must be translation|rigid|similarity, got {self.transform_type!r}"
            )
        m = str(self.metric).strip().lower()
        if m not in ("ncc", "mattes_mi", "mi"):
            raise ValueError(f"metric must be ncc or mattes_mi, got {self.metric!r}")

    @property
    def n_window_frames(self) -> int:
        """Number of frames in ``[start_frame, end_frame)``."""
        return int(self.end_frame) - int(self.start_frame)

    @property
    def frame_window(self) -> list[int]:
        return [int(self.start_frame), int(self.end_frame)]

    @property
    def metric_name(self) -> str:
        key = str(self.metric).strip().lower()
        if key in ("mattes_mi", "mi"):
            return "mattes_mi"
        return "ncc"

    @property
    def transform_type_name(self) -> str:
        return str(self.transform_type).strip().lower()

    def resolved_output_dir(self, repo: Path | None = None) -> Path:
        from src.paths import resolve_data_path

        root = repo or project_root()
        p = Path(self.output_dir)
        if p.is_absolute():
            return p
        # Portable Data/... paths resolve to the workspace sibling Data tree.
        if str(self.output_dir).startswith("Data/"):
            return resolve_data_path(self.output_dir, root)
        return (root / p).resolve()

    def pair_dir(self, repo: Path | None = None) -> Path:
        if not self.moving_session:
            raise ValueError("moving_session is required")
        return (
            self.resolved_output_dir(repo)
            / f"{self.monkey}__{self.fixed_session}__{self.moving_session}"
        )

    def anchor_dir(self, repo: Path | None = None) -> Path:
        return self.resolved_output_dir(repo) / f"{self.monkey}__{self.fixed_session}"

    def anchor_landmarks_path(self, repo: Path | None = None) -> Path:
        return self.anchor_dir(repo) / "anchor_landmarks.yaml"

    def session_dir(self, session: str, repo: Path | None = None) -> Path:
        return self.resolved_output_dir(repo) / f"{self.monkey}__{session}"

    def session_landmarks_path(self, session: str, repo: Path | None = None) -> Path:
        return self.session_dir(session, repo) / "landmarks.yaml"

    def cache_dir(self, repo: Path | None = None) -> Path:
        return self.resolved_output_dir(repo) / "cache"

    def with_updates(self, **kwargs: Any) -> SessionRegisterConfig:
        data = self.to_mapping()
        data.update(kwargs)
        return SessionRegisterConfig.from_mapping(data)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "monkey": self.monkey,
            "fixed_session": self.fixed_session,
            "moving_session": self.moving_session,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "spatial_size": list(self.spatial_size),
            "normalization": self.normalization,
            "n_trials": "all" if self.n_trials is None else self.n_trials,
            "seed": self.seed,
            "conditions": (
                "all"
                if self.conditions == "all"
                else list(self.conditions)
                if isinstance(self.conditions, tuple)
                else self.conditions
            ),
            "invert_intensity": self.invert_intensity,
            "vessel_method": self.vessel_method,
            "clip_percentiles": list(self.clip_percentiles),
            "hard_clip_percentiles": (
                None
                if self.hard_clip_percentiles is None
                else list(self.hard_clip_percentiles)
            ),
            "tophat_radius_px": self.tophat_radius_px,
            "gaussian_sigma": self.gaussian_sigma,
            "vessel_percentile": self.vessel_percentile,
            "background_sigma": self.background_sigma,
            "n_vessel_objects": self.n_vessel_objects,
            "n_landmarks": self.n_landmarks,
            "vessel_start_frame": self.vessel_start_frame,
            "vessel_end_frame": self.vessel_end_frame,
            "landmark_match_max_px": self.landmark_match_max_px,
            "hysteresis_percentiles": list(self.hysteresis_percentiles),
            "chamber_rim_px": self.chamber_rim_px,
            "roi_radius_px": self.roi_radius_px,
            "vessel_trace_px": self.vessel_trace_px,
            "transform_type": self.transform_type_name,
            "metric": self.metric_name,
            "mask_border_px": self.mask_border_px,
            "max_rotation_deg": self.max_rotation_deg,
            "max_translation_px": self.max_translation_px,
            "output_dir": self.output_dir,
            "verbose": int(self.verbose),
            "save_qc": bool(self.save_qc),
            "split_csv": self.split_csv,
            "trials_index_csv": self.trials_index_csv,
            "avg_method": self.avg_method,
            "overwrite": self.overwrite,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> SessionRegisterConfig:
        if not data:
            return cls()
        spatial = data.get("spatial_size", (100, 100))
        return cls(
            monkey=str(data.get("monkey", "gandalf")),
            fixed_session=str(data.get("fixed_session", "100718a")),
            moving_session=(
                None
                if data.get("moving_session") in (None, "")
                else str(data.get("moving_session"))
            ),
            start_frame=int(data.get("start_frame", 35)),
            end_frame=int(data.get("end_frame", 40)),
            spatial_size=_as_int_pair(spatial, "spatial_size"),
            normalization=str(data.get("normalization", "none")),
            n_trials=_n_trials(data.get("n_trials", "all")),
            seed=int(data.get("seed", 17)),
            conditions=_conditions(data.get("conditions", "all")),
            invert_intensity=bool(data.get("invert_intensity", False)),
            vessel_method=str(data.get("vessel_method", "meijering")),
            clip_percentiles=_as_float_pair(
                data.get("clip_percentiles", (1.0, 99.0)), "clip_percentiles"
            ),
            hard_clip_percentiles=_optional_float_pair(
                data.get("hard_clip_percentiles"), "hard_clip_percentiles"
            ),
            tophat_radius_px=float(data.get("tophat_radius_px", 6.0)),
            gaussian_sigma=float(data.get("gaussian_sigma", 0.6)),
            vessel_percentile=float(data.get("vessel_percentile", 75.0)),
            background_sigma=float(data.get("background_sigma", 8.0)),
            n_vessel_objects=int(data.get("n_vessel_objects", 2)),
            n_landmarks=int(data.get("n_landmarks", 6)),
            vessel_start_frame=int(data.get("vessel_start_frame", 5)),
            vessel_end_frame=int(data.get("vessel_end_frame", 25)),
            landmark_match_max_px=float(data.get("landmark_match_max_px", 12.0)),
            hysteresis_percentiles=_as_float_pair(
                data.get("hysteresis_percentiles", (74.0, 91.0)),
                "hysteresis_percentiles",
            ),
            chamber_rim_px=int(data.get("chamber_rim_px", 12)),
            roi_radius_px=int(data.get("roi_radius_px", 40)),
            vessel_trace_px=int(data.get("vessel_trace_px", 1)),
            transform_type=str(data.get("transform_type", "rigid")),
            metric=str(data.get("metric", "ncc")),
            mask_border_px=int(data.get("mask_border_px", 4)),
            max_rotation_deg=float(data.get("max_rotation_deg", 25.0)),
            max_translation_px=float(data.get("max_translation_px", 30.0)),
            output_dir=str(
                data.get("output_dir", "experiments/session_register/runs")
            ),
            verbose=int(data.get("verbose", 0)),
            save_qc=bool(data.get("save_qc", False)),
            split_csv=str(
                data.get(
                    "split_csv",
                    "Data/FoundationData/ProcessedData/splits/"
                    "split_v3_seed17_session_condition_group_gandalf.csv",
                )
            ),
            trials_index_csv=(
                None
                if data.get("trials_index_csv") in (None, "")
                else str(data.get("trials_index_csv"))
            ),
            avg_method=str(data.get("avg_method", "mean")),
            overwrite=bool(data.get("overwrite", False)),
            extra={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "monkey",
                    "fixed_session",
                    "moving_session",
                    "start_frame",
                    "end_frame",
                    "spatial_size",
                    "normalization",
                    "n_trials",
                    "seed",
                    "conditions",
                    "invert_intensity",
                    "vessel_method",
                    "clip_percentiles",
                    "hard_clip_percentiles",
                    "tophat_radius_px",
                    "gaussian_sigma",
                    "vessel_percentile",
                    "background_sigma",
                    "n_vessel_objects",
                    "n_landmarks",
                    "vessel_start_frame",
                    "vessel_end_frame",
                    "landmark_match_max_px",
                    "hysteresis_percentiles",
                    "chamber_rim_px",
                    "roi_radius_px",
                    "vessel_trace_px",
                    "transform_type",
                    "metric",
                    "mask_border_px",
                    "max_rotation_deg",
                    "max_translation_px",
                    "output_dir",
                    "verbose",
                    "save_qc",
                    "split_csv",
                    "trials_index_csv",
                    "avg_method",
                    "overwrite",
                    "paths",
                }
            },
        )


def load_config(
    path: str | Path,
    *,
    moving_session: str | None = None,
    repo: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> SessionRegisterConfig:
    """Load YAML, filling split/spatial defaults from ``configs/default.yaml``."""
    repo = repo or project_root()
    data: dict[str, Any] = {}
    default_path = repo / "configs/default.yaml"
    if default_path.is_file():
        loaded = yaml.safe_load(default_path.read_text()) or {}
        if isinstance(loaded, dict):
            data.update(loaded)
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {cfg_path} must be a mapping")
    data.update(raw)
    if moving_session:
        data["moving_session"] = moving_session
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return SessionRegisterConfig.from_mapping(data)
