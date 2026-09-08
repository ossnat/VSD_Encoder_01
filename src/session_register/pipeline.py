"""Register a moving session camera onto an anchor from vessel landmarks.

Pipeline default is quiet and artifact-light: write ``transform.yaml``, skip
work if that file already exists, and do not render QC figures.  Pass
``save_qc=True`` / ``verbose=1`` (or the CLI flags) when you want plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.paths import project_root
from src.session_register.config import SessionRegisterConfig, check_same_monkey
from src.session_register.fit import (
    fit_transform,
    landmark_id_label,
    landmark_rmsd_px,
    match_landmarks_by_id,
    rigid_from_point_pairs,
)
from src.session_register.load import load_session_trial_stack
from src.session_register.transform import SessionTransform, default_center_xy
from src.session_register.vessels import (
    resolve_anchor_landmarks,
    stability_ridge_landmarks,
    vessel_params_dict,
)


def _trough_fit_image(ridge_mask: np.ndarray) -> np.ndarray:
    """Bright 1-px troughs on black — fallback intensity registration."""
    return np.asarray(ridge_mask, dtype=np.float32)


def _log(verbose: int, level: int, msg: str) -> None:
    if int(verbose) >= int(level):
        print(msg, flush=True)


def _result_from_yaml(yaml_path: Path, *, skipped: bool) -> dict[str, Any]:
    transform, raw = SessionTransform.load_yaml(yaml_path)
    out = dict(raw)
    out["transform"] = transform
    out["yaml_path"] = str(yaml_path)
    out["pair_dir"] = str(yaml_path.parent)
    out["skipped"] = bool(skipped)
    return out


def load_registration(
    source: str | Path | SessionRegisterConfig,
    *,
    repo: Path | None = None,
    moving_session: str | None = None,
) -> dict[str, Any]:
    """Load a saved pair ``transform.yaml`` (no H5, no plots).

    ``source`` is a YAML path or a config whose ``pair_dir`` contains the file.
    """
    repo = repo or project_root()
    if isinstance(source, SessionRegisterConfig):
        cfg = source
        if moving_session:
            cfg = cfg.with_updates(moving_session=moving_session)
        yaml_path = cfg.pair_dir(repo) / "transform.yaml"
    else:
        yaml_path = Path(source)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"No registration YAML at {yaml_path}")
    return _result_from_yaml(yaml_path, skipped=True)


def run_registration(
    cfg: SessionRegisterConfig,
    *,
    repo: Path | None = None,
    save_qc: bool | None = None,
    verbose: int | None = None,
    skip_existing: bool = True,
    include_maps: bool = False,
) -> dict[str, Any]:
    """Fit moving → fixed from homologous vessel stations.

    Always writes ``pair_dir/transform.yaml`` (or reuses it).  QC PNGs are
    optional.  Returns a dict with ``transform`` (``SessionTransform``) plus
    scalar metadata — not the VSD maps, unless ``include_maps=True``.
    """
    repo = repo or project_root()
    if not cfg.moving_session:
        raise ValueError("moving_session is required (pass --moving-session)")
    check_same_monkey(cfg.monkey)
    if save_qc is None:
        save_qc = bool(cfg.save_qc)
    if verbose is None:
        verbose = int(cfg.verbose)

    pair_dir = cfg.pair_dir(repo)
    pair_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = pair_dir / "transform.yaml"
    if skip_existing and yaml_path.is_file() and not cfg.overwrite:
        _log(verbose, 1, f"skip existing {yaml_path}")
        return _result_from_yaml(yaml_path, skipped=True)

    fixed_stack, _ = load_session_trial_stack(
        cfg,
        cfg.fixed_session,
        repo=repo,
        start_frame=cfg.vessel_start_frame,
        end_frame=cfg.vessel_end_frame,
    )
    moving_stack, _ = load_session_trial_stack(
        cfg,
        cfg.moving_session,
        repo=repo,
        start_frame=cfg.vessel_start_frame,
        end_frame=cfg.vessel_end_frame,
    )
    fixed_ridge = stability_ridge_landmarks(fixed_stack, cfg=cfg)
    moving_ridge = stability_ridge_landmarks(moving_stack, cfg=cfg)
    fixed_ridge, freeze_msg = resolve_anchor_landmarks(
        fixed_ridge,
        cfg.anchor_landmarks_path(repo),
        cfg.fixed_session,
    )
    _log(verbose, 2, f"anchor landmarks: {freeze_msg}")
    moving_ridge, moving_msg = resolve_anchor_landmarks(
        moving_ridge,
        cfg.session_landmarks_path(cfg.moving_session, repo),
        cfg.moving_session,
        write_if_missing=False,
    )
    _log(verbose, 2, f"moving landmarks: {moving_msg}")

    fixed_fit = _trough_fit_image(fixed_ridge.ridge_mask)
    moving_fit = _trough_fit_image(moving_ridge.ridge_mask)

    matched_fixed, matched_moving, matched_ids = match_landmarks_by_id(
        fixed_ridge.landmarks,
        fixed_ridge.landmark_ids,
        moving_ridge.landmarks,
        moving_ridge.landmark_ids,
    )
    n_matched = int(matched_fixed.shape[0])
    lm_rmsd = float("nan")
    spatial_size = (
        int(fixed_ridge.mean_map.shape[0]),
        int(fixed_ridge.mean_map.shape[1]),
    )
    center_xy = default_center_xy(fixed_ridge.mean_map.shape)
    fit_meta: dict[str, Any] = {}
    if n_matched >= 2:
        transform = rigid_from_point_pairs(
            matched_moving,
            matched_fixed,
            center_xy=center_xy,
            spatial_size=spatial_size,
        )
        lm_rmsd = landmark_rmsd_px(matched_moving, matched_fixed, transform)
        backend = "homologous_landmarks"
        fit_meta = {
            "backend": backend,
            "metric": "landmark_rmsd",
            "metric_value": float(lm_rmsd),
        }
    else:
        transform, fit_meta = fit_transform(fixed_fit, moving_fit, cfg=cfg)
        backend = str(fit_meta.get("backend", "phasecorr_grid"))

    payload: dict[str, Any] = {
        "fixed_session": cfg.fixed_session,
        "moving_session": cfg.moving_session,
        "monkey": cfg.monkey,
        "frame_window": [int(cfg.vessel_start_frame), int(cfg.vessel_end_frame)],
        "n_window_frames": int(cfg.vessel_end_frame) - int(cfg.vessel_start_frame),
        "n_trials_fixed": int(fixed_ridge.n_trials),
        "n_trials_moving": int(moving_ridge.n_trials),
        "normalization": cfg.normalization,
        "vessel": vessel_params_dict(cfg),
        "metric": "landmark_rmsd" if n_matched >= 2 else fit_meta["metric"],
        "metric_value": float(lm_rmsd) if n_matched >= 2 else fit_meta["metric_value"],
        "rmsd_px": float(lm_rmsd) if np.isfinite(lm_rmsd) else fit_meta.get("rmsd_px"),
        "n_landmarks_fixed": int(len(fixed_ridge.landmarks)),
        "n_landmarks_moving": int(len(moving_ridge.landmarks)),
        "n_landmarks_matched": n_matched,
        "landmarks_fixed": np.asarray(fixed_ridge.landmarks).tolist(),
        "landmarks_moving": np.asarray(moving_ridge.landmarks).tolist(),
        "landmark_ids_fixed": np.asarray(fixed_ridge.landmark_ids).tolist(),
        "landmark_ids_moving": np.asarray(moving_ridge.landmark_ids).tolist(),
        "landmark_ids_matched": np.asarray(matched_ids).tolist(),
        "backend": backend,
        "max_rotation_deg": cfg.max_rotation_deg,
        "max_translation_px": cfg.max_translation_px,
        "mask_border_px": cfg.mask_border_px,
        "landmark_match_max_px": cfg.landmark_match_max_px,
    }
    transform.save_yaml(yaml_path, extra=payload)
    _log(verbose, 2, f"Wrote {yaml_path}")

    id_txt = ",".join(
        landmark_id_label(int(a), int(b))
        for a, b in np.asarray(matched_ids).reshape(-1, 2)
    ) if n_matched else ""
    _log(
        verbose,
        1,
        f"{cfg.moving_session} → {cfg.fixed_session}  "
        f"rotation={transform.rotation_deg:.3f} deg  "
        f"translation_xy=[{transform.dx_col:.3f}, {transform.dy_row:.3f}] px  "
        f"matched={n_matched}  ids=[{id_txt}]  backend={backend}  "
        f"landmark_rmsd={lm_rmsd:.3f} px",
    )

    qc_paths: dict[str, str | None] = {
        "qc_path": None,
        "vision_path": None,
        "landmark_qc_path": None,
    }
    warped_raw = None
    if save_qc or include_maps:
        warped_raw = transform.apply_to_map(
            moving_ridge.mean_map, inverse=False, cval=0.0
        )
    if save_qc:
        qc_paths = _write_qc(
            cfg=cfg,
            pair_dir=pair_dir,
            transform=transform,
            payload=payload,
            fixed_ridge=fixed_ridge,
            moving_ridge=moving_ridge,
            fixed_fit=fixed_fit,
            moving_fit=moving_fit,
            warped_raw=warped_raw,
            n_matched=n_matched,
            lm_rmsd=lm_rmsd,
        )
        for key in ("qc_path", "vision_path", "landmark_qc_path"):
            if qc_paths[key]:
                _log(verbose, 2, f"Wrote {qc_paths[key]}")

    result = dict(payload)
    result["transform"] = transform
    result["yaml_path"] = str(yaml_path)
    result["pair_dir"] = str(pair_dir)
    result["skipped"] = False
    result.update(qc_paths)
    if include_maps:
        result["fixed_raw"] = fixed_ridge.mean_map
        result["moving_raw"] = moving_ridge.mean_map
        result["warped_moving"] = np.asarray(warped_raw)
        result["fixed_fit"] = fixed_fit
        result["moving_fit"] = moving_fit
    return result


def _write_qc(
    *,
    cfg: SessionRegisterConfig,
    pair_dir: Path,
    transform: SessionTransform,
    payload: dict[str, Any],
    fixed_ridge: Any,
    moving_ridge: Any,
    fixed_fit: np.ndarray,
    moving_fit: np.ndarray,
    warped_raw: np.ndarray | None,
    n_matched: int,
    lm_rmsd: float,
) -> dict[str, str | None]:
    """Import matplotlib only when QC figures are requested."""
    from src.session_register.plot import (
        plot_landmark_register_qc,
        plot_registration_qc,
        plot_vision_panel,
        plot_vsd_yellow_blue_landmarks,
        save_gray_png,
    )

    save_gray_png(
        fixed_ridge.mean_map,
        pair_dir / f"mean_raw__{cfg.fixed_session}.png",
        title=f"baseline mean {cfg.fixed_session}",
    )
    save_gray_png(
        moving_ridge.mean_map,
        pair_dir / f"mean_raw__{cfg.moving_session}.png",
        title=f"baseline mean {cfg.moving_session}",
    )
    plot_vsd_yellow_blue_landmarks(
        session=cfg.fixed_session,
        mean_raw=fixed_ridge.mean_map,
        score_display=fixed_ridge.score_display,
        chamber=fixed_ridge.chamber_mask,
        landmarks=fixed_ridge.landmarks,
        path=pair_dir / f"landmarks__{cfg.fixed_session}.png",
        ridge_mask=fixed_ridge.ridge_mask,
        landmark_ids=fixed_ridge.landmark_ids,
    )
    plot_vsd_yellow_blue_landmarks(
        session=cfg.moving_session,
        mean_raw=moving_ridge.mean_map,
        score_display=moving_ridge.score_display,
        chamber=moving_ridge.chamber_mask,
        landmarks=moving_ridge.landmarks,
        path=pair_dir / f"landmarks__{cfg.moving_session}.png",
        ridge_mask=moving_ridge.ridge_mask,
        landmark_ids=moving_ridge.landmark_ids,
    )

    warped_fit = transform.apply_to_map(moving_fit, inverse=False, cval=0.0)
    if warped_raw is None:
        warped_raw = transform.apply_to_map(
            moving_ridge.mean_map, inverse=False, cval=0.0
        )
    lms_m = np.asarray(moving_ridge.landmarks, dtype=np.float64).reshape(-1, 2)
    if lms_m.size:
        wr, wc = transform.apply_to_points(lms_m[:, 0], lms_m[:, 1])
        warped_lms = np.stack([wr, wc], axis=1)
    else:
        warped_lms = np.zeros((0, 2))

    fixed_display = 1.0 - fixed_fit
    moving_display = 1.0 - moving_fit
    warped_display = 1.0 - warped_fit
    metric_value = float(payload["metric_value"]) if np.isfinite(payload["metric_value"]) else 0.0
    qc_path = plot_registration_qc(
        fixed_vessel=fixed_display,
        moving_vessel=moving_display,
        warped_moving=warped_display,
        transform=transform,
        metric_name=str(payload["metric"]),
        metric_value=metric_value,
        path=pair_dir / "qc_register.png",
        fixed_session=cfg.fixed_session,
        moving_session=cfg.moving_session,
    )
    vision_path = plot_vision_panel(
        fixed_raw=fixed_ridge.mean_map,
        moving_raw=moving_ridge.mean_map,
        fixed_vessel=fixed_display,
        moving_vessel=moving_display,
        warped_moving=warped_display,
        transform=transform,
        metric_name=str(payload["metric"]),
        metric_value=metric_value,
        path=pair_dir / "qc_vision.png",
        fixed_session=cfg.fixed_session,
        moving_session=cfg.moving_session,
    )
    lm_qc = plot_landmark_register_qc(
        fixed_raw=fixed_ridge.mean_map,
        moving_raw=moving_ridge.mean_map,
        warped_moving=warped_raw,
        fixed_landmarks=fixed_ridge.landmarks,
        moving_landmarks=moving_ridge.landmarks,
        warped_moving_landmarks=warped_lms,
        transform=transform,
        path=pair_dir / "qc_landmarks.png",
        fixed_session=cfg.fixed_session,
        moving_session=cfg.moving_session,
        n_matched=n_matched,
        landmark_rmsd=float(lm_rmsd) if np.isfinite(lm_rmsd) else float("nan"),
        fixed_ids=fixed_ridge.landmark_ids,
        moving_ids=moving_ridge.landmark_ids,
    )
    return {
        "qc_path": str(qc_path),
        "vision_path": str(vision_path),
        "landmark_qc_path": str(lm_qc),
    }
