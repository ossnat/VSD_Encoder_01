"""Within-animal session camera registration from vasculature.

A **new session** (moving) is a different 2D camera pose of the same cortex
as an **anchor session** (fixed, first: Gandalf ``100718a``). This package
fits a rigid warp from moving pixels onto the anchor grid so later code can
reuse the anchor Schira + ``CorticalAffine``.

Do **not** mix this with landmark Schira fitting in ``src.retinotopy.register``.
Composition with Schira is a later step: moving pixel → ``T`` → anchor pixel
→ ``CorticalAffine.pixel_to_w``.

Pipeline usage (quiet, skip-existing ``transform.yaml``, no QC figures)::

    from src.session_register import load_config, run_registration

    cfg = load_config(
        "experiments/session_register/configs/gandalf_100718a_anchor.yaml",
        moving_session="240718a",
    )
    out = run_registration(cfg)  # save_qc=False, verbose=0
    warped = out["transform"].apply_to_map(moving_frame)
"""

from src.session_register.config import (
    MixedMonkeyError,
    RegistrationBoundsError,
    SessionRegisterConfig,
    check_same_monkey,
    load_config,
)
from src.session_register.fit import (
    fit_transform,
    landmark_id_label,
    match_landmarks,
    match_landmarks_by_id,
    rigid_from_point_pairs,
)
from src.session_register.load import load_session_trial_stack
from src.session_register.pipeline import load_registration, run_registration
from src.session_register.transform import SessionTransform
from src.session_register.vessels import (
    compute_stability_score_map,
    correlation_vessel_map,
    emphasize_vessels,
    extract_vessels,
    stability_ridge_landmarks,
)

__all__ = [
    "MixedMonkeyError",
    "RegistrationBoundsError",
    "SessionRegisterConfig",
    "SessionTransform",
    "check_same_monkey",
    "compute_stability_score_map",
    "correlation_vessel_map",
    "emphasize_vessels",
    "extract_vessels",
    "fit_transform",
    "landmark_id_label",
    "load_config",
    "load_registration",
    "load_session_trial_stack",
    "match_landmarks",
    "match_landmarks_by_id",
    "rigid_from_point_pairs",
    "run_registration",
    "stability_ridge_landmarks",
]
