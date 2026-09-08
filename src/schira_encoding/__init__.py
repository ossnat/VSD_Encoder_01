"""Schira-geometry encoding: LUT, feature warp, local / global-channel ridge.

Pipeline (anchor-centric)::

    stimulus → CNN features (C, Hf, Wf)
    anchor VSD (c, r) → Schira LUT → sample features → (C, 100, 100)
    other sessions' VSD → SessionTransform → anchor grid
    local ridge: y_p = w_p · f_p  (C weights per pixel)
    global channel ridge: y = Σ_c w_c · f_c + b  (shared w over pixels)
"""

from src.schira_encoding.global_channel_model import (
    GlobalChannelRidgeResult,
    fit_global_channel_ridge,
    predict_global_channel_ridge,
)
from src.schira_encoding.lut import SchiraLUT, build_schira_lut
from src.schira_encoding.model import (
    LocalRidgeResult,
    fit_local_ridge,
    predict_local_ridge,
)
from src.schira_encoding.schema import (
    global_channel_ridge_loo_dir,
    local_ridge_loo_dir,
    local_ridge_output_dir,
    lut_path,
    warped_feature_dir,
    warped_feature_map_path,
)
from src.schira_encoding.targets import is_anchor_session, warp_target_to_anchor
from src.schira_encoding.warp_features import warp_feature_map, warp_stimulus_rgb

__all__ = [
    "GlobalChannelRidgeResult",
    "LocalRidgeResult",
    "SchiraLUT",
    "build_schira_lut",
    "fit_global_channel_ridge",
    "fit_local_ridge",
    "global_channel_ridge_loo_dir",
    "is_anchor_session",
    "local_ridge_loo_dir",
    "local_ridge_output_dir",
    "lut_path",
    "predict_global_channel_ridge",
    "predict_local_ridge",
    "warp_feature_map",
    "warp_stimulus_rgb",
    "warp_target_to_anchor",
    "warped_feature_dir",
    "warped_feature_map_path",
]
