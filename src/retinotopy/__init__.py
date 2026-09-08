"""Retinotopic (Schira / Schwartz) maps between visual field and cortex."""

from src.retinotopy.affine import CorticalAffine
from src.retinotopy.params import load_schira_set, load_session_retinotopy
from src.retinotopy.schira import (
    SHEAR_AYZENSHTAT,
    SHEAR_CONSTANT,
    SHEAR_DOUBLE_SECH,
    SchiraParams,
    cartesian_to_polar,
    compressed_polar,
    cortical_uv_to_visual_deg,
    forward_schira,
    inverse_schira,
    polar_to_cartesian,
    shear_fa,
)
from src.retinotopy.visual_field import (
    cartesian_deg_to_image_px,
    cartesian_grid_line_degrees,
    image_px_to_cartesian_deg,
    image_px_to_polar,
    render_cartesian_degree_grid,
    render_polar_degree_grid,
    stimulus_ink_mask,
    upsample_render_canvas,
)
from src.retinotopy.warp import (
    forward_ink_cloud_w,
    sample_stimulus_onto_cortical_w,
    sample_stimulus_onto_vsd_grid,
)

__all__ = [
    "CorticalAffine",
    "SHEAR_AYZENSHTAT",
    "SHEAR_CONSTANT",
    "SHEAR_DOUBLE_SECH",
    "SchiraParams",
    "cartesian_deg_to_image_px",
    "cartesian_grid_line_degrees",
    "cartesian_to_polar",
    "compressed_polar",
    "cortical_uv_to_visual_deg",
    "forward_ink_cloud_w",
    "forward_schira",
    "image_px_to_cartesian_deg",
    "image_px_to_polar",
    "inverse_schira",
    "load_schira_set",
    "load_session_retinotopy",
    "polar_to_cartesian",
    "render_cartesian_degree_grid",
    "render_polar_degree_grid",
    "sample_stimulus_onto_cortical_w",
    "sample_stimulus_onto_vsd_grid",
    "shear_fa",
    "stimulus_ink_mask",
    "upsample_render_canvas",
]
