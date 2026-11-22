"""Retinotopic maps between visual field and cortex.

- ``schira``: the Schira 2007 monopole with shear (forward / inverse).
- ``revised``: the mapping used everywhere: stretch r -> Schira (thesis alpha) -> exact 2-anchor camera
  (``configs/schira/revised.yaml``, ``registration_revised/README.md``).
- ``visual_field``: stimulus pixels <-> visual degrees (35 px/deg, fixation at the canvas corner) and grid stimuli.
"""

from src.retinotopy.revised import RevisedGeometry, load_revised_geometry
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

__all__ = [
    "RevisedGeometry",
    "SHEAR_AYZENSHTAT",
    "SHEAR_CONSTANT",
    "SHEAR_DOUBLE_SECH",
    "SchiraParams",
    "cartesian_deg_to_image_px",
    "cartesian_grid_line_degrees",
    "cartesian_to_polar",
    "compressed_polar",
    "cortical_uv_to_visual_deg",
    "forward_schira",
    "image_px_to_cartesian_deg",
    "image_px_to_polar",
    "inverse_schira",
    "load_revised_geometry",
    "polar_to_cartesian",
    "render_cartesian_degree_grid",
    "render_polar_degree_grid",
    "shear_fa",
    "stimulus_ink_mask",
    "upsample_render_canvas",
]
