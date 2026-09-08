"""Validate Schira 201118 VF scale: 35 px/deg, 1° letters, pixel→polar.

The existing 5×5 probe in ``experiments/retinotopic_map/render_grid_stimulus.py``
is equal-gap, not 0.5° spacing. The 0.5° Cartesian grid used here is
``render_cartesian_degree_grid``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image as PILImage

from src.paths import project_root, resolve_data_path
from src.retinotopy.params import load_schira_set
from src.retinotopy.plotting import (
    apply_vf_hm_vm_degree_ticks,
    draw_schira_ayzenshtat_c,
    draw_vf_cartesian_deg,
    draw_vf_polar_overlay,
    overlay_schira_polar_grid,
    plot_schira_cortex_comparison,
    set_schira_axes_visual_deg,
)
from src.retinotopy.schira import (
    cartesian_to_polar,
    cortical_uv_to_visual_deg,
    forward_schira,
    fovea_w,
    inverse_schira,
    polar_to_cartesian,
)
from src.retinotopy.visual_field import (
    _stamp_polyline_px,
    cartesian_deg_to_image_px,
    cartesian_grid_line_degrees,
    image_px_to_cartesian_deg,
    image_px_to_polar,
    overlay_ink,
    polar_grid_eccentricities_deg,
    polar_grid_theta_deg,
    render_cartesian_degree_grid,
    render_polar_degree_grid,
    stimulus_ink_mask,
    upsample_render_canvas,
)
from src.retinotopy.warp import forward_ink_cloud_w, sample_stimulus_onto_cortical_w
from src.stimuli.catalog import StimulusSpec, stimulus_spec_from_mapping
from src.stimuli.render import RenderConfig, _deg_to_px, render_stimulus

LAB_PX_PER_DEG = 35.0
LETTER_BOX_DEG = 1.0
QUADRANT_EXTENT_DEG = 6.0
CANONICAL_CANVAS = 210
SCHIRA_SET = "201118"
GRID_PLOT_NAME = "vf_grid_0p5deg__schira_201118.png"
POLAR_PLOT_NAME = "vf_grid_polar_0p5deg__schira_201118.png"
FIG12_PLOT_NAME = "vf_grid_fig12_polar_cartesian__schira_201118.png"
FIG12_POLAR_DEG_PLOT_NAME = (
    "vf_grid_fig12_polar_deg_D_1_4_3__schira_201118.png"
)
FIG12_CART_DEG_PLOT_NAME = (
    "vf_grid_fig12_cartesian_deg_D_1_4_3__schira_201118.png"
)
FIG12_DEG_GRIDS_ONLY_PLOT_NAME = (
    "vf_grid_fig12_polar_cartesian_deg__schira_201118.png"
)
AYZENSHTAT_FIG2_PLOT_NAME = "vf_ayzenshtat_fig2_style__schira_201118.png"
AYZENSHTAT_FIG2_FAR_D_PLOT_NAME = (
    "vf_ayzenshtat_fig2_style_D_ecc3p5__schira_201118.png"
)
AYZENSHTAT_NEAR_FAR_PLOT_NAME = (
    "vf_ayzenshtat_fig2_D_near_vs_far_same_scale__schira_201118.png"
)
AYZENSHTAT_MONKEY_PLOT_NAME = (
    "vf_ayzenshtat_fig2_monkey_2to5deg__paper_params.png"
)
AYZENSHTAT_MONKEY_SRC = Path(
    "/Users/ossnat/.cursor/projects/Users-ossnat-GondaResearch-VSD-FM-VSD-Encoder-01"
    "/assets/Screenshot_2026-09-01_at_16.58.32-426f60e6-2c53-473d-b066-3ad102f63bec.png"
)
# Ayzenshtat et al. 2012 fitted (a, k, α); k in mm. Sech S1/S2 stay Schira defaults.
AYZENSHTAT_PAPER_MONKEY_C = ("monkey C", 0.57, 7.7, 0.52)
AYZENSHTAT_PAPER_MONKEY_L = ("monkey L", 0.46, 4.87, 0.51)
AYZENSHTAT_FIG2A_FULL_PLOT_NAME = (
    "vf_ayzenshtat_fig2a_full_lower_left__paper_params.png"
)
AYZENSHTAT_FIG2A_FULL_SRC = Path(
    "/Users/ossnat/.cursor/projects/Users-ossnat-GondaResearch-VSD-FM-VSD-Encoder-01"
    "/assets/Screenshot_2026-09-01_at_17.33.50-1ee3b80c-c169-42ca-9669-db3b5c54543f.png"
)
# Full Fig. 2A frame: origin at top-right; long axis (VM, x=0) = 6°.
AYZENSHTAT_FIG2A_VM_DEG = 6.0
# Shared 2C window so near vs far magnification is comparable (not per-letter zoom).
SHARED_CORTEX_VF_DEG = 5.0
# 201118 catalog / thesis Fig. 13c
LETTER_D_POS_X_DEG = 1.0
LETTER_D_POS_Y_DEG = -0.9
LETTER_D_FAR_ECC_DEG = 3.5
LETTER_D_ECC_NEAR_DEG = 1.0
LETTER_D_ECC_FAR_DEG = 4.0
LETTER_D_LARGE_BOX_DEG = 3.0
# 3° D in the lower-right monkey patch: inner corner on E=3°, 45° ray.
LETTER_D_MONKEY_INNER_ECC_DEG = 3.0
LETTER_D_DISPLAY_RGB = (255, 255, 255)
UPSAMPLE = 3
UPSAMPLED_CANVAS = CANONICAL_CANVAS * UPSAMPLE  # 630
UPSAMPLED_PPD = LAB_PX_PER_DEG * UPSAMPLE  # 105


def _stimuli_yaml() -> dict:
    path = project_root() / "configs" / "stimuli" / "default.yaml"
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _load_201118_params():
    path = project_root() / "configs" / "schira" / "sets.yaml"
    name, params, _affine, block = load_schira_set(path, set_name=SCHIRA_SET)
    assert name == SCHIRA_SET
    return params, block


def _canonical_render_cfg() -> RenderConfig:
    from experiments.schira2007.run_schira_only import _render_config

    return _render_config(_stimuli_yaml(), canvas_size=CANONICAL_CANVAS)


def _filled_letter_spec(tmp_path: Path, *, pos_x: float, pos_y: float) -> StimulusSpec:
    bmp = tmp_path / "box.bmp"
    # Gray frame so BMP ink detection can crop a solid 20×20 white square.
    arr = np.full((22, 22, 3), 128, dtype=np.uint8)
    arr[1:-1, 1:-1] = 255
    PILImage.fromarray(arr).save(bmp)
    return StimulusSpec(
        monkey="gandalf",
        csv_date="20/11/2018",
        session_letter="c",
        h5_session="201118c",
        condition="condAN4",
        condition_num=4,
        stimulus_text="letter D",
        color="white",
        shape_type="letter",
        size_deg=LETTER_BOX_DEG,
        pos_x_deg=pos_x,
        pos_y_deg=pos_y,
        is_blank=False,
        cortex_file=None,
        letter="D",
        source_path=str(bmp),
    )


def test_schira_config_and_render_use_35px_per_degree():
    """Canonical Schira stimuli: 1° = 35 px on a 210 px / 6° quadrant."""
    yaml_cfg = _stimuli_yaml()
    assert float(yaml_cfg["pixels_per_deg"]) == pytest.approx(LAB_PX_PER_DEG)
    assert int(yaml_cfg["canvas_size"]) == CANONICAL_CANVAS
    assert float(yaml_cfg["quadrant_extent_deg"]) == pytest.approx(QUADRANT_EXTENT_DEG)
    assert float(yaml_cfg["letter_box_deg"]) == pytest.approx(LETTER_BOX_DEG)
    assert CANONICAL_CANVAS / QUADRANT_EXTENT_DEG == pytest.approx(LAB_PX_PER_DEG)

    defaults = RenderConfig()
    assert defaults.pixels_per_deg == pytest.approx(LAB_PX_PER_DEG)
    assert defaults.canvas_size == CANONICAL_CANVAS
    assert defaults.letter_box_deg == pytest.approx(LETTER_BOX_DEG)
    assert _deg_to_px(1.0, defaults) == pytest.approx(LAB_PX_PER_DEG)
    assert _deg_to_px(0.5, defaults) == pytest.approx(LAB_PX_PER_DEG / 2.0)

    schira_cfg = _canonical_render_cfg()
    assert schira_cfg.pixels_per_deg == pytest.approx(LAB_PX_PER_DEG)
    assert schira_cfg.canvas_size == CANONICAL_CANVAS
    assert schira_cfg.letter_box_deg == pytest.approx(LETTER_BOX_DEG)

    # 3× YAML canvas (210→630) keeps 1° = 35×3 = 105 px. 672 was 224×3 (old 37.33 px/deg).
    from experiments.schira2007.run_schira_only import _render_config

    assert upsample_render_canvas(CANONICAL_CANVAS) == UPSAMPLED_CANVAS
    hi = _render_config(yaml_cfg, canvas_size=None)
    # _render_config(None) stays at YAML canvas; run_schira_only applies 3×.
    assert hi.canvas_size == CANONICAL_CANVAS
    hi3 = _render_config(yaml_cfg, canvas_size=UPSAMPLED_CANVAS)
    assert hi3.pixels_per_deg == pytest.approx(UPSAMPLED_PPD)
    assert hi3.canvas_size == UPSAMPLED_CANVAS
    assert hi3.canvas_size / hi3.pixels_per_deg == pytest.approx(QUADRANT_EXTENT_DEG)
    assert UPSAMPLED_PPD == pytest.approx(LAB_PX_PER_DEG * 3)
    assert UPSAMPLED_CANVAS != 700

    params, block = _load_201118_params()
    assert block["date_prefix"] == SCHIRA_SET
    assert params.a == pytest.approx(0.72)
    assert params.alpha == pytest.approx(1.5)
    assert params.k == pytest.approx(1.0)
    assert params.shear == "double_sech"
    assert params.fa_combine == "power"
    assert params.sech_ecc_k == pytest.approx(0.76)
    assert params.sech_amp == pytest.approx(0.1821)


def test_schira_letter_box_is_one_degree_35_by_35_px(tmp_path: Path):
    """Letter bounding square is 1° = 35×35 px top↔bottom and left↔right."""
    cfg = _canonical_render_cfg()
    pos_x, pos_y = 1.0, -0.9
    spec = _filled_letter_spec(tmp_path, pos_x=pos_x, pos_y=pos_y)
    image = render_stimulus(spec, cfg)
    ink = stimulus_ink_mask(image, background_gray=cfg.background_gray)
    ys, xs = np.where(ink)
    assert ys.size > 0

    box_px = int(round(LETTER_BOX_DEG * cfg.pixels_per_deg))
    assert box_px == 35
    width = int(xs.max() - xs.min()) + 1
    height = int(ys.max() - ys.min()) + 1
    assert width == box_px
    assert height == box_px

    cx_px, cy_px = cartesian_deg_to_image_px(pos_x, pos_y, cfg)
    half = 0.5 * cfg.pixels_per_deg
    assert xs.min() >= int(round(cx_px - half)) - 1
    assert xs.max() <= int(round(cx_px + half)) + 1
    assert ys.min() >= int(round(cy_px - half)) - 1
    assert ys.max() <= int(round(cy_px + half)) + 1

    x_deg, y_deg = image_px_to_cartesian_deg(
        xs.astype(np.float64), ys.astype(np.float64), cfg
    )
    np.testing.assert_allclose(x_deg.min(), pos_x - 0.5, atol=1.5 / cfg.pixels_per_deg)
    np.testing.assert_allclose(x_deg.max(), pos_x + 0.5, atol=1.5 / cfg.pixels_per_deg)
    np.testing.assert_allclose(y_deg.max(), pos_y + 0.5, atol=1.5 / cfg.pixels_per_deg)
    np.testing.assert_allclose(y_deg.min(), pos_y - 0.5, atol=1.5 / cfg.pixels_per_deg)


def test_schira_converts_pixels_to_polar_then_forward_w(tmp_path: Path):
    """Ink pixels → Cartesian deg → (E, θ) → Schira w, using 201118 params."""
    cfg = _canonical_render_cfg()
    params, _ = _load_201118_params()
    spec = _filled_letter_spec(tmp_path, pos_x=1.0, pos_y=-0.9)
    image = render_stimulus(spec, cfg)
    ink = stimulus_ink_mask(image, background_gray=cfg.background_gray)
    yy, xx = np.where(ink)

    # Unfolded conversion (must match visual_field + schira + warp).
    x_deg = xx.astype(np.float64) / cfg.pixels_per_deg
    y_deg = -yy.astype(np.float64) / cfg.pixels_per_deg
    ecc = np.hypot(x_deg, y_deg)
    theta = np.arctan2(y_deg, x_deg)

    x2, y2 = image_px_to_cartesian_deg(xx.astype(np.float64), yy.astype(np.float64), cfg)
    e2, th2 = image_px_to_polar(xx.astype(np.float64), yy.astype(np.float64), cfg)
    e3, th3 = cartesian_to_polar(x_deg, y_deg)
    np.testing.assert_allclose(x2, x_deg)
    np.testing.assert_allclose(y2, y_deg)
    np.testing.assert_allclose(e2, ecc)
    np.testing.assert_allclose(th2, theta)
    np.testing.assert_allclose(e3, ecc)
    np.testing.assert_allclose(th3, theta)

    w_manual = forward_schira(ecc, theta, params)
    fu, fv, w_pipe = forward_ink_cloud_w(image, params=params, render_cfg=cfg)
    ok = np.isfinite(w_manual.real) & np.isfinite(w_manual.imag)
    np.testing.assert_allclose(fu, w_manual.real[ok])
    np.testing.assert_allclose(fv, w_manual.imag[ok])
    np.testing.assert_allclose(w_pipe, w_manual[ok])

    # Spot-check one pixel against the closed-form polar step.
    i = int(np.argmin(np.abs(x_deg - 1.0) + np.abs(y_deg + 0.9)))
    assert cfg.pixels_per_deg == pytest.approx(LAB_PX_PER_DEG)
    assert x_deg[i] == pytest.approx(xx[i] / LAB_PX_PER_DEG)
    assert y_deg[i] == pytest.approx(-yy[i] / LAB_PX_PER_DEG)


def _encoder_data_root() -> Path | None:
    path = resolve_data_path("Data/EncoderData")
    return path if path.is_dir() else None


def _letter_d_theta() -> float:
    return float(np.arctan2(LETTER_D_POS_Y_DEG, LETTER_D_POS_X_DEG))


def _letter_d_pos_at_ecc(ecc_deg: float) -> tuple[float, float]:
    th = _letter_d_theta()
    return float(ecc_deg * np.cos(th)), float(ecc_deg * np.sin(th))


def _letter_box_in_lower_right(
    pos_x: float,
    pos_y: float,
    size_deg: float,
    *,
    extent_deg: float = QUADRANT_EXTENT_DEG,
) -> bool:
    half = 0.5 * float(size_deg)
    return (
        pos_x - half >= 0.0
        and pos_y + half <= 0.0
        and pos_x + half <= float(extent_deg)
        and pos_y - half >= -float(extent_deg)
    )


def _monkey_patch_3deg_d_center() -> tuple[float, float]:
    """Center of a 3° box whose fovea-ward corner sits at E=3° on the 45° ray.

    Lower-right analogue of the Ayzenshtat monkey square: inner E=3°,
    outer corner ~7.2° (canvas still contains the 3° square).
    """
    half = 0.5 * LETTER_D_LARGE_BOX_DEG
    inner = LETTER_D_MONKEY_INNER_ECC_DEG / np.sqrt(2.0)
    return float(inner + half), float(-inner - half)


def _letter_d_201118_spec(
    tmp_path: Path,
    *,
    pos_x: float | None = None,
    pos_y: float | None = None,
    size_deg: float | None = None,
) -> StimulusSpec:
    """Catalog letter D glyph; position defaults to 201118 (1.0, −0.9)."""
    if pos_x is None:
        pos_x = LETTER_D_POS_X_DEG
    if pos_y is None:
        pos_y = LETTER_D_POS_Y_DEG
    if size_deg is None:
        size_deg = LETTER_BOX_DEG
    encoder = _encoder_data_root()
    source: Path | None = None
    if encoder is not None:
        candidate = encoder / "letters_stimuli" / "D.bmp"
        if candidate.is_file():
            source = candidate
    if source is None:
        source = tmp_path / "D_fallback.bmp"
        arr = np.full((48, 48, 3), 128, dtype=np.uint8)
        # Rough D so the inspect figure still has a glyph without EncoderData.
        arr[6:42, 10:16] = 0
        arr[6:12, 10:34] = 0
        arr[36:42, 10:34] = 0
        arr[12:36, 30:36] = 0
        PILImage.fromarray(arr).save(source)
    return StimulusSpec(
        monkey="gandalf",
        csv_date="20/11/2018",
        session_letter="c",
        h5_session="201118c",
        condition="condAN4",
        condition_num=4,
        stimulus_text="letter D",
        color="white",
        shape_type="letter",
        size_deg=float(size_deg),
        pos_x_deg=float(pos_x),
        pos_y_deg=float(pos_y),
        is_blank=False,
        cortex_file=None,
        letter="D",
        source_path=str(source),
    )


def _annotate_letter_d(
    ax,
    pos_x: float = LETTER_D_POS_X_DEG,
    pos_y: float = LETTER_D_POS_Y_DEG,
) -> None:
    ax.plot(
        pos_x,
        pos_y,
        marker="+",
        color="#1a5276",
        markersize=7,
        markeredgewidth=1.0,
        zorder=5,
    )
    ax.annotate(
        f"D ({pos_x:g}, {pos_y:g})",
        xy=(pos_x, pos_y),
        xytext=(pos_x + 0.85, pos_y + 0.55),
        fontsize=7,
        color="#1a5276",
        arrowprops=dict(arrowstyle="->", color="#1a5276", lw=0.6),
    )


def _composite_grids_with_letter_d(
    tmp_path: Path,
    cfg: RenderConfig,
    *,
    pos_x: float = LETTER_D_POS_X_DEG,
    pos_y: float = LETTER_D_POS_Y_DEG,
    size_deg: float = LETTER_BOX_DEG,
):
    spec = _letter_d_201118_spec(
        tmp_path, pos_x=pos_x, pos_y=pos_y, size_deg=size_deg
    )
    letter = render_stimulus(spec, cfg)
    polar_th, _ = render_polar_degree_grid(
        cfg, ecc_spacing_deg=1.0, theta_step_deg=15.0
    )
    cart_th, _ = render_cartesian_degree_grid(cfg, spacing_deg=1.0)
    polar = overlay_ink(
        polar_th,
        letter,
        background_gray=cfg.background_gray,
        color=LETTER_D_DISPLAY_RGB,
    )
    cart = overlay_ink(
        cart_th,
        letter,
        background_gray=cfg.background_gray,
        color=LETTER_D_DISPLAY_RGB,
    )
    return letter, polar, cart


def _stamp_1deg_box(
    rgb: np.ndarray,
    cfg: RenderConfig,
    pos_x: float,
    pos_y: float,
    *,
    size_deg: float = LETTER_BOX_DEG,
    color: tuple[int, int, int] = (255, 200, 0),
    width_px: int = 2,
) -> np.ndarray:
    """Axis-aligned square around the letter (default 1°) so the warp is visible."""
    out = np.array(rgb, dtype=np.uint8, copy=True)
    half = 0.5 * float(size_deg)
    n = 48
    xs = np.concatenate(
        [
            np.linspace(pos_x - half, pos_x + half, n),
            np.full(n, pos_x + half),
            np.linspace(pos_x + half, pos_x - half, n),
            np.full(n, pos_x - half),
        ]
    )
    ys = np.concatenate(
        [
            np.full(n, pos_y + half),
            np.linspace(pos_y + half, pos_y - half, n),
            np.full(n, pos_y - half),
            np.linspace(pos_y - half, pos_y + half, n),
        ]
    )
    x_px, y_px = cartesian_deg_to_image_px(xs, ys, cfg)
    _stamp_polyline_px(out, x_px, y_px, color=color, width=width_px)
    return out


def _cortical_window_for_deg_box(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    params,
    *,
    margin_frac: float = 0.12,
    n: int = 48,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Cortical ``(u, v)`` bbox of an axis-aligned VF rectangle."""
    xs = np.linspace(float(x0), float(x1), int(n))
    ys = np.linspace(float(y0), float(y1), int(n))
    xx, yy = np.meshgrid(xs, ys)
    ecc, theta = cartesian_to_polar(xx.ravel(), yy.ravel())
    w = forward_schira(ecc, theta, params)
    ok = np.isfinite(w.real) & np.isfinite(w.imag)
    u, v = w.real[ok], w.imag[ok]
    span = max(float(u.max() - u.min()), float(v.max() - v.min()), 1e-3)
    pad = float(margin_frac) * span
    return (float(u.min() - pad), float(u.max() + pad)), (
        float(v.min() - pad),
        float(v.max() + pad),
    )


def _cortical_window_for_vf(
    rgb: np.ndarray,
    params,
    cfg: RenderConfig,
    *,
    vf_max_deg: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    u_range, v_range = _w_bbox_in_vf_window(
        rgb, params, cfg, x_max=vf_max_deg, y_min=-vf_max_deg
    )
    u0, u1 = u_range
    v0, _v1 = v_range
    u_fov = float(np.real(fovea_w(params)))
    return (min(u0, u_fov), u1), (v0, 0.0)


def _w_bbox_in_vf_window(
    rgb: np.ndarray,
    params,
    cfg: RenderConfig,
    *,
    x_max: float,
    y_min: float,
    margin_frac: float = 0.18,
) -> tuple[tuple[float, float], tuple[float, float]]:
    ink = stimulus_ink_mask(rgb, background_gray=cfg.background_gray)
    yy, xx = np.where(ink)
    x_deg, y_deg = image_px_to_cartesian_deg(
        xx.astype(np.float64), yy.astype(np.float64), cfg
    )
    inside = (x_deg >= 0.0) & (x_deg <= x_max) & (y_deg <= 0.0) & (y_deg >= y_min)
    ecc, theta = cartesian_to_polar(x_deg[inside], y_deg[inside])
    w = forward_schira(ecc, theta, params)
    ok = np.isfinite(w.real) & np.isfinite(w.imag)
    u, v = w.real[ok], w.imag[ok]
    span = max(float(u.max() - u.min()), float(v.max() - v.min()), 1e-3)
    pad = float(margin_frac) * span
    return (float(u.min() - pad), float(u.max() + pad)), (
        float(v.min() - pad),
        float(v.max() + pad),
    )


@pytest.mark.skipif(_encoder_data_root() is None, reason="Data/EncoderData not mounted")
def test_catalog_201118_letters_use_one_degree_box():
    """Catalog letters that Schira actually maps are size_deg=1 and fit in 35×35."""
    from src.stimuli.catalog import load_full_encoder_catalog
    from src.stimuli.identity import attach_stimulus_ids

    encoder_root = _encoder_data_root()
    assert encoder_root is not None
    cfg = _canonical_render_cfg()
    catalog = load_full_encoder_catalog(
        encoder_root, monkey="gandalf", bar_length_deg=cfg.bar_length_deg
    )
    letters = catalog[
        catalog["h5_session"].astype(str).str.startswith(SCHIRA_SET)
        & (catalog["shape_type"].astype(str) == "letter")
        & (~catalog["is_blank"].astype(bool))
    ].copy()
    letters = attach_stimulus_ids(letters)
    letters = letters.groupby("stimulus_id", sort=True).first().reset_index()
    assert not letters.empty
    np.testing.assert_allclose(
        letters["size_deg"].astype(float).to_numpy(), LETTER_BOX_DEG
    )

    box_px = int(round(LETTER_BOX_DEG * cfg.pixels_per_deg))
    assert box_px == 35
    checked = 0
    for _, row in letters.iterrows():
        spec = stimulus_spec_from_mapping(row.to_dict())
        if spec.source_path is None or not Path(spec.source_path).is_file():
            continue
        image = render_stimulus(spec, cfg)
        ink = stimulus_ink_mask(image, background_gray=cfg.background_gray)
        ys, xs = np.where(ink)
        assert ys.size > 0, spec.stimulus_id
        width = int(xs.max() - xs.min()) + 1
        height = int(ys.max() - ys.min()) + 1
        assert width <= box_px + 1, spec.stimulus_id
        assert height <= box_px + 1, spec.stimulus_id
        cx_px, cy_px = cartesian_deg_to_image_px(spec.pos_x_deg, spec.pos_y_deg, cfg)
        half = 0.5 * cfg.pixels_per_deg
        assert xs.min() >= int(round(cx_px - half)) - 1
        assert xs.max() <= int(round(cx_px + half)) + 1
        assert ys.min() >= int(round(cy_px - half)) - 1
        assert ys.max() <= int(round(cy_px + half)) + 1
        checked += 1
    assert checked >= 1


def test_schira_201118_half_degree_line_grid(tmp_path: Path):
    """0.5° H/V grid through the same 201118 Schira path as the letters."""
    cfg = _canonical_render_cfg()
    params, _ = _load_201118_params()
    expected = cartesian_grid_line_degrees(
        spacing_deg=0.5, extent_deg=QUADRANT_EXTENT_DEG, include_zero=True
    )
    np.testing.assert_allclose(np.diff(expected), 0.5)
    assert expected[0] == pytest.approx(0.0)
    assert expected[-1] == pytest.approx(5.5)

    rgb, meta = render_cartesian_degree_grid(cfg, spacing_deg=0.5, line_width_px=1)
    assert rgb.shape == (CANONICAL_CANVAS, CANONICAL_CANVAS, 3)
    assert stimulus_ink_mask(rgb, background_gray=cfg.background_gray)[0].any()
    np.testing.assert_allclose(meta["vertical_x_deg"], expected)
    np.testing.assert_allclose(meta["horizontal_y_deg"], -expected)
    np.testing.assert_allclose(meta["pixels_per_deg"], LAB_PX_PER_DEG)

    ink = stimulus_ink_mask(rgb, background_gray=cfg.background_gray)
    yy, xx = np.where(ink)
    x_deg, y_deg = image_px_to_cartesian_deg(
        xx.astype(np.float64), yy.astype(np.float64), cfg
    )
    # Interior pixels of vertical lines (not crossings) have constant x.
    on_vertical = np.min(np.abs(x_deg[:, None] - expected[None, :]), axis=1) <= (
        0.6 / LAB_PX_PER_DEG
    )
    on_horizontal = np.min(np.abs(y_deg[:, None] + expected[None, :]), axis=1) <= (
        0.6 / LAB_PX_PER_DEG
    )
    assert on_vertical.mean() > 0.4
    assert on_horizontal.mean() > 0.4

    ecc, theta = image_px_to_polar(xx.astype(np.float64), yy.astype(np.float64), cfg)
    w_manual = forward_schira(ecc, theta, params)
    fu, fv, w_pipe = forward_ink_cloud_w(rgb, params=params, render_cfg=cfg)
    ok = np.isfinite(w_manual.real)
    np.testing.assert_allclose(w_pipe, w_manual[ok])
    assert fu.size > 100

    warped, valid, u_range, v_range = sample_stimulus_onto_cortical_w(
        rgb,
        params=params,
        render_cfg=cfg,
        grid_size=180,
        max_ecc_deg=QUADRANT_EXTENT_DEG * np.sqrt(2.0),
    )
    assert valid.any()
    assert warped.shape[0] == 180

    param_note = (
        f"set={SCHIRA_SET} · 0.5° H/V grid · 35 px/deg · "
        f"a={params.a:g} α={params.alpha:g} k={params.k:g} "
        f"shear={params.shear} fa_combine={params.fa_combine} "
        f"sech_amp={params.sech_amp:g}"
    )
    tmp_fig = tmp_path / GRID_PLOT_NAME
    plot_schira_cortex_comparison(
        rgb,
        fu,
        fv,
        warped,
        u_range,
        v_range,
        output_path=tmp_fig,
        title="0.5° Cartesian grid · Schira 201118 (no camera affine)",
        param_note=param_note,
        point_size=1.2,
        show_inverse=False,
    )
    assert tmp_fig.is_file() and tmp_fig.stat().st_size > 0

    inspect_dir = project_root() / "experiments" / "schira2007" / "phase_0_schira"
    inspect_dir.mkdir(parents=True, exist_ok=True)
    inspect_fig = inspect_dir / GRID_PLOT_NAME
    plot_schira_cortex_comparison(
        rgb,
        fu,
        fv,
        warped,
        u_range,
        v_range,
        output_path=inspect_fig,
        title="0.5° Cartesian grid · Schira 201118 (no camera affine)",
        param_note=param_note,
        point_size=1.2,
        show_inverse=False,
    )
    assert inspect_fig.is_file()


def _schira_map_grid(rgb, params, cfg, *, grid_size: int = 180):
    fu, fv, _w = forward_ink_cloud_w(rgb, params=params, render_cfg=cfg)
    warped, valid, u_range, v_range = sample_stimulus_onto_cortical_w(
        rgb,
        params=params,
        render_cfg=cfg,
        grid_size=grid_size,
        max_ecc_deg=QUADRANT_EXTENT_DEG * np.sqrt(2.0),
    )
    return fu, fv, warped, valid, u_range, v_range


def _inspect_dir() -> Path:
    path = project_root() / "experiments" / "schira2007" / "phase_0_schira"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_schira_201118_polar_grid(tmp_path: Path):
    """Iso-E / iso-θ polar grid (thesis Fig. 12A) through 201118 Schira params."""
    import matplotlib.pyplot as plt

    cfg = _canonical_render_cfg()
    params, _ = _load_201118_params()
    eccs = polar_grid_eccentricities_deg(
        spacing_deg=0.5, extent_deg=QUADRANT_EXTENT_DEG
    )
    thetas = polar_grid_theta_deg(step_deg=15.0)
    np.testing.assert_allclose(np.diff(eccs), 0.5)
    np.testing.assert_allclose(thetas[0], 0.0)
    np.testing.assert_allclose(thetas[-1], -90.0)

    rgb, meta = render_polar_degree_grid(
        cfg, ecc_spacing_deg=0.5, theta_step_deg=15.0, line_width_px=1
    )
    assert rgb.shape == (CANONICAL_CANVAS, CANONICAL_CANVAS, 3)
    np.testing.assert_allclose(meta["eccentricity_deg"], eccs)
    np.testing.assert_allclose(meta["theta_deg"], thetas)
    np.testing.assert_allclose(meta["pixels_per_deg"], LAB_PX_PER_DEG)

    ink = stimulus_ink_mask(rgb, background_gray=cfg.background_gray)
    yy, xx = np.where(ink)
    x_deg, y_deg = image_px_to_cartesian_deg(
        xx.astype(np.float64), yy.astype(np.float64), cfg
    )
    ecc, theta = image_px_to_polar(xx.astype(np.float64), yy.astype(np.float64), cfg)
    on_ring = np.min(np.abs(ecc[:, None] - eccs[None, :]), axis=1) <= (
        1.2 / LAB_PX_PER_DEG
    )
    on_ray = np.min(np.abs(theta[:, None] - np.deg2rad(thetas)[None, :]), axis=1) <= (
        np.deg2rad(3.0)
    )
    assert (on_ring | on_ray).mean() > 0.85
    assert on_ring.any() and on_ray.any()

    w_manual = forward_schira(ecc, theta, params)
    fu, fv, warped, valid, u_range, v_range = _schira_map_grid(rgb, params, cfg)
    ok = np.isfinite(w_manual.real)
    np.testing.assert_allclose(np.column_stack([fu, fv]), np.column_stack([
        w_manual.real[ok], w_manual.imag[ok]
    ]))
    assert fu.size > 100
    assert valid.any()

    param_note = (
        f"set={SCHIRA_SET} · polar ΔE=0.5° Δθ=15° · 35 px/deg · "
        f"a={params.a:g} α={params.alpha:g} k={params.k:g} "
        f"shear={params.shear} fa_combine={params.fa_combine} "
        f"sech_amp={params.sech_amp:g}"
    )
    for dest in (tmp_path / POLAR_PLOT_NAME, _inspect_dir() / POLAR_PLOT_NAME):
        plot_schira_cortex_comparison(
            rgb,
            fu,
            fv,
            warped,
            u_range,
            v_range,
            output_path=dest,
            title="Polar grid (iso-E, iso-θ) · Schira 201118 (no camera affine)",
            param_note=param_note,
            point_size=1.2,
            show_inverse=False,
        )
        assert dest.is_file() and dest.stat().st_size > 0


def test_fig12_grids_with_letter_d(tmp_path: Path):
    """Thesis Fig. 12 grids plus catalog letter D at (1.0, −0.9), same Schira path."""
    import matplotlib.pyplot as plt

    cfg = _canonical_render_cfg()
    params, _ = _load_201118_params()
    spec = _letter_d_201118_spec(tmp_path)
    letter = render_stimulus(spec, cfg)
    letter_ink = stimulus_ink_mask(letter, background_gray=cfg.background_gray)
    yy, xx = np.where(letter_ink)
    assert yy.size > 0
    x_deg, y_deg = image_px_to_cartesian_deg(
        xx.astype(np.float64), yy.astype(np.float64), cfg
    )
    np.testing.assert_allclose(x_deg.mean(), LETTER_D_POS_X_DEG, atol=0.35)
    np.testing.assert_allclose(y_deg.mean(), LETTER_D_POS_Y_DEG, atol=0.35)

    polar_th, _ = render_polar_degree_grid(
        cfg, ecc_spacing_deg=1.0, theta_step_deg=15.0
    )
    cart_th, _ = render_cartesian_degree_grid(cfg, spacing_deg=1.0)
    polar = overlay_ink(
        polar_th,
        letter,
        background_gray=cfg.background_gray,
        color=LETTER_D_DISPLAY_RGB,
    )
    cart = overlay_ink(
        cart_th,
        letter,
        background_gray=cfg.background_gray,
        color=LETTER_D_DISPLAY_RGB,
    )
    assert stimulus_ink_mask(polar, background_gray=cfg.background_gray).sum() > letter_ink.sum()
    assert stimulus_ink_mask(cart, background_gray=cfg.background_gray).sum() > letter_ink.sum()

    p_grid_u, p_grid_v, _ = forward_ink_cloud_w(polar_th, params=params, render_cfg=cfg)
    c_grid_u, c_grid_v, _ = forward_ink_cloud_w(cart_th, params=params, render_cfg=cfg)
    letter_u, letter_v, _ = forward_ink_cloud_w(letter, params=params, render_cfg=cfg)
    assert letter_u.size > 0

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.2), layout="constrained")
    rows = [
        ("A  polar (Fig. 12A)", polar, p_grid_u, p_grid_v),
        ("B  cartesian (Fig. 12B)", cart, c_grid_u, c_grid_v),
    ]
    for r, (label, stim, grid_u, grid_v) in enumerate(rows):
        grid_u = np.asarray(grid_u, dtype=np.float64)
        grid_v = np.asarray(grid_v, dtype=np.float64)
        span = max(float(grid_u.max() - grid_u.min()), float(grid_v.max() - grid_v.min()), 1e-3)
        pad = 0.12 * span
        u_lo, u_hi = float(grid_u.min() - pad), float(grid_u.max() + pad)
        v_lo, v_hi = float(grid_v.min() - pad), float(grid_v.max() + pad)
        bg = float(cfg.background_gray) / 255.0
        draw_vf_cartesian_deg(axes[r, 0], stim, cfg)
        _annotate_letter_d(axes[r, 0])
        axes[r, 0].set_title(f"{label} · VF (x, y) deg", fontsize=9)
        draw_vf_polar_overlay(axes[r, 1], stim, cfg)
        _annotate_letter_d(axes[r, 1])
        axes[r, 1].set_title("same pixels + (E, θ) overlay", fontsize=9)
        axes[r, 2].set_facecolor((bg, bg, bg))
        axes[r, 2].scatter(
            grid_u, grid_v, s=1.0, c="black", marker="s", linewidths=0, zorder=1
        )
        axes[r, 2].scatter(
            letter_u,
            letter_v,
            s=2.4,
            c="white",
            marker="s",
            linewidths=0.15,
            edgecolors="0.15",
            zorder=2,
        )
        axes[r, 2].set_xlim(u_lo, u_hi)
        axes[r, 2].set_ylim(v_lo, v_hi)
        axes[r, 2].set_aspect("equal")
        axes[r, 2].set_xlabel("u", fontsize=9)
        axes[r, 2].set_ylabel("v", fontsize=9)
        axes[r, 2].set_title("Schira w (u, v) · D on top", fontsize=9)
        axes[r, 2].axhline(0.0, color="0.75", lw=0.5, ls="--")
        axes[r, 2].axvline(0.0, color="0.75", lw=0.5, ls="--")
    fig.suptitle(
        "Thesis Fig. 12 style · grids + letter D · Schira 201118 (no camera affine)",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        f"set={SCHIRA_SET} · letter D at ({LETTER_D_POS_X_DEG:g}, {LETTER_D_POS_Y_DEG:g}) "
        f"· 1° box · polar 1°/15° · cartesian 1° · 35 px/deg · "
        f"a={params.a:g} α={params.alpha:g} k={params.k:g} sech_amp={params.sech_amp:g}",
        ha="center",
        fontsize=7,
        color="0.35",
    )
    fig12 = _inspect_dir() / FIG12_PLOT_NAME
    fig.savefig(tmp_path / FIG12_PLOT_NAME, dpi=160, bbox_inches="tight")
    fig.savefig(fig12, dpi=160, bbox_inches="tight")
    plt.close(fig)
    assert fig12.is_file()


def _write_fig12_degree_grid(
    tmp_path: Path,
    *,
    grid: str,
    output_name: str,
) -> None:
    """Fig. 12 polar or cartesian grid, degree axes, thesis 201118, lower-right."""
    import matplotlib.pyplot as plt

    assert grid in ("polar", "cartesian")
    cfg = _canonical_render_cfg()
    params, _ = _load_201118_params()
    near_x, near_y = _letter_d_pos_at_ecc(LETTER_D_ECC_NEAR_DEG)
    far_x, far_y = _letter_d_pos_at_ecc(LETTER_D_ECC_FAR_DEG)
    big_x, big_y = _monkey_patch_3deg_d_center()
    assert _letter_box_in_lower_right(near_x, near_y, LETTER_BOX_DEG)
    assert _letter_box_in_lower_right(far_x, far_y, LETTER_BOX_DEG)
    assert not _letter_box_in_lower_right(
        near_x, near_y, LETTER_D_LARGE_BOX_DEG
    )
    assert _letter_box_in_lower_right(big_x, big_y, LETTER_D_LARGE_BOX_DEG)

    cases = [
        (f"E={LETTER_D_ECC_NEAR_DEG:g}° · 1° D", near_x, near_y, LETTER_BOX_DEG),
        (f"E={LETTER_D_ECC_FAR_DEG:g}° · 1° D", far_x, far_y, LETTER_BOX_DEG),
        ("3–6° patch · 3° D", big_x, big_y, LETTER_D_LARGE_BOX_DEG),
    ]
    if grid == "polar":
        base_grid, _ = render_polar_degree_grid(
            cfg, ecc_spacing_deg=1.0, theta_step_deg=15.0
        )
        grid_label = "polar (Fig. 12A)"
    else:
        base_grid, _ = render_cartesian_degree_grid(cfg, spacing_deg=1.0)
        grid_label = "cartesian (Fig. 12B)"

    u_range, v_range = _cortical_window_for_vf(
        base_grid, params, cfg, vf_max_deg=QUADRANT_EXTENT_DEG
    )
    ecc_ticks = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    theta_ticks = [0.0, -15.0, -30.0, -45.0, -60.0, -75.0, -90.0]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.4), layout="constrained")
    for c, (label, px, py, size_deg) in enumerate(cases):
        letter, polar, cart = _composite_grids_with_letter_d(
            tmp_path, cfg, pos_x=px, pos_y=py, size_deg=size_deg
        )
        stim = polar if grid == "polar" else cart
        stim = _stamp_1deg_box(stim, cfg, px, py, size_deg=size_deg)
        warped, valid, ur, vr = sample_stimulus_onto_cortical_w(
            stim,
            params=params,
            render_cfg=cfg,
            grid_size=280,
            u_range=u_range,
            v_range=v_range,
            max_ecc_deg=QUADRANT_EXTENT_DEG * np.sqrt(2.0),
        )
        assert valid.any()

        draw_vf_cartesian_deg(axes[0, c], stim, cfg)
        axes[0, c].plot(0.0, 0.0, "o", color="#c0392b", markersize=5, zorder=6)
        _annotate_letter_d(axes[0, c], px, py)
        apply_vf_hm_vm_degree_ticks(axes[0, c], cfg)
        axes[0, c].set_title(f"VF · {label}", fontsize=9)

        draw_schira_ayzenshtat_c(
            axes[1, c],
            warped,
            ur,
            vr,
            params,
            ecc_ticks_deg=ecc_ticks,
            theta_ticks_deg=theta_ticks,
            ecc_ref_deg=2.5,
            fovea_on_right=False,
            hm_on_bottom=False,
        )
        lu, lv, _ = forward_ink_cloud_w(letter, params=params, render_cfg=cfg)
        axes[1, c].scatter(
            lu, lv, s=4.0, c="white", marker="s", linewidths=0, zorder=5
        )
        axes[1, c].set_title("Schira (fovea left)", fontsize=9)
        assert axes[1, c].get_xlim()[0] < axes[1, c].get_xlim()[1]
        assert axes[1, c].get_ylim()[0] < axes[1, c].get_ylim()[1]

    fig.suptitle(
        f"Thesis Fig. 12 {grid_label} · D at 1° & 4° + 3° D in 3–6° monkey patch · "
        "Schira 201118 lower-right · fovea left (no 2C flip)",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        "Thesis params, lower-right only (x>0, y<0). 1° Ds: catalog polar. "
        "3° D: fovea-ward corner at E=3° on 45° (monkey square); outer corner ~7°. "
        f"a={params.a:g} α={params.alpha:g} k={params.k:g} "
        f"fa={params.fa_combine} sech_amp={params.sech_amp:g}.",
        ha="center",
        fontsize=7,
        color="0.35",
    )
    out = _inspect_dir() / output_name
    fig.savefig(tmp_path / output_name, dpi=160, bbox_inches="tight")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    assert out.is_file()


def test_fig12_polar_degree_d_at_1_and_4deg(tmp_path: Path):
    _write_fig12_degree_grid(
        tmp_path, grid="polar", output_name=FIG12_POLAR_DEG_PLOT_NAME
    )


def test_fig12_cartesian_degree_d_at_1_and_4deg(tmp_path: Path):
    _write_fig12_degree_grid(
        tmp_path, grid="cartesian", output_name=FIG12_CART_DEG_PLOT_NAME
    )


def test_fig12_degree_grids_only(tmp_path: Path):
    """Fig. 12 polar + cartesian grids in degrees, no letter (thesis 201118)."""
    import matplotlib.pyplot as plt

    cfg = _canonical_render_cfg()
    params, _ = _load_201118_params()
    polar, _ = render_polar_degree_grid(
        cfg, ecc_spacing_deg=1.0, theta_step_deg=15.0
    )
    cart, _ = render_cartesian_degree_grid(cfg, spacing_deg=1.0)
    ecc_ticks = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    theta_ticks = [0.0, -15.0, -30.0, -45.0, -60.0, -75.0, -90.0]
    rows = [
        ("A  polar (Fig. 12A)", polar),
        ("B  cartesian (Fig. 12B)", cart),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.4), layout="constrained")
    for r, (label, stim) in enumerate(rows):
        u_range, v_range = _cortical_window_for_vf(
            stim, params, cfg, vf_max_deg=QUADRANT_EXTENT_DEG
        )
        warped, valid, ur, vr = sample_stimulus_onto_cortical_w(
            stim,
            params=params,
            render_cfg=cfg,
            grid_size=280,
            u_range=u_range,
            v_range=v_range,
            max_ecc_deg=QUADRANT_EXTENT_DEG * np.sqrt(2.0),
        )
        assert valid.any()
        draw_vf_cartesian_deg(axes[r, 0], stim, cfg)
        axes[r, 0].plot(0.0, 0.0, "o", color="#c0392b", markersize=5, zorder=6)
        apply_vf_hm_vm_degree_ticks(axes[r, 0], cfg)
        axes[r, 0].set_title(f"{label} · VF", fontsize=9)
        draw_schira_ayzenshtat_c(
            axes[r, 1],
            warped,
            ur,
            vr,
            params,
            ecc_ticks_deg=ecc_ticks,
            theta_ticks_deg=theta_ticks,
            ecc_ref_deg=2.5,
            fovea_on_right=False,
            hm_on_bottom=False,
        )
        axes[r, 1].set_title("Schira (fovea left)", fontsize=9)
        assert axes[r, 1].get_xlim()[0] < axes[r, 1].get_xlim()[1]
        assert axes[r, 1].get_ylim()[0] < axes[r, 1].get_ylim()[1]
    fig.suptitle(
        "Thesis Fig. 12 · polar & cartesian grids · Schira 201118 "
        "lower-right · fovea left (no 2C flip)",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        "Thesis params, lower-right only (x>0, y<0). Polar 1°/15°, cartesian 1°. "
        f"a={params.a:g} α={params.alpha:g} k={params.k:g} "
        f"fa={params.fa_combine} sech_amp={params.sech_amp:g}.",
        ha="center",
        fontsize=7,
        color="0.35",
    )
    out = _inspect_dir() / FIG12_DEG_GRIDS_ONLY_PLOT_NAME
    fig.savefig(tmp_path / FIG12_DEG_GRIDS_ONLY_PLOT_NAME, dpi=160, bbox_inches="tight")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    assert out.is_file()


def test_cortical_uv_inverse_matches_ayzenshtat_radial_formula():
    """On the HM, E = exp(u/k)−a equals a·(exp(u'/k)−1) with u' = u − k log a."""
    params, _ = _load_201118_params()
    ecc = np.array([0.5, 1.0, 2.0, 4.0])
    w = forward_schira(ecc, np.zeros_like(ecc), params)
    e_inv, th_inv, x_inv, y_inv = cortical_uv_to_visual_deg(w.real, w.imag, params)
    np.testing.assert_allclose(e_inv, ecc, atol=1e-6)
    np.testing.assert_allclose(th_inv, 0.0, atol=1e-6)
    np.testing.assert_allclose(x_inv, ecc, atol=1e-6)
    np.testing.assert_allclose(y_inv, 0.0, atol=1e-6)

    e_hm = np.exp(w.real / params.k) - params.a
    np.testing.assert_allclose(e_hm, ecc, atol=1e-10)
    u_shift = w.real - params.k * np.log(params.a)
    e_paper = params.a * (np.exp(u_shift / params.k) - 1.0)
    np.testing.assert_allclose(e_paper, ecc, atol=1e-10)


def test_ayzenshtat_fig2_style_grids_with_letter_d(tmp_path: Path):
    """Ayzenshtat 2012 Fig. 2A–C layout: VF, stimulus-zone zoom, Schira in visual deg."""
    _write_ayzenshtat_fig2(
        tmp_path,
        pos_x=LETTER_D_POS_X_DEG,
        pos_y=LETTER_D_POS_Y_DEG,
        zoom_deg=2.5,
        output_name=AYZENSHTAT_FIG2_PLOT_NAME,
        note="catalog D",
    )


def test_ayzenshtat_fig2_letter_d_at_3p5deg_same_polar(tmp_path: Path):
    """Same polar angle as catalog D, eccentricity 3.5° (away from the fovea)."""
    pos_x, pos_y = _letter_d_pos_at_ecc(LETTER_D_FAR_ECC_DEG)
    th0 = _letter_d_theta()
    np.testing.assert_allclose(np.arctan2(pos_y, pos_x), th0, atol=1e-12)
    np.testing.assert_allclose(np.hypot(pos_x, pos_y), LETTER_D_FAR_ECC_DEG, atol=1e-12)
    assert pos_x + 0.5 < QUADRANT_EXTENT_DEG
    assert pos_y - 0.5 > -QUADRANT_EXTENT_DEG
    _write_ayzenshtat_fig2(
        tmp_path,
        pos_x=pos_x,
        pos_y=pos_y,
        zoom_deg=5.0,
        output_name=AYZENSHTAT_FIG2_FAR_D_PLOT_NAME,
        note="same P, E=3.5°",
    )


def _write_ayzenshtat_fig2(
    tmp_path: Path,
    *,
    pos_x: float,
    pos_y: float,
    zoom_deg: float,
    output_name: str,
    note: str,
) -> None:
    import matplotlib.pyplot as plt

    cfg = _canonical_render_cfg()
    assert cfg.pixels_per_deg == pytest.approx(LAB_PX_PER_DEG)
    params, _ = _load_201118_params()
    _letter, polar, cart = _composite_grids_with_letter_d(
        tmp_path, cfg, pos_x=pos_x, pos_y=pos_y
    )
    polar = _stamp_1deg_box(polar, cfg, pos_x, pos_y)
    cart = _stamp_1deg_box(cart, cfg, pos_x, pos_y)
    letter_ecc = float(np.hypot(pos_x, pos_y))
    ecc_ticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    theta_ticks = [0.0, -15.0, -30.0, -45.0, -60.0, -75.0, -90.0]
    tick_ecc_ref = 2.5

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4), layout="constrained")
    rows = [
        ("A  polar grid + D", polar),
        ("A  cartesian grid + D", cart),
    ]
    for r, (label, stim) in enumerate(rows):
        u_range, v_range = _cortical_window_for_vf(
            stim, params, cfg, vf_max_deg=SHARED_CORTEX_VF_DEG
        )
        warped, _valid, u_range, v_range = sample_stimulus_onto_cortical_w(
            stim,
            params=params,
            render_cfg=cfg,
            grid_size=280,
            u_range=u_range,
            v_range=v_range,
            max_ecc_deg=QUADRANT_EXTENT_DEG * np.sqrt(2.0),
        )

        draw_vf_cartesian_deg(axes[r, 0], stim, cfg)
        axes[r, 0].plot(0.0, 0.0, "o", color="#c0392b", markersize=5, zorder=6)
        _annotate_letter_d(axes[r, 0], pos_x, pos_y)
        apply_vf_hm_vm_degree_ticks(axes[r, 0], cfg)
        axes[r, 0].set_title(f"{label}  (Fig. 2A)", fontsize=9)

        draw_vf_cartesian_deg(axes[r, 1], stim, cfg)
        axes[r, 1].plot(0.0, 0.0, "o", color="#c0392b", markersize=5, zorder=6)
        _annotate_letter_d(axes[r, 1], pos_x, pos_y)
        axes[r, 1].set_xlim(0.0, zoom_deg)
        axes[r, 1].set_ylim(-zoom_deg, 0.0)
        apply_vf_hm_vm_degree_ticks(axes[r, 1], cfg)
        axes[r, 1].set_title("B  stimulus zone (Fig. 2B)", fontsize=9)

        draw_schira_ayzenshtat_c(
            axes[r, 2],
            warped,
            u_range,
            v_range,
            params,
            ecc_ticks_deg=ecc_ticks,
            theta_ticks_deg=theta_ticks,
            ecc_ref_deg=tick_ecc_ref,
        )
        lu, lv, _ = forward_ink_cloud_w(_letter, params=params, render_cfg=cfg)
        axes[r, 2].scatter(
            lu, lv, s=5.0, c="white", marker="s", linewidths=0, zorder=5
        )
        axes[r, 2].set_title("C  Schira, axes in visual deg (Fig. 2C)", fontsize=9)
        xlim = axes[r, 2].get_xlim()
        ylim = axes[r, 2].get_ylim()
        assert xlim[0] > xlim[1], "fovea (small u) should sit on the right"
        assert ylim[0] > ylim[1], "HM (v=0) should sit on the bottom"

    w_d = forward_schira(np.hypot(pos_x, pos_y), np.arctan2(pos_y, pos_x), params)
    e_d, th_d, x_d, y_d = cortical_uv_to_visual_deg(w_d.real, w_d.imag, params)
    np.testing.assert_allclose(float(x_d), pos_x, atol=1e-6)
    np.testing.assert_allclose(float(y_d), pos_y, atol=1e-6)
    np.testing.assert_allclose(float(e_d), letter_ecc, atol=1e-6)
    np.testing.assert_allclose(
        float(th_d), np.rad2deg(np.arctan2(pos_y, pos_x)), atol=1e-5
    )

    fig.suptitle(
        f"Ayzenshtat 2012 Fig. 2 style · {note} · Schira 201118 (no camera affine)",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        "A/B: 1° ticks (35 px/deg) · x=HM, y=VM. "
        "C: same 0–5° cortical window for near and far D (foveal mag not hidden by zoom). "
        f"D at ({pos_x:.2f}, {pos_y:.2f}) · E={letter_ecc:.2f}° · "
        f"a={params.a:g} α={params.alpha:g} k={params.k:g}",
        ha="center",
        fontsize=7,
        color="0.35",
    )
    out = _inspect_dir() / output_name
    fig.savefig(tmp_path / output_name, dpi=160, bbox_inches="tight")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    assert out.is_file()


def _letter_cortical_span(tmp_path: Path, pos_x: float, pos_y: float, cfg, params):
    spec = _letter_d_201118_spec(tmp_path, pos_x=pos_x, pos_y=pos_y)
    img = render_stimulus(spec, cfg)
    ink = stimulus_ink_mask(img, background_gray=cfg.background_gray)
    yy, xx = np.where(ink)
    x_deg, y_deg = image_px_to_cartesian_deg(
        xx.astype(np.float64), yy.astype(np.float64), cfg
    )
    _ecc, theta = cartesian_to_polar(x_deg, y_deg)
    fu, fv, _ = forward_ink_cloud_w(img, params=params, render_cfg=cfg)
    return {
        "theta_span_deg": float(np.rad2deg(theta.max() - theta.min())),
        "du": float(fu.max() - fu.min()),
        "dv": float(fv.max() - fv.min()),
        "area": float((fu.max() - fu.min()) * (fv.max() - fv.min())),
        "u": fu,
        "v": fv,
        "img": img,
    }


def test_near_d_has_larger_cortex_footprint_than_far_d(tmp_path: Path):
    """Log-polar mag: 1° D at ~1.3° occupies more cortex than the same glyph at 3.5°.

    Shape stays D-like because w=k log(z+a) is conformal (local similarity).
    """
    cfg = _canonical_render_cfg()
    params, _ = _load_201118_params()
    far_x, far_y = _letter_d_pos_at_ecc(LETTER_D_FAR_ECC_DEG)
    near = _letter_cortical_span(
        tmp_path, LETTER_D_POS_X_DEG, LETTER_D_POS_Y_DEG, cfg, params
    )
    far = _letter_cortical_span(tmp_path, far_x, far_y, cfg, params)
    assert near["theta_span_deg"] > 2.0 * far["theta_span_deg"]
    assert near["du"] > 1.6 * far["du"]
    assert near["area"] > 3.0 * far["area"]
    # Local Jacobian is nearly isotropic, so bbox aspect stays similar.
    near_asp = near["dv"] / near["du"]
    far_asp = far["dv"] / far["du"]
    assert abs(near_asp - far_asp) < 0.35


def test_ayzenshtat_near_vs_far_same_cortical_scale(tmp_path: Path):
    """Near and far D on identical 2C axes so foveal magnification is visible."""
    import matplotlib.pyplot as plt

    cfg = _canonical_render_cfg()
    params, _ = _load_201118_params()
    far_x, far_y = _letter_d_pos_at_ecc(LETTER_D_FAR_ECC_DEG)
    cases = [
        ("near E≈1.35°", LETTER_D_POS_X_DEG, LETTER_D_POS_Y_DEG),
        ("far E=3.5°", far_x, far_y),
    ]
    # Shared (u,v) window from polar grid over 0–5°.
    polar_grid, _ = render_polar_degree_grid(
        cfg, ecc_spacing_deg=1.0, theta_step_deg=15.0
    )
    u_range, v_range = _cortical_window_for_vf(
        polar_grid, params, cfg, vf_max_deg=SHARED_CORTEX_VF_DEG
    )
    ecc_ticks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    theta_ticks = [0.0, -15.0, -30.0, -45.0, -60.0, -75.0, -90.0]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.4), layout="constrained")
    for c, (label, px, py) in enumerate(cases):
        letter, polar, _cart = _composite_grids_with_letter_d(
            tmp_path, cfg, pos_x=px, pos_y=py
        )
        polar = _stamp_1deg_box(polar, cfg, px, py)
        warped, _valid, ur, vr = sample_stimulus_onto_cortical_w(
            polar,
            params=params,
            render_cfg=cfg,
            grid_size=320,
            u_range=u_range,
            v_range=v_range,
            max_ecc_deg=QUADRANT_EXTENT_DEG * np.sqrt(2.0),
        )
        draw_vf_cartesian_deg(axes[0, c], polar, cfg)
        axes[0, c].plot(0.0, 0.0, "o", color="#c0392b", markersize=5, zorder=6)
        _annotate_letter_d(axes[0, c], px, py)
        apply_vf_hm_vm_degree_ticks(axes[0, c], cfg)
        axes[0, c].set_title(f"VF · {label}", fontsize=9)

        draw_schira_ayzenshtat_c(
            axes[1, c],
            warped,
            ur,
            vr,
            params,
            ecc_ticks_deg=ecc_ticks,
            theta_ticks_deg=theta_ticks,
            ecc_ref_deg=2.5,
        )
        lu, lv, _ = forward_ink_cloud_w(letter, params=params, render_cfg=cfg)
        axes[1, c].scatter(
            lu, lv, s=6.0, c="white", marker="s", linewidths=0, zorder=5
        )
        axes[1, c].set_title(f"C same (u,v) window · {label}", fontsize=9)
        np.testing.assert_allclose(ur, u_range)
        np.testing.assert_allclose(vr, v_range)

    fig.suptitle(
        "Same 1° D, same P, two eccentricities · identical cortical scale",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        "Schira is conformal locally: a 1° D keeps D-shape and mainly changes size "
        "(~4× cortical area near vs 3.5°). The Ayzenshtat monkey looks elongated "
        "because the whole photo spans many degrees, not because a small glyph shears. "
        "Yellow = 1° box. White = letter ink.",
        ha="center",
        fontsize=7,
        color="0.35",
    )
    out = _inspect_dir() / AYZENSHTAT_NEAR_FAR_PLOT_NAME
    fig.savefig(tmp_path / AYZENSHTAT_NEAR_FAR_PLOT_NAME, dpi=160, bbox_inches="tight")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    assert out.is_file()


def _load_ayzenshtat_monkey_rgb() -> np.ndarray:
    path = AYZENSHTAT_MONKEY_SRC
    if not path.is_file():
        alt = project_root() / "experiments/schira2007/_ayzenshtat_2012_monkey.png"
        path = alt if alt.is_file() else path
    if not path.is_file():
        raise FileNotFoundError(f"Ayzenshtat 2012 monkey screenshot not found: {path}")
    im = PILImage.open(path).convert("RGBA")
    bg = PILImage.new("RGB", im.size, (128, 128, 128))
    bg.paste(im, mask=im.split()[-1])
    return np.asarray(bg, dtype=np.uint8)


def _place_monkey_2_to_5deg(cfg: RenderConfig, monkey_rgb: np.ndarray) -> np.ndarray:
    """Paste the Fig. 2B photo into x∈[2,5], y∈[-5,-2] (lower-right hemifield).

    The paper crop has the fovea off the **top-right** (lower-left VF). Our
    monopole only maps Re(z+a)>0, so this is the matching patch with x>0,
    y<0. Horizontal flip keeps the fovea-ward side of the photo toward x=2.
    """
    canvas, _ = render_polar_degree_grid(
        cfg, ecc_spacing_deg=1.0, theta_step_deg=15.0
    )
    x_inner, x_outer = 2.0, 5.0
    y_near_hm, y_far = -2.0, -5.0
    x0_px, y_near_px = cartesian_deg_to_image_px(x_inner, y_near_hm, cfg)
    x1_px, y_far_px = cartesian_deg_to_image_px(x_outer, y_far, cfg)
    left = int(round(min(float(x0_px), float(x1_px))))
    right = int(round(max(float(x0_px), float(x1_px))))
    top = int(round(min(float(y_near_px), float(y_far_px))))
    bottom = int(round(max(float(y_near_px), float(y_far_px))))
    w_px = max(right - left, 1)
    h_px = max(bottom - top, 1)
    face = PILImage.fromarray(monkey_rgb).transpose(PILImage.FLIP_LEFT_RIGHT)
    face = face.resize((w_px, h_px), resample=PILImage.Resampling.LANCZOS)
    out = PILImage.fromarray(canvas)
    out.paste(face, (left, top))
    return np.asarray(out, dtype=np.uint8)


def _ayzenshtat_paper_params(a: float, k: float, alpha: float):
    """Paper (a, k, α) on the same double-sech + power fa as the YAML sets."""
    base, _ = _load_201118_params()
    params = replace(base, a=float(a), k=float(k), alpha=float(alpha))
    assert params.fa_combine == "power"
    assert params.shear == "double_sech"
    return params


def _warp_monkey_photo(vf: np.ndarray, cfg: RenderConfig, params):
    u_range, v_range = _cortical_window_for_deg_box(
        2.0, 5.0, -5.0, -2.0, params, margin_frac=0.12
    )
    return sample_stimulus_onto_cortical_w(
        vf,
        params=params,
        render_cfg=cfg,
        grid_size=360,
        u_range=u_range,
        v_range=v_range,
        max_ecc_deg=QUADRANT_EXTENT_DEG * np.sqrt(2.0),
    )


@pytest.mark.skipif(
    not AYZENSHTAT_MONKEY_SRC.is_file()
    and not (project_root() / "experiments/schira2007/_ayzenshtat_2012_monkey.png").is_file(),
    reason="Ayzenshtat 2012 monkey screenshot not on disk",
)
def test_ayzenshtat_2012_monkey_at_2_to_5deg(tmp_path: Path):
    """Paper monkey photo at 2–5° with both Ayzenshtat 2012 (a, k, α) sets."""
    import matplotlib.pyplot as plt

    cfg = RenderConfig(
        canvas_size=630,
        pixels_per_deg=105.0,
        quadrant_extent_deg=QUADRANT_EXTENT_DEG,
        background_gray=128,
        letter_box_deg=1.0,
    )
    params_c = _ayzenshtat_paper_params(*AYZENSHTAT_PAPER_MONKEY_C[1:])
    params_l = _ayzenshtat_paper_params(*AYZENSHTAT_PAPER_MONKEY_L[1:])
    monkey = _load_ayzenshtat_monkey_rgb()
    vf = _place_monkey_2_to_5deg(cfg, monkey)
    e_r, th_r = cartesian_to_polar(3.5, -3.5)
    assert np.isfinite(forward_schira(e_r, th_r, params_c))
    assert np.isfinite(forward_schira(e_r, th_r, params_l))

    warped_c, valid_c, ur_c, vr_c = _warp_monkey_photo(vf, cfg, params_c)
    warped_l, valid_l, ur_l, vr_l = _warp_monkey_photo(vf, cfg, params_l)
    assert valid_c.any() and valid_l.any()

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.6), layout="constrained")
    draw_vf_cartesian_deg(axes[0, 0], vf, cfg)
    axes[0, 0].plot(0.0, 0.0, "o", color="#c0392b", markersize=6, zorder=6)
    apply_vf_hm_vm_degree_ticks(axes[0, 0], cfg)
    axes[0, 0].set_title("A  VF · monkey in [2,5]×[-5,-2]°", fontsize=9)

    draw_vf_cartesian_deg(axes[0, 1], vf, cfg)
    axes[0, 1].set_xlim(1.5, 5.5)
    axes[0, 1].set_ylim(-5.5, -1.5)
    apply_vf_hm_vm_degree_ticks(axes[0, 1], cfg)
    axes[0, 1].set_title("B  stimulus zone (Fig. 2B)", fontsize=9)

    ecc_ticks = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    theta_ticks = [0.0, -15.0, -30.0, -45.0, -60.0, -75.0, -90.0]
    draw_schira_ayzenshtat_c(
        axes[1, 0],
        warped_c,
        ur_c,
        vr_c,
        params_c,
        ecc_ticks_deg=ecc_ticks,
        theta_ticks_deg=theta_ticks,
        ecc_ref_deg=3.5,
    )
    axes[1, 0].set_title(
        f"C  {AYZENSHTAT_PAPER_MONKEY_C[0]}  "
        f"a={params_c.a:g} k={params_c.k:g} α={params_c.alpha:g}",
        fontsize=9,
    )
    draw_schira_ayzenshtat_c(
        axes[1, 1],
        warped_l,
        ur_l,
        vr_l,
        params_l,
        ecc_ticks_deg=ecc_ticks,
        theta_ticks_deg=theta_ticks,
        ecc_ref_deg=3.5,
    )
    axes[1, 1].set_title(
        f"C  {AYZENSHTAT_PAPER_MONKEY_L[0]}  "
        f"a={params_l.a:g} k={params_l.k:g} α={params_l.alpha:g}",
        fontsize=9,
    )
    for ax in (axes[1, 0], axes[1, 1]):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] > xlim[1]
        assert ylim[0] > ylim[1]

    fig.suptitle(
        "Ayzenshtat 2012 monkey · 2–5° · paper (a, k, α) · fa=power (no camera affine)",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        "Paper 2B is lower-left (fovea at top-right of the crop). "
        "Photo is in the matching lower-right patch (x>0, y<0), h-flipped "
        "so the fovea-ward edge faces the fovea (contralateral V1). "
        f"fa = sech(P)**(sech_E·S2); S1={params_c.sech_ecc_k:g} S2={params_c.sech_amp:g}.",
        ha="center",
        fontsize=7,
        color="0.35",
    )
    out = _inspect_dir() / AYZENSHTAT_MONKEY_PLOT_NAME
    fig.savefig(tmp_path / AYZENSHTAT_MONKEY_PLOT_NAME, dpi=160, bbox_inches="tight")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    assert out.is_file()


def _fig2a_full_src() -> Path:
    path = AYZENSHTAT_FIG2A_FULL_SRC
    alt = project_root() / "experiments/schira2007/_ayzenshtat_2012_fig2a_full.png"
    if path.is_file():
        return path
    if alt.is_file():
        return alt
    return path


def _load_fig2a_full_rgb() -> np.ndarray:
    path = _fig2a_full_src()
    if not path.is_file():
        raise FileNotFoundError(f"Ayzenshtat Fig. 2A full frame not found: {path}")
    im = PILImage.open(path).convert("RGBA")
    bg = PILImage.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[-1])
    return np.asarray(bg, dtype=np.uint8)


def _fig2a_ppd(height_px: int) -> float:
    return (int(height_px) - 1) / AYZENSHTAT_FIG2A_VM_DEG


def _fig2a_deg_to_px(
    x_deg: np.ndarray,
    y_deg: np.ndarray,
    *,
    width_px: int,
    height_px: int,
    ppd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """VF degrees → pixels when (0,0) is the top-right corner."""
    x_px = np.asarray(x_deg, dtype=np.float64) * ppd + (int(width_px) - 1)
    y_px = -np.asarray(y_deg, dtype=np.float64) * ppd
    return x_px, y_px


def _fig2a_px_to_deg(
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    width_px: int,
    ppd: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_deg = (np.asarray(x_px, dtype=np.float64) - (int(width_px) - 1)) / ppd
    y_deg = -np.asarray(y_px, dtype=np.float64) / ppd
    return x_deg, y_deg


def _cortical_window_for_fig2a(rgb: np.ndarray, params, ppd: float, *, n: int = 80):
    h, w = rgb.shape[:2]
    ys = np.linspace(0.0, h - 1, int(n))
    xs = np.linspace(0.0, w - 1, int(n))
    xx, yy = np.meshgrid(xs, ys)
    x_deg, y_deg = _fig2a_px_to_deg(xx, yy, width_px=w, ppd=ppd)
    ecc, theta = cartesian_to_polar(x_deg.ravel(), y_deg.ravel())
    ww = forward_schira(ecc, theta, params)
    ok = np.isfinite(ww.real) & np.isfinite(ww.imag)
    u, v = ww.real[ok], ww.imag[ok]
    span = max(float(u.max() - u.min()), float(v.max() - v.min()), 1e-3)
    pad = 0.10 * span
    return (float(u.min() - pad), float(u.max() + pad)), (
        float(v.min() - pad),
        float(v.max() + pad),
    )


def _warp_fig2a_origin_top_right(rgb: np.ndarray, params, ppd: float, *, grid_size: int = 400):
    from scipy.ndimage import map_coordinates

    h, w = rgb.shape[:2]
    u_range, v_range = _cortical_window_for_fig2a(rgb, params, ppd)
    u0, u1 = u_range
    v0, v1 = v_range
    u_edges = np.linspace(u0, u1, grid_size + 1)
    v_edges = np.linspace(v1, v0, grid_size + 1)
    u_c = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_c = 0.5 * (v_edges[:-1] + v_edges[1:])
    uu, vv = np.meshgrid(u_c, v_c, indexing="xy")
    ecc, polar = inverse_schira(uu + 1j * vv, params, min_real_z=0.0)
    x_deg, y_deg = polar_to_cartesian(ecc, polar)
    x_px, y_px = _fig2a_deg_to_px(x_deg, y_deg, width_px=w, height_px=h, ppd=ppd)
    in_img = (
        np.isfinite(ecc)
        & (x_px >= -0.5)
        & (y_px >= -0.5)
        & (x_px < w - 0.5)
        & (y_px < h - 0.5)
    )
    warped = np.full((grid_size, grid_size, 3), 255, dtype=np.uint8)
    coords = np.vstack([y_px.ravel(), x_px.ravel()])
    for ch in range(3):
        sampled = map_coordinates(
            rgb[:, :, ch].astype(np.float64),
            coords,
            order=1,
            mode="constant",
            cval=255.0,
        ).reshape(grid_size, grid_size)
        plane = np.full((grid_size, grid_size), 255.0)
        plane[in_img] = sampled[in_img]
        warped[:, :, ch] = np.clip(np.rint(plane), 0, 255).astype(np.uint8)
    return warped, in_img, (u0, u1), (v0, v1)


def _draw_fig2a_vf(ax, rgb: np.ndarray, ppd: float) -> tuple[float, float]:
    h, w = rgb.shape[:2]
    x_left = -(w - 1) / ppd
    y_bottom = -(h - 1) / ppd
    ax.imshow(
        rgb,
        origin="upper",
        extent=[x_left, 0.0, y_bottom, 0.0],
        interpolation="nearest",
    )
    ax.set_xlim(x_left, 0.0)
    ax.set_ylim(y_bottom, 0.0)
    ax.set_aspect("equal")
    ax.plot(0.0, 0.0, "o", color="#c0392b", markersize=6, zorder=6)
    xt = np.arange(np.ceil(x_left), 0.01, 1.0)
    yt = np.arange(np.floor(y_bottom), 0.01, 1.0)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t:g}°" for t in xt], fontsize=8)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{t:g}°" for t in yt], fontsize=8)
    ax.set_xlabel("HM (left)", fontsize=9)
    ax.set_ylabel("VM (lower)", fontsize=9)
    return x_left, y_bottom


def _draw_schira_left_field_with_grid(
    ax,
    warped: np.ndarray,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    params,
    *,
    ecc_ticks: list[float],
    theta_ticks: list[float],
    ecc_ref_deg: float,
) -> None:
    u0, u1 = u_range
    v0, v1 = v_range
    ax.imshow(
        warped,
        origin="upper",
        extent=[u0, u1, v0, v1],
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_xlim(u1, u0)  # fovea (small u) on the right
    # Paper Fig. 2C: invert v so the left HM (top of the photo, θ≈−180°)
    # sits at the top and the lower VM (θ≈−90°) at the bottom. Default
    # imshow puts increasing row index downward, which is the opposite.
    ax.set_ylim(v1, v0)
    overlay_schira_polar_grid(
        ax, params, ecc_deg=ecc_ticks, theta_deg=theta_ticks
    )
    set_schira_axes_visual_deg(
        ax,
        params,
        ecc_ticks_deg=ecc_ticks,
        theta_ticks_deg=theta_ticks,
        ecc_ref_deg=ecc_ref_deg,
    )


@pytest.mark.skipif(
    not AYZENSHTAT_FIG2A_FULL_SRC.is_file()
    and not (project_root() / "experiments/schira2007/_ayzenshtat_2012_fig2a_full.png").is_file(),
    reason="Ayzenshtat Fig. 2A full frame not on disk",
)
def test_ayzenshtat_fig2a_full_lower_left_paper_params(tmp_path: Path):
    """Full Fig. 2A, origin top-right, unflipped lower-left, Schira-space polar grid."""
    import matplotlib.pyplot as plt

    rgb = _load_fig2a_full_rgb()
    h, w = rgb.shape[:2]
    ppd = _fig2a_ppd(h)
    params_c = _ayzenshtat_paper_params(*AYZENSHTAT_PAPER_MONKEY_C[1:])
    params_l = _ayzenshtat_paper_params(*AYZENSHTAT_PAPER_MONKEY_L[1:])

    x_left = -(w - 1) / ppd
    y_bottom = -(h - 1) / ppd
    e_ll, th_ll = cartesian_to_polar(x_left, y_bottom)
    assert np.isfinite(forward_schira(e_ll, th_ll, params_c))

    warped_c, valid_c, ur_c, vr_c = _warp_fig2a_origin_top_right(rgb, params_c, ppd)
    warped_l, valid_l, ur_l, vr_l = _warp_fig2a_origin_top_right(rgb, params_l, ppd)
    assert valid_c.any() and valid_l.any()

    ecc_ticks = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    theta_ticks = [-90.0, -105.0, -120.0, -135.0, -150.0, -165.0, -180.0]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.4), layout="constrained")
    _draw_fig2a_vf(axes[0], rgb, ppd)
    axes[0].set_title("A  VF · origin top-right · no extra grid", fontsize=9)

    _draw_schira_left_field_with_grid(
        axes[1],
        warped_c,
        ur_c,
        vr_c,
        params_c,
        ecc_ticks=ecc_ticks,
        theta_ticks=theta_ticks,
        ecc_ref_deg=3.5,
    )
    axes[1].set_title(
        f"C  {AYZENSHTAT_PAPER_MONKEY_C[0]}  "
        f"a={params_c.a:g} k={params_c.k:g} α={params_c.alpha:g}",
        fontsize=9,
    )
    _draw_schira_left_field_with_grid(
        axes[2],
        warped_l,
        ur_l,
        vr_l,
        params_l,
        ecc_ticks=ecc_ticks,
        theta_ticks=theta_ticks,
        ecc_ref_deg=3.5,
    )
    axes[2].set_title(
        f"C  {AYZENSHTAT_PAPER_MONKEY_L[0]}  "
        f"a={params_l.a:g} k={params_l.k:g} α={params_l.alpha:g}",
        fontsize=9,
    )
    for ax in (axes[1], axes[2]):
        assert ax.get_xlim()[0] > ax.get_xlim()[1]
        assert ax.get_ylim()[0] > ax.get_ylim()[1]

    fig.suptitle(
        "Ayzenshtat 2012 Fig. 2A full · lower-left · paper (a, k, α) · fa=power",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        "Unflipped VF: (0,0) is the top-right pixel. Scale: VM (image height) = "
        f"{AYZENSHTAT_FIG2A_VM_DEG:g}°, {ppd:.2f} px/deg, "
        f"x∈[{x_left:.2f},0], y∈[{y_bottom:.2f},0]. "
        "C: fovea right, v inverted (left HM / photo top at top of panel; "
        "LVM at bottom) to match paper Fig. 2C. "
        "Yellow = iso-E, blue = iso-θ. "
        f"S1={params_c.sech_ecc_k:g} S2={params_c.sech_amp:g}.",
        ha="center",
        fontsize=7,
        color="0.35",
    )
    out = _inspect_dir() / AYZENSHTAT_FIG2A_FULL_PLOT_NAME
    fig.savefig(tmp_path / AYZENSHTAT_FIG2A_FULL_PLOT_NAME, dpi=160, bbox_inches="tight")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    assert out.is_file()




