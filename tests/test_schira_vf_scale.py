"""Stimulus scale for Schira: 35 px/deg, 1-degree letter box (stimulus renderer and catalog)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import yaml
from PIL import Image as PILImage
from src.paths import project_root, resolve_data_path
from src.retinotopy.visual_field import cartesian_deg_to_image_px, image_px_to_cartesian_deg, stimulus_ink_mask
from src.stimuli.catalog import StimulusSpec, stimulus_spec_from_mapping
from src.stimuli.render import RenderConfig, render_stimulus

LETTER_BOX_DEG = 1.0
CANONICAL_CANVAS = 210
SCHIRA_SET = "201118"
def _stimuli_yaml() -> dict:
    path = project_root() / "configs" / "stimuli" / "default.yaml"
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _render_config(stimuli_cfg: dict, *, canvas_size: int | None = None) -> RenderConfig:
    base_canvas = int(stimuli_cfg.get("canvas_size", 210))
    canvas = int(canvas_size) if canvas_size is not None else base_canvas
    quadrant_extent_deg = float(stimuli_cfg.get("quadrant_extent_deg", 6.0))
    # Keep degrees/pixel consistent with the base catalog canvas.
    base_ppd = stimuli_cfg.get("pixels_per_deg")
    if base_ppd is None:
        base_ppd = base_canvas / quadrant_extent_deg
    pixels_per_deg = float(base_ppd) * (canvas / base_canvas)
    return RenderConfig(
        canvas_size=canvas,
        pixels_per_deg=pixels_per_deg,
        quadrant_extent_deg=quadrant_extent_deg,
        background_gray=int(stimuli_cfg.get("background_gray", 128)),
        bar_length_deg=float(stimuli_cfg.get("bar_length_deg", 1.0)),
        bar_width_px=max(
            1, int(round(int(stimuli_cfg.get("bar_width_px", 1)) * canvas / base_canvas))
        ),
        contour_width_px=max(
            1,
            int(round(int(stimuli_cfg.get("contour_width_px", 1)) * canvas / base_canvas)),
        ),
        assume_size_is_diameter=bool(stimuli_cfg.get("assume_size_is_diameter", True)),
        draw_fixation=bool(stimuli_cfg.get("draw_fixation", False)),
        letter_box_deg=float(stimuli_cfg.get("letter_box_deg", 1.0)),
    )


def _canonical_render_cfg() -> RenderConfig:
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


def _encoder_data_root() -> Path | None:
    path = resolve_data_path("Data/EncoderData")
    return path if path.is_dir() else None


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
