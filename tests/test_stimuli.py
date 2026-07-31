from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from src.stimuli.catalog import (
    condition_label,
    csv_date_to_h5_prefix,
    h5_session_id,
    parse_stimulus_rows,
)
from src.stimuli.render import (
    RenderConfig,
    _deg_point_to_px,
    _draw_triangle_contour,
    _size_to_radius_px,
    render_stimulus,
)


def test_h5_session_mapping():
    assert csv_date_to_h5_prefix("27/6/2018") == "270618"
    assert csv_date_to_h5_prefix("23/5/18") == "230518"
    assert h5_session_id("27/6/2018", "b") == "270618b"
    assert condition_label(1) == "condAN1"


def test_parse_session_block():
    df = pd.DataFrame(
        [
            {
                "Monkey": "Gandalf",
                "Date": "27/6/2018",
                "Session": "b",
                "cortex file": "gan_2018_06_27b.1",
                "stimulus (need to check r/d)": "cond1: black point 0.1 diameter",
                "Stimulus Position": "(0.6,-0.75)",
            },
            {
                "Monkey": np.nan,
                "Date": np.nan,
                "Session": np.nan,
                "cortex file": np.nan,
                "stimulus (need to check r/d)": "cond5: black bar vertical",
                "Stimulus Position": np.nan,
            },
        ]
    )
    specs = parse_stimulus_rows(df, monkey="gandalf", bar_length_deg=0.3)
    assert len(specs) == 2
    assert specs[0].h5_session == "270618b"
    assert specs[0].condition == "condAN1"
    assert specs[1].shape_type == "bar_vertical"
    assert specs[1].size_deg == 0.3


def test_parse_bar_length_from_csv():
    df = pd.DataFrame(
        [
            {
                "Monkey": "Gandalf",
                "Date": "10/7/2018",
                "Session": "a",
                "cortex file": "gan_2018_07_10a.1",
                "stimulus (need to check r/d)": "cond5: black bar vertical, 1deg",
                "Stimulus Position": "(0.6,-0.75)",
            },
            {
                "Monkey": np.nan,
                "Date": np.nan,
                "Session": np.nan,
                "cortex file": np.nan,
                "stimulus (need to check r/d)": "cond7: black bar horizontal 1deg",
                "Stimulus Position": np.nan,
            },
        ]
    )
    specs = parse_stimulus_rows(df, monkey="gandalf", bar_length_deg=0.3)
    assert len(specs) == 2
    assert specs[0].shape_type == "bar_vertical"
    assert specs[0].size_deg == 1.0
    assert specs[1].shape_type == "bar_horizontal"
    assert specs[1].size_deg == 1.0


def test_parse_multi_session_block_expands_all_letters():
    """Session='a,b' / 'a,b,c' applies every condition to each listed session."""
    df = pd.DataFrame(
        [
            {
                "Monkey": "Gandalf",
                "Date": "10/7/2018",
                "Session": "a,b",
                "cortex file": "gan_2018_07_10a.1",
                "stimulus (need to check r/d)": "cond1: black point 0.1 diameter",
                "Stimulus Position": "(0.6,-0.75)",
            },
            {
                "Monkey": np.nan,
                "Date": np.nan,
                "Session": np.nan,
                "cortex file": np.nan,
                "stimulus (need to check r/d)": "cond2: black point 0.05 diameter",
                "Stimulus Position": np.nan,
            },
            {
                "Monkey": np.nan,
                "Date": "24/7/2018",
                "Session": "a,b,c",
                "cortex file": "gan_2018_07_10a.1",
                "stimulus (need to check r/d)": "cond1: black point 0.1 diameter",
                "Stimulus Position": "(0.6,-0.75)",
            },
            {
                "Monkey": np.nan,
                "Date": np.nan,
                "Session": np.nan,
                # Cortex suffixes must NOT reassign conditions to a single session.
                "cortex file": "gan_2018_07_10b.1",
                "stimulus (need to check r/d)": "cond2: black point 0.95 radius",
                "Stimulus Position": "(0,0)",
            },
        ]
    )
    specs = parse_stimulus_rows(df, monkey="gandalf", bar_length_deg=0.3)

    july10 = [s for s in specs if s.csv_date == "10/7/2018"]
    assert {(s.h5_session, s.condition) for s in july10} == {
        ("100718a", "condAN1"),
        ("100718b", "condAN1"),
        ("100718a", "condAN2"),
        ("100718b", "condAN2"),
    }

    july24 = [s for s in specs if s.csv_date == "24/7/2018"]
    assert {(s.h5_session, s.condition) for s in july24} == {
        ("240718a", "condAN1"),
        ("240718b", "condAN1"),
        ("240718c", "condAN1"),
        ("240718a", "condAN2"),
        ("240718b", "condAN2"),
        ("240718c", "condAN2"),
    }
    cond2 = [s for s in july24 if s.condition == "condAN2"]
    assert all(s.pos_x_deg == 0.0 and s.pos_y_deg == 0.0 for s in cond2)


def test_point_size_ratio():
    """0.1 deg and 0.05 deg points should produce different radii."""
    cfg = RenderConfig()
    r1 = _size_to_radius_px(0.1, cfg)
    r2 = _size_to_radius_px(0.05, cfg)
    assert r1 > r2
    assert r1 / r2 == pytest.approx(2.0, rel=0.01)


def test_quadrant_degree_scale():
    """224 px canvas = 6 deg; 1 deg diameter is twice 0.5 deg."""
    ppd = 224.0 / 6.0
    cfg = RenderConfig(canvas_size=224, pixels_per_deg=ppd, quadrant_extent_deg=6.0)
    d1 = _size_to_radius_px(1.0, cfg) * 2
    d05 = _size_to_radius_px(0.5, cfg) * 2
    assert d1 == pytest.approx(ppd, rel=0.01)
    assert d05 == pytest.approx(ppd / 2.0, rel=0.01)
    assert d1 / d05 == pytest.approx(2.0, rel=0.01)


def test_quadrant_position_from_fixation():
    ppd = 224.0 / 6.0
    cfg = RenderConfig(canvas_size=224, pixels_per_deg=ppd, quadrant_extent_deg=6.0)
    x_px, y_px = _deg_point_to_px(0.6, -0.75, cfg)
    assert x_px == pytest.approx(0.6 * ppd, rel=0.01)
    assert y_px == pytest.approx(0.75 * ppd, rel=0.01)


def test_render_white_stimulus_rgb():
    df = pd.DataFrame(
        [
            {
                "Monkey": "Gandalf",
                "Date": "29/5/2018",
                "Session": "a",
                "cortex file": "gan_2018_05_29a.1",
                "stimulus (need to check r/d)": "cond1: white point 0.1 diameter",
                "Stimulus Position": "(0.6,-0.75)",
            }
        ]
    )
    spec = parse_stimulus_rows(df, monkey="gandalf")[0]
    assert spec.color == "white"
    image = render_stimulus(spec, RenderConfig())
    unique = np.unique(image.reshape(-1, 3), axis=0)
    assert [255, 0, 0] not in unique.tolist()
    assert [255, 255, 255] in unique.tolist()


def test_render_stimulus_shape():
    df = pd.DataFrame(
        [
            {
                "Monkey": "Gandalf",
                "Date": "27/6/2018",
                "Session": "b",
                "cortex file": "gan_2018_06_27b.1",
                "stimulus (need to check r/d)": "cond1: black point 0.1 diameter",
                "Stimulus Position": "(0.6,-0.75)",
            }
        ]
    )
    spec = parse_stimulus_rows(df, monkey="gandalf")[0]
    image = render_stimulus(spec, RenderConfig())
    assert image.shape == (224, 224, 3)
    assert image.dtype == np.uint8


def test_triangle_contour_tip_points_right():
    """Equilateral triangle tip should point right (+x), not up."""
    cfg = RenderConfig(canvas_size=224, pixels_per_deg=224.0 / 6.0)
    spec = parse_stimulus_rows(
        pd.DataFrame(
            [
                {
                    "Monkey": "Gandalf",
                    "Date": "10/7/2018",
                    "Session": "a",
                    "cortex file": "gan_2018_07_10a.1",
                    "stimulus (need to check r/d)": "cond3: black triangle contour 0.4 radius",
                    "Stimulus Position": "(0.6,-0.75)",
                }
            ]
        ),
        monkey="gandalf",
    )[0]
    img = Image.new("RGB", (cfg.canvas_size, cfg.canvas_size), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    _draw_triangle_contour(draw, spec, cfg)
    arr = np.asarray(img)
    dark = arr[:, :, 0] < 64
    ys, xs = np.where(dark)
    x_px, y_px = _deg_point_to_px(spec.pos_x_deg, spec.pos_y_deg, cfg)
    radius_px = _size_to_radius_px(spec.size_deg, cfg)
    # Tip at 0° sits at center_x + radius; left vertices are left of center.
    assert float(xs.max()) == pytest.approx(x_px + radius_px, abs=2.0)
    assert float(xs.min()) < x_px - radius_px * 0.3
    assert float(ys.mean()) == pytest.approx(y_px, abs=2.0)
