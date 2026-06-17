from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stimuli.catalog import (
    condition_label,
    csv_date_to_h5_prefix,
    h5_session_id,
    parse_stimulus_rows,
)
from src.stimuli.render import (
    RenderConfig,
    _deg_point_to_px,
    _size_to_radius_px,
    render_stimulus,
)


def test_h5_session_mapping():
    assert csv_date_to_h5_prefix("27/6/2018") == "270618"
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
