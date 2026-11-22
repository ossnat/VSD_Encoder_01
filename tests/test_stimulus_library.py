"""Stimulus rules agreed on 15/11/2026 (radius vs diameter, exact bars, straight triangle side, Excel polarity,
24/07 cond6 typo) and the stimulus library identity (stimulus_uid)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stimuli.catalog import StimulusSpec, parse_stimulus_rows, size_as_diameter
from src.stimuli.contrast_letters_catalog import parse_contrast_letters_rows
from src.stimuli.identity import stimulus_uid_from_row
from src.stimuli.render import RenderConfig, render_stimulus


def _spec(**kw):
    base = dict(monkey="gandalf", csv_date="", session_letter="a", h5_session="x", condition="condAN1", condition_num=1,
                stimulus_text="", color="black", shape_type="point", size_deg=0.1, pos_x_deg=0.6, pos_y_deg=-0.75,
                is_blank=False, cortex_file=None)
    base.update(kw)
    return StimulusSpec(**base)


@pytest.mark.parametrize("text,value,diam,unclear", [
    ("black triangle contour 0.4 radius", 0.4, 0.8, False),
    ("black circle contour 0.3 diameter", 0.3, 0.3, False),
    ("white filled circle 0.3 diameter/radius?", 0.3, 0.3, True),
    ("black point 0.1 d/r?", 0.1, 0.1, True),
])
def test_size_as_diameter(text, value, diam, unclear):
    d, note = size_as_diameter(text, value)
    assert d == pytest.approx(diam)
    assert ("unclear" in note) == unclear


def _shapes_df(rows):
    cols = ["Monkey", "Date", "Session", "cortex file", "stimulus (need to check r/d)", "Stimulus Position"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def test_radius_rows_are_stored_as_diameter_and_2407_cond6_bar_is_cond7():
    df = _shapes_df([
        ["Gandalf", "10/7/2018", "a", "f", "cond3: black triangle contour 0.4 radius", "(0.6,-0.75)"],
        [None, "24/7/2018", "a", "g", "Cond 6 Blank", "(0.6,-0.75)"],
        [None, None, None, None, "cond6: black bar horizontal", None],
    ])
    specs = parse_stimulus_rows(df, monkey="gandalf")
    tri = [s for s in specs if s.shape_type == "triangle_contour"][0]
    assert tri.size_deg == pytest.approx(0.8)
    bar = [s for s in specs if s.shape_type == "bar_horizontal"][0]
    assert bar.condition == "condAN7"
    blank = [s for s in specs if s.is_blank][0]
    assert blank.condition == "condAN6"


@pytest.mark.parametrize("shape", ["bar_vertical", "bar_horizontal"])
def test_bars_are_35_by_1_px_and_straight(shape):
    img = render_stimulus(_spec(shape_type=shape, size_deg=1.0), RenderConfig())
    ys, xs = np.nonzero(img[..., 0] != 128)
    along, across = (ys, xs) if shape == "bar_vertical" else (xs, ys)
    assert np.ptp(along) + 1 == 35 and np.ptp(across) + 1 == 1


@pytest.mark.parametrize("diam", [0.4, 0.8, 1.2])
def test_triangle_vertical_side_is_one_column(diam):
    img = render_stimulus(_spec(shape_type="triangle_contour", size_deg=diam), RenderConfig())
    ys, xs = np.nonzero(img[..., 0] != 128)
    left = xs.min()
    assert all(xs[ys == r].min() == left for r in range(ys.min(), ys.max() + 1))


def test_contrast_follows_excel_rgb_and_background(tmp_path):
    df = pd.DataFrame([
        {"Date": "13/06/2018", "Session": "c", "Paradigm": "Contrast curve - White",
         "Target Size (diameter in deg)": "filled circle (r=0.15 deg)", "Target Location (below HM; from VM)": "(-0.75,+0.6)",
         "Cond1": "100 (249)", "Cond2": "100 (0)", "Cond3": "32 (209)", "Cond4": "16 (198)", "Cond5": "8 (192)",
         "Cond6": "Blank (186)", "Cond7": "4 (190)", "Cond8": "Error"},
    ])
    specs = {s.condition: s for s in parse_contrast_letters_rows(df, monkey="gandalf", letters_root=tmp_path)}
    assert specs["condAN2"].rgb == (0, 0, 0) and specs["condAN2"].color == "black"
    assert specs["condAN1"].rgb == (249, 249, 249)
    assert specs["condAN6"].is_blank and specs["condAN6"].background_gray == 186
    assert all(s.background_gray == 186 for s in specs.values())
    assert specs["condAN1"].size_deg == pytest.approx(0.3)


def test_stimulus_uid():
    assert stimulus_uid_from_row(_spec(shape_type="triangle_contour", size_deg=0.8).__dict__) == \
        "triangle_contour__black__d0.8__x0.6_y-0.75__bg128"
    assert stimulus_uid_from_row(_spec(shape_type="bar_vertical", size_deg=1.0).__dict__) == \
        "bar_vertical__black__len1__x0.6_y-0.75__bg128"
    assert stimulus_uid_from_row(_spec(shape_type="filled_circle", size_deg=0.3, rgb=(158, 158, 158), background_gray=186).__dict__) == \
        "filled_circle__rgb158__d0.3__x0.6_y-0.75__bg186"
    assert stimulus_uid_from_row(_spec(shape_type="letter", letter="D", size_deg=1.0, pos_x_deg=1.0, pos_y_deg=-0.9).__dict__) == \
        "letter_D__x1_y-0.9__bg128"
    assert stimulus_uid_from_row(_spec(shape_type="blank", is_blank=True).__dict__) == "blank__bg128"
    # same picture, different session/condition -> same uid
    a = _spec(h5_session="100718a", condition="condAN1").__dict__; b = _spec(h5_session="270618b", condition="condAN1").__dict__
    assert stimulus_uid_from_row(a) == stimulus_uid_from_row(b)
