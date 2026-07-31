from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.stimuli.contrast_letters_catalog import (
    _parse_target_location_swapped,
    parse_contrast_letters_rows,
)
from src.stimuli.render import RenderConfig, render_stimulus


def test_target_location_swap():
    assert _parse_target_location_swapped("(-0.75,+0.6)") == (0.6, -0.75)
    assert _parse_target_location_swapped("(-0.8, 1.7)") == (1.7, -0.8)
    assert _parse_target_location_swapped("(- 0.9, +1)") == (1.0, -0.9)


def test_parse_contrast_and_letters(tmp_path: Path):
    letters = tmp_path / "letters_stimuli"
    letters.mkdir()
    # Palette-like BMP: gray field with a black letter block near center.
    from PIL import Image as PILImage

    for letter in "GANDFL":
        arr = np.full((600, 800, 3), 188, dtype=np.uint8)
        arr[290:310, 390:410] = 0
        PILImage.fromarray(arr).save(letters / f"{letter}.bmp")

    df = pd.DataFrame(
        [
            {
                "Date": "23/5/18",
                "Session": np.nan,
                "Paradigm": np.nan,
                "Target Size (diameter in deg)": np.nan,
                "Target Location (below HM; from VM)": np.nan,
                "Cond1": np.nan,
                "Cond2": np.nan,
                "Cond3": np.nan,
                "Cond4": np.nan,
                "Cond5": np.nan,
                "Cond6": np.nan,
                "Cond7": np.nan,
                "Cond8": np.nan,
            },
            {
                "Date": np.nan,
                "Session": "b",
                "Paradigm": "Contrast curve - White",
                "Target Size (diameter in deg)": "filled circle (r=0.4 deg)",
                "Target Location (below HM; from VM)": "(-0.75,+0.6)",
                "Cond1": "100 (249)",
                "Cond2": "64 (229)",
                "Cond3": "32 (209)",
                "Cond4": "16 (198)",
                "Cond5": "8 (192)",
                "Cond6": "Blank (186)",
                "Cond7": "4 (190)",
                "Cond8": "Error",
            },
            {
                "Date": "20/11/2018",
                "Session": "a",
                "Paradigm": "Letters",
                "Target Size (diameter in deg)": "1 (letter size)",
                "Target Location (below HM; from VM)": "(- 0.9, +1)",
                "Cond1": "G",
                "Cond2": "A",
                "Cond3": "N",
                "Cond4": "D",
                "Cond5": "F",
                "Cond6": "Blank",
                "Cond7": "L",
                "Cond8": "Error",
            },
            {
                "Date": np.nan,
                "Session": "b",
                "Paradigm": "Control attention",
                "Target Size (diameter in deg)": "Circle contour 3 deg",
                "Target Location (below HM; from VM)": "(-0.75, 0.55)",
                "Cond1": "out",
                "Cond2": "out",
                "Cond3": "out",
                "Cond4": "in",
                "Cond5": np.nan,
                "Cond6": "Blank",
                "Cond7": "Error",
                "Cond8": np.nan,
            },
        ]
    )
    specs = parse_contrast_letters_rows(df, monkey="gandalf", letters_root=letters)
    # 6 contrast (skip blank+error); 201118a letters excluded (bad frames).
    assert len(specs) == 6

    contrast = [s for s in specs if s.h5_session == "230518b"]
    assert len(contrast) == 6
    assert contrast[0].rgb == (249, 249, 249)
    assert contrast[0].pos_x_deg == pytest.approx(0.6)
    assert contrast[0].pos_y_deg == pytest.approx(-0.75)
    assert contrast[0].size_deg == pytest.approx(0.8)  # r=0.4 → diameter
    assert contrast[0].background_gray == 128
    assert {s.condition for s in contrast} == {
        "condAN1",
        "condAN2",
        "condAN3",
        "condAN4",
        "condAN5",
        "condAN7",
    }
    assert all(s.rgb is not None and s.rgb[0] >= 186 for s in contrast)

    letters_specs = [s for s in specs if s.h5_session == "201118a"]
    assert len(letters_specs) == 0

    image = render_stimulus(contrast[0], RenderConfig())
    assert image.shape == (224, 224, 3)
    assert image.dtype == np.uint8
    # Canonical background gray 128 (session blank 186 used only for polarity).
    assert image[0, 0, 0] == 128
    # Target pixel near expected location should be near 249.
    assert (image == 249).any()
