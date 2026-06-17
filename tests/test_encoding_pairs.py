from __future__ import annotations

import pandas as pd

from src.encoding.pairs import dedupe_stimulus_manifest


def test_dedupe_stimulus_manifest_prefers_non_blank():
    df = pd.DataFrame(
        [
            {
                "h5_session": "240718c",
                "condition": "condAN6",
                "condition_num": 6,
                "is_blank": True,
                "stimulus_text": "Cond 6 Blank",
            },
            {
                "h5_session": "240718c",
                "condition": "condAN6",
                "condition_num": 6,
                "is_blank": False,
                "stimulus_text": "cond6: black bar horizontal",
            },
        ]
    )
    out = dedupe_stimulus_manifest(df)
    assert len(out) == 1
    assert out.iloc[0]["stimulus_text"] == "cond6: black bar horizontal"
