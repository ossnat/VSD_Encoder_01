from __future__ import annotations

from src.stimuli.exclusions import (
    EXCLUDED_201118A_LETTER_CONDITIONS,
    is_excluded_encoding_trial,
    is_excluded_letter_session,
)


def test_201118a_letter_session_excluded():
    assert is_excluded_letter_session("201118a")
    assert not is_excluded_letter_session("201118c")
    assert not is_excluded_letter_session("201118d")


def test_201118a_letter_trials_excluded():
    for cond in EXCLUDED_201118A_LETTER_CONDITIONS:
        assert is_excluded_encoding_trial("201118a", cond, shape_type="letter")
    assert not is_excluded_encoding_trial("201118c", "condAN1", shape_type="letter")
    assert not is_excluded_encoding_trial("201118d", "condAN1", shape_type="letter")
