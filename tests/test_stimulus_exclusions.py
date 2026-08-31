from __future__ import annotations

from src.stimuli.exclusions import (
    EXCLUDED_201118A_LETTER_CONDITIONS,
    EXCLUDED_H5_SESSIONS,
    is_excluded_encoding_trial,
    is_excluded_letter_session,
)


def test_excluded_sessions_are_b_only():
    assert EXCLUDED_H5_SESSIONS == frozenset({"201118b"})


def test_201118_bad_sessions_excluded_from_catalog():
    assert not is_excluded_letter_session("201118a")
    assert is_excluded_letter_session("201118b")
    assert not is_excluded_letter_session("201118c")
    assert not is_excluded_letter_session("201118d")


def test_201118_bad_sessions_excluded_from_encoding():
    for cond in EXCLUDED_201118A_LETTER_CONDITIONS | {"condAN1", "condAN2"}:
        assert not is_excluded_encoding_trial("201118a", cond, shape_type="letter")
        assert is_excluded_encoding_trial("201118b", cond, shape_type="letter")
        assert is_excluded_encoding_trial("201118b", cond, shape_type="point")
        assert is_excluded_encoding_trial("201118b", cond)
    assert not is_excluded_encoding_trial("201118c", "condAN1", shape_type="letter")
    assert not is_excluded_encoding_trial("201118d", "condAN1", shape_type="letter")
