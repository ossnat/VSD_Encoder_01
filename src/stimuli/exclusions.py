"""Trial and catalog exclusions for stimulus / encoding pipeline."""

from __future__ import annotations

# 11.20.18 session a: bad VSD frames; exclude all letter trials from this session.
EXCLUDED_LETTER_H5_SESSIONS = frozenset({"201118a"})

# Letter conditions present in 201118a (condAN6 blank / condAN8 error are never encoded).
EXCLUDED_201118A_LETTER_CONDITIONS = frozenset(
    {
        "condAN1",  # G
        "condAN2",  # A
        "condAN3",  # N
        "condAN4",  # D
        "condAN5",  # F
        "condAN7",  # L
    }
)


def is_excluded_letter_session(h5_session: str) -> bool:
    return h5_session in EXCLUDED_LETTER_H5_SESSIONS


def is_excluded_encoding_trial(date: str, condition: str, *, shape_type: str | None = None) -> bool:
    """
    Return True when a trial must not enter training, prediction, or encoding pairs.

    Currently excludes letter trials from 201118a only; sessions 201118c/d are kept.
    """
    if date not in EXCLUDED_LETTER_H5_SESSIONS:
        return False
    if shape_type is not None and shape_type != "letter":
        return False
    if condition not in EXCLUDED_201118A_LETTER_CONDITIONS:
        return False
    return True
