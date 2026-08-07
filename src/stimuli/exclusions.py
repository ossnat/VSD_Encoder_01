"""Trial and catalog exclusions for stimulus / encoding pipeline."""

from __future__ import annotations

# 11.20.18 (h5 date prefix 201118): only sessions c and d are usable.
# - 201118a: bad VSD frames (letters paradigm)
# - 201118b: excluded; do not use for encoding / catalog
EXCLUDED_H5_SESSIONS = frozenset({"201118a", "201118b"})

# Alias kept for letter-catalog filtering and older call sites.
EXCLUDED_LETTER_H5_SESSIONS = EXCLUDED_H5_SESSIONS

# Letter conditions historically present on 201118a
# (condAN6 blank / condAN8 error are never encoded).
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
    """True when an h5 session must not appear in the letter / stimulus catalog."""
    return h5_session in EXCLUDED_H5_SESSIONS


def is_excluded_encoding_trial(
    date: str, condition: str, *, shape_type: str | None = None
) -> bool:
    """
    Return True when a trial must not enter training, prediction, or encoding pairs.

    Excludes **all** trials from ``EXCLUDED_H5_SESSIONS`` (currently ``201118a``
    and ``201118b``). Sessions ``201118c`` / ``201118d`` are kept.

    ``condition`` and ``shape_type`` are accepted for call-site compatibility;
    session membership alone decides exclusion.
    """
    del condition, shape_type  # API compatibility; session id is sufficient.
    return date in EXCLUDED_H5_SESSIONS
