"""Trial and catalog exclusions for stimulus / encoding pipeline."""

from __future__ import annotations

# 11.20.18 (h5 date prefix 201118):
# - 201118a: letters paradigm (historically dropped for bad VSD frames).
#   Re-included in the catalog / encoding pairs so those trials can train;
#   never use it as a Protocol A/C *test* fold — pass ``--train-only-dates
#   201118a`` or ``train_only_sessions`` in the held-out YAML.
# - 201118b: Control-attention (not letters); keep out of encoding / catalog.
# - 201118c / 201118d: letters; usable as train and as letter test folds.
EXCLUDED_H5_SESSIONS = frozenset({"201118b"})

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

    Excludes **all** trials from ``EXCLUDED_H5_SESSIONS`` (currently
    ``201118b``). Sessions ``201118a`` / ``201118c`` / ``201118d`` are kept
    in pairs; skip ``201118a`` as a *test* fold via train-only dates.

    ``condition`` and ``shape_type`` are accepted for call-site compatibility;
    session membership alone decides exclusion.
    """
    del condition, shape_type  # API compatibility; session id is sufficient.
    return date in EXCLUDED_H5_SESSIONS
