"""Move session VSD maps onto the Schira anchor camera grid."""

from __future__ import annotations

import numpy as np

from src.session_register.transform import SessionTransform


def warp_target_to_anchor(
    target: np.ndarray,
    transform: SessionTransform | None,
    *,
    order: int = 1,
    cval: float = np.nan,
) -> np.ndarray:
    """Warp a moving-session map onto the anchor grid.

    ``transform`` maps **moving → fixed (anchor)**. Pass ``None`` (or an
    identity) when ``target`` is already on the anchor camera.
    """
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D target map, got {arr.shape}")
    if transform is None:
        return arr.copy()
    return transform.apply_to_map(arr, order=order, cval=cval).astype(np.float32)


def is_anchor_session(session: str, anchor_session: str) -> bool:
    """True when ``session`` is exactly the configured anchor (e.g. ``100718a``)."""
    return str(session).strip() == str(anchor_session).strip()
