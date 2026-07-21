"""Parse ContrastCurve / Letters experiment summary into StimulusSpec rows."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.stimuli.catalog import (
    StimulusSpec,
    condition_label,
    h5_session_id,
)

CONTRAST_LETTERS_GLOB = "ContrastCurve_Letters_*_ExpSummary.csv"

# Columns after the shared metadata (0-based in the wide CSV).
_COND_COLS = {
    1: "Contrast (RGB)",
    2: "Unnamed: 7",
    3: "Unnamed: 8",
    4: "Unnamed: 9",
    5: "Unnamed: 10",
    6: "Unnamed: 11",
    7: "Unnamed: 12",
    8: "Unnamed: 13",
}


def _normalize_condition_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the wide CSV's condition columns onto Cond1..Cond8.

    The file has a two-line header: top-level ``Contrast (RGB)`` plus a second
    row that literally contains Cond1..Cond8 under Unnamed columns.
    """
    out = df.copy()
    # If a header row still exists in the body, use it to rename.
    for _, row in out.head(3).iterrows():
        renamed = {}
        for col in out.columns:
            val = row.get(col)
            if pd.isna(val):
                continue
            token = str(val).strip()
            m = re.match(r"(?i)cond\s*(\d+)$", token)
            if m:
                renamed[col] = f"Cond{int(m.group(1))}"
        if len(renamed) >= 6:
            out = out.rename(columns=renamed)
            # Drop the Cond1.. label row(s).
            mask = out.apply(
                lambda r: any(
                    isinstance(v, str) and re.match(r"(?i)cond\s*\d+$", v.strip())
                    for v in r.values
                    if pd.notna(v)
                ),
                axis=1,
            )
            out = out.loc[~mask].reset_index(drop=True)
            break
    else:
        # Fallback to positional names from a single-line header export.
        rename = {src: f"Cond{num}" for num, src in _COND_COLS.items() if src in out.columns}
        out = out.rename(columns=rename)
    return out


def contrast_letters_catalog_path(encoder_data_root: Path) -> Path | None:
    matches = sorted(encoder_data_root.glob(CONTRAST_LETTERS_GLOB))
    return matches[0] if matches else None


def _normalize_date(text: str) -> str:
    """Normalize dates like 23/5/18 or 13/06/2018 to D/M/YYYY."""
    day, month, year = (p.strip() for p in text.split("/"))
    y = int(year)
    if y < 100:
        y += 2000
    return f"{int(day)}/{int(month)}/{y}"


def _parse_target_location_swapped(text: str) -> tuple[float, float]:
    """
    New-file Target Location lists components reversed vs the shapes CSV.

    Example: ``(-0.75,+0.6)`` → old-style ``(0.6, -0.75)``.
    Also handles spaced signs like ``(- 0.9, +1)``.
    """
    nums: list[float] = []
    for match in re.finditer(r"([+-]?)\s*(\d+(?:\.\d+)?)", text):
        sign, digits = match.group(1), match.group(2)
        value = float(digits)
        if sign == "-":
            value = -value
        nums.append(value)
    if len(nums) < 2:
        raise ValueError(f"Could not parse Target Location from {text!r}")
    return nums[1], nums[0]


def _parse_filled_circle_radius_deg(text: str) -> float:
    match = re.search(r"r\s*=\s*(\d+(?:\.\d+)?)", str(text), flags=re.I)
    if not match:
        raise ValueError(f"Could not parse filled-circle radius from {text!r}")
    radius = float(match.group(1))
    # Downstream render treats size_deg as diameter.
    return radius * 2.0


def _parse_letter_size_deg(text: str) -> float:
    """Parse letter size; catalog value is the diameter of the letter circle (deg)."""
    match = re.search(r"(\d+(?:\.\d+)?)", str(text))
    if not match:
        raise ValueError(f"Could not parse letter size from {text!r}")
    return float(match.group(1))


def _parse_contrast_cell(text: str) -> tuple[str, int | None]:
    """
    Return ``(kind, rgb_gray)`` where kind is contrast|blank|error|skip.

    Examples: ``100 (249)``, ``Blank (186)``, ``Error``.
    """
    raw = str(text).strip()
    lower = raw.lower()
    if not raw or lower in {"nan"}:
        return "skip", None
    if "error" in lower:
        return "error", None
    if "blank" in lower:
        m = re.search(r"\((\d+)\)", raw)
        return "blank", int(m.group(1)) if m else None
    m = re.search(r"\((\d+)\)", raw)
    if not m:
        raise ValueError(f"Contrast cell missing RGB parentheses: {raw!r}")
    return "contrast", int(m.group(1))


def _enforce_contrast_polarity(
    gray: int,
    *,
    polarity: str,
    background: int,
    anchor: int | None,
) -> int:
    """
    Keep all targets on the paradigm side of the blank background.

    Some catalog rows (e.g. 13/06/2018 Cond2) list the opposite-polarity RGB by
    mistake; white sessions must stay ≥ background, black sessions ≤ background.
    """
    if polarity == "white":
        if gray < background:
            if anchor is not None and anchor >= background:
                return int(anchor)
            return 249
        return int(gray)
    if polarity == "black":
        if gray > background:
            if anchor is not None and anchor <= background:
                return int(anchor)
            return 0
        return int(gray)
    return int(gray)


def _letter_source_path(
    letters_root: Path, *, session_letter: str, letter: str
) -> Path:
    """Prefer root-level BMPs; fall back to session mats if needed."""
    letter = letter.upper()
    bmp = letters_root / f"{letter}.bmp"
    if bmp.exists():
        return bmp
    session_dir = {
        "c": letters_root / "2011C",
        "d": letters_root / "2011D",
    }.get(session_letter.lower())
    if session_dir is not None:
        candidate = session_dir / f"{letter}.mat"
        if candidate.exists():
            return candidate
    root_mat = letters_root / f"{letter}.mat"
    if root_mat.exists():
        return root_mat
    raise FileNotFoundError(
        f"No letter bmp/mat for {letter!r} (session={session_letter!r}) under {letters_root}"
    )


def parse_contrast_letters_rows(
    df: pd.DataFrame,
    *,
    monkey: str,
    letters_root: Path,
) -> list[StimulusSpec]:
    """
    Parse the wide ContrastCurve/Letters summary into StimulusSpec rows.

    Skips Cond6 Blank, Cond8 Error, and Control-attention paradigms.
    """
    # Drop the Cond1.. header row if present as row 0 with empty Date/Session.
    rows_out: list[StimulusSpec] = []
    current_date: str | None = None

    for _, row in df.iterrows():
        date_val = row.get("Date")
        if pd.notna(date_val) and str(date_val).strip():
            current_date = _normalize_date(str(date_val).strip())

        session_val = row.get("Session")
        paradigm_val = row.get("Paradigm")
        if pd.isna(session_val) or not str(session_val).strip():
            continue
        if pd.isna(paradigm_val) or not str(paradigm_val).strip():
            continue
        if current_date is None:
            raise ValueError(f"Session row missing date header: {session_val!r}")

        session_letter = str(session_val).strip().lower()
        paradigm = str(paradigm_val).strip()
        if "control attention" in paradigm.lower():
            continue

        loc_raw = row.get("Target Location (below HM; from VM)")
        if pd.isna(loc_raw) or not str(loc_raw).strip():
            raise ValueError(f"Missing Target Location for {current_date} {session_letter}")
        pos_x, pos_y = _parse_target_location_swapped(str(loc_raw))

        size_raw = row.get("Target Size (diameter in deg)")
        h5 = h5_session_id(current_date, session_letter)

        if "contrast curve" in paradigm.lower():
            diameter_deg = _parse_filled_circle_radius_deg(str(size_raw))
            polarity = "white" if "white" in paradigm.lower() else "black"
            # Background from Blank cell when present (screen mean luminance).
            blank_cell = row.get("Cond6")
            bg = 128
            if blank_cell is not None and pd.notna(blank_cell):
                kind, gray = _parse_contrast_cell(str(blank_cell))
                if kind == "blank" and gray is not None:
                    bg = gray

            # Cond1 anchors polarity when a later cell lists the opposite RGB.
            anchor: int | None = None
            cond1_cell = row.get("Cond1")
            if cond1_cell is not None and pd.notna(cond1_cell):
                kind1, gray1 = _parse_contrast_cell(str(cond1_cell))
                if kind1 == "contrast" and gray1 is not None:
                    anchor = gray1

            for cond_num in range(1, 9):
                col = f"Cond{cond_num}"
                if col not in row.index:
                    continue
                if cond_num in (6, 8):
                    continue
                cell = row.get(col)
                if pd.isna(cell) or not str(cell).strip():
                    continue
                kind, gray = _parse_contrast_cell(str(cell))
                if kind in {"blank", "error", "skip"}:
                    continue
                assert gray is not None
                gray = _enforce_contrast_polarity(
                    gray, polarity=polarity, background=bg, anchor=anchor
                )
                rgb = (gray, gray, gray)
                stim_text = (
                    f"cond{cond_num}: {paradigm} filled circle "
                    f"rgb=({gray},{gray},{gray})"
                )
                rows_out.append(
                    StimulusSpec(
                        monkey=monkey,
                        csv_date=current_date,
                        session_letter=session_letter,
                        h5_session=h5,
                        condition=condition_label(cond_num),
                        condition_num=cond_num,
                        stimulus_text=stim_text,
                        color=polarity,
                        shape_type="filled_circle",
                        size_deg=diameter_deg,
                        pos_x_deg=pos_x,
                        pos_y_deg=pos_y,
                        is_blank=False,
                        cortex_file=None,
                        rgb=rgb,
                        letter=None,
                        source_path=None,
                        background_gray=bg,
                    )
                )
            continue

        if "letter" in paradigm.lower():
            letter_size = _parse_letter_size_deg(str(size_raw))
            for cond_num in range(1, 9):
                col = f"Cond{cond_num}"
                if col not in row.index:
                    continue
                if cond_num in (6, 8):
                    continue
                cell = row.get(col)
                if pd.isna(cell) or not str(cell).strip():
                    continue
                token = str(cell).strip()
                if "blank" in token.lower() or "error" in token.lower():
                    continue
                if len(token) != 1 or not token.isalpha():
                    raise ValueError(
                        f"Expected letter in {col} for {current_date} {session_letter}, got {token!r}"
                    )
                letter = token.upper()
                mat_path = _letter_source_path(
                    letters_root, session_letter=session_letter, letter=letter
                )
                rows_out.append(
                    StimulusSpec(
                        monkey=monkey,
                        csv_date=current_date,
                        session_letter=session_letter,
                        h5_session=h5,
                        condition=condition_label(cond_num),
                        condition_num=cond_num,
                        stimulus_text=f"cond{cond_num}: letter {letter}",
                        color="white",
                        shape_type="letter",
                        size_deg=letter_size,
                        pos_x_deg=pos_x,
                        pos_y_deg=pos_y,
                        is_blank=False,
                        cortex_file=None,
                        rgb=None,
                        letter=letter,
                        source_path=str(mat_path),
                        background_gray=None,
                    )
                )
            continue

        # Unknown paradigm — ignore rather than fail the whole build.
        continue

    return rows_out


def load_contrast_letters_catalog(
    catalog_path: Path,
    *,
    monkey: str,
    letters_root: Path,
) -> pd.DataFrame:
    df = pd.read_csv(catalog_path)
    # Strip UTF-8 BOM on Date column if present.
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    df = _normalize_condition_columns(df)
    specs = parse_contrast_letters_rows(
        df, monkey=monkey, letters_root=letters_root
    )
    return pd.DataFrame([spec.__dict__ for spec in specs])
