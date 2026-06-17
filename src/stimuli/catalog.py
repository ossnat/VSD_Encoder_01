"""Parse stimulus description CSV files from Data/EncoderData."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

STIMULUS_COL = "stimulus (need to check r/d)"
POSITION_COL = "Stimulus Position"


@dataclass(frozen=True)
class StimulusSpec:
    monkey: str
    csv_date: str
    session_letter: str
    h5_session: str
    condition: str
    condition_num: int
    stimulus_text: str
    color: str
    shape_type: str
    size_deg: float | None
    pos_x_deg: float
    pos_y_deg: float
    is_blank: bool
    cortex_file: str | None


def monkey_catalog_path(encoder_data_root: Path, monkey: str) -> Path:
    """Find {Monkey}_VSDI_*.csv under EncoderData."""
    monkey_cap = monkey[:1].upper() + monkey[1:].lower()
    matches = sorted(encoder_data_root.glob(f"{monkey_cap}_VSDI_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No catalog CSV for monkey={monkey!r} under {encoder_data_root}"
        )
    return matches[0]


def csv_date_to_h5_prefix(csv_date: str) -> str:
    day, month, year = (p.strip() for p in csv_date.split("/"))
    return f"{int(day):02d}{int(month):02d}{str(year)[-2:]}"


def h5_session_id(csv_date: str, session_letter: str) -> str:
    return f"{csv_date_to_h5_prefix(csv_date)}{session_letter.lower()}"


def condition_label(condition_num: int) -> str:
    return f"condAN{condition_num}"


def _parse_position(text: str) -> tuple[float, float]:
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if len(nums) < 2:
        raise ValueError(f"Could not parse position from {text!r}")
    return nums[0], nums[1]


def _session_letter_from_cortex(cortex_file: str | None) -> str | None:
    if not cortex_file or not isinstance(cortex_file, str):
        return None
    base = cortex_file.strip().split("/")[-1]
    stem = base.rsplit(".", 1)[0]
    letter = stem[-1]
    return letter if letter.isalpha() else None


def _parse_stimulus_text(text: str, *, bar_length_deg: float) -> tuple[str, str, float | None, bool]:
    raw = text.strip()
    lower = raw.lower()
    if "blank" in lower:
        return "none", "blank", None, True

    color = "black"
    if "white" in lower:
        color = "white"
    elif "black" in lower:
        color = "black"

    size_deg: float | None = None
    size_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:deg(?:ree)?s?)?\s*(?:diameter|radius|d/r)?",
        lower,
    )
    if size_match:
        size_deg = float(size_match.group(1))
    elif re.search(r"(\d+(?:\.\d+)?)\s*d(?:iameter)?/r", lower):
        pass

    if "point" in lower:
        return color, "point", size_deg, False
    if "triangle" in lower and "contour" in lower:
        return color, "triangle_contour", size_deg, False
    if "circle" in lower and "contour" in lower:
        return color, "circle_contour", size_deg, False
    if "filled circle" in lower or ("circle" in lower and "contour" not in lower):
        return color, "filled_circle", size_deg, False
    if "bar vertical" in lower or "vertical bar" in lower:
        return color, "bar_vertical", bar_length_deg, False
    if "bar horizontal" in lower or "horizontal bar" in lower:
        return color, "bar_horizontal", bar_length_deg, False

    raise ValueError(f"Unrecognized stimulus description: {text!r}")


def parse_stimulus_rows(
    df: pd.DataFrame,
    *,
    monkey: str,
    bar_length_deg: float = 0.3,
) -> list[StimulusSpec]:
    """Expand grouped CSV rows into one StimulusSpec per condition."""
    rows: list[StimulusSpec] = []
    current_monkey = monkey
    current_date: str | None = None
    current_session_field: str | None = None
    current_position: tuple[float, float] | None = None
    current_cortex: str | None = None

    for _, row in df.iterrows():
        if pd.notna(row.get("Monkey")) and str(row["Monkey"]).strip():
            current_monkey = str(row["Monkey"]).strip().lower()

        if pd.notna(row.get("Date")) and str(row["Date"]).strip():
            current_date = str(row["Date"]).strip()
            current_session_field = (
                str(row["Session"]).strip() if pd.notna(row.get("Session")) else None
            )
            current_position = None
            current_cortex = None

        if pd.notna(row.get("cortex file")) and str(row["cortex file"]).strip():
            current_cortex = str(row["cortex file"]).strip()

        if pd.notna(row.get(POSITION_COL)) and str(row[POSITION_COL]).strip():
            current_position = _parse_position(str(row[POSITION_COL]))

        stim_text = row.get(STIMULUS_COL)
        if pd.isna(stim_text) or not str(stim_text).strip():
            continue

        if current_date is None or current_session_field is None:
            raise ValueError(f"Stimulus row missing session header: {stim_text!r}")
        if current_position is None:
            raise ValueError(
                f"Missing position for stimulus {stim_text!r} in session {current_date}"
            )

        raw_stim = str(stim_text).strip()
        cond_match = re.match(r"(?i)cond\s*(\d+)\s*:\s*(.*)$", raw_stim)
        if cond_match:
            cond_num = int(cond_match.group(1))
            desc = cond_match.group(2).strip()
        else:
            blank_match = re.match(r"(?i)cond\s*(\d+)\s+blank\s*$", raw_stim)
            if blank_match:
                cond_num = int(blank_match.group(1))
                desc = "blank"
            else:
                raise ValueError(f"Could not parse condition label from {stim_text!r}")

        session_letter = _session_letter_from_cortex(current_cortex)
        if session_letter is None:
            if "," in current_session_field:
                session_letter = current_session_field.split(",")[0].strip()
            else:
                session_letter = current_session_field.strip()

        color, shape_type, size_deg, is_blank = _parse_stimulus_text(
            desc, bar_length_deg=bar_length_deg
        )

        rows.append(
            StimulusSpec(
                monkey=current_monkey,
                csv_date=current_date,
                session_letter=session_letter.lower(),
                h5_session=h5_session_id(current_date, session_letter),
                condition=condition_label(cond_num),
                condition_num=cond_num,
                stimulus_text=str(stim_text).strip(),
                color=color,
                shape_type=shape_type,
                size_deg=size_deg,
                pos_x_deg=current_position[0],
                pos_y_deg=current_position[1],
                is_blank=is_blank,
                cortex_file=current_cortex,
            )
        )

    return rows


def load_stimulus_catalog(
    catalog_path: Path,
    *,
    monkey: str,
    bar_length_deg: float = 0.3,
) -> pd.DataFrame:
    df = pd.read_csv(catalog_path)
    specs = parse_stimulus_rows(df, monkey=monkey, bar_length_deg=bar_length_deg)
    return pd.DataFrame([spec.__dict__ for spec in specs])
