"""Add 'radius' renderings of stimuli whose catalog size is labelled as a RADIUS but which the repo renderer draws as a
DIAMETER (configs/stimuli/default.yaml: assume_size_is_diameter: true).

Added (same response map, same condition, second image):
  100718a, 100718b  condAN3  triangle contour 0.4 radius  -> drawn with diameter 0.8 deg (circumscribed circle)
  100718a, 100718b  condAN4  circle contour   0.3 radius  -> drawn with diameter 0.6 deg
Images are rendered with the repo renderer (src/stimuli/render.py, assume_size_is_diameter=False) into
schira_dashboard/stimuli/<session>/<condition>__radius.png; the repo's Data/ images are not touched.
The catalog gets a column 'drawn_diameter_deg' (diameter actually drawn) and a 'variant' column ('as rendered' / 'radius').
Run after prepare_data.py (prepare_data.py calls it automatically).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO))
from src.stimuli.render import RenderConfig, render_stimulus   # noqa: E402
from src.stimuli.catalog import StimulusSpec                    # noqa: E402

VARIANTS = [("100718a", "condAN3"), ("100718a", "condAN4"), ("100718b", "condAN3"), ("100718b", "condAN4")]


def add_variants(monkey: str = "gandalf") -> pd.DataFrame:
    path = HERE / f"cache/{monkey}_catalog.csv"
    cat = pd.read_csv(path)
    cat = cat[cat.get("variant", pd.Series("as rendered", index=cat.index)).fillna("as rendered") == "as rendered"].copy()
    cat["variant"] = "as rendered"
    cat["drawn_diameter_deg"] = np.where(cat.shape_type.isin(["point", "filled_circle", "circle_contour", "triangle_contour"]),
                                         cat.size_deg, np.nan)
    new = []
    for s, c in VARIANTS:
        r = cat[(cat.session == s) & (cat.condition == c)]
        if r.empty:
            print(f"skip {s} {c}: not in catalog"); continue
        r = r.iloc[0]
        spec = StimulusSpec(monkey=monkey, csv_date="", session_letter=s[-1], h5_session=s, condition=c, condition_num=0,
                            stimulus_text=r.stimulus_text, color=r.color, shape_type=r.shape_type, size_deg=float(r.size_deg),
                            pos_x_deg=float(r.pos_x_deg), pos_y_deg=float(r.pos_y_deg), is_blank=False, cortex_file=None)
        img = render_stimulus(spec, RenderConfig(assume_size_is_diameter=False))
        out = HERE / f"stimuli/{s}/{c}__radius.png"; out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out)
        row = r.copy()
        row["variant"] = "radius"; row["drawn_diameter_deg"] = 2 * float(r.size_deg)
        row["image_path"] = str(out.relative_to(REPO))
        new.append(row)
        print(f"{s} {c}: {r.shape_type} size {r.size_deg} as radius -> diameter {2 * r.size_deg:.1f} deg -> {out.relative_to(REPO)}")
    cat = pd.concat([cat, pd.DataFrame(new)], ignore_index=True)
    cat.to_csv(path, index=False)
    return cat


if __name__ == "__main__":
    add_variants(sys.argv[1] if len(sys.argv) > 1 else "gandalf")
