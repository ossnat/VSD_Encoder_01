"""Write held-out trial r for none / schira / camera / all geometry modes.

Reads each mode's ``loo_summary.csv`` and writes one comparison table.

Example::

  scripts/py scripts/23_compare_schira_opt_modes.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.paths import project_root


def _load_summary(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict] = {}
    for row in rows:
        sid = str(row["heldout_stimulus_id"])
        out[sid] = row
    return out


def _r(row: dict | None) -> float:
    if row is None:
        return float("nan")
    return float(row["r_mean_test_masked"])


def main() -> None:
    repo = project_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=repo
        / "experiments/schira_encoding/gandalf/loo/win_0035_0046"
        / "vgg16_imagenet/block1_prepool/set-100718__anchor-100718a"
        / "global_channel_ridge",
    )
    args = p.parse_args()
    root = args.root if args.root.is_absolute() else repo / args.root

    paths = {
        "none": root / "protocol_B_noise_ceiling_hull" / "loo_summary.csv",
        "schira": root / "schira_opt_schira" / "loo_summary.csv",
        "camera": root / "schira_opt" / "loo_summary.csv",
        "all": root / "schira_opt_all" / "loo_summary.csv",
    }
    tables = {mode: _load_summary(path) for mode, path in paths.items()}
    ids = sorted(tables["none"])
    out_dir = root / "schira_opt_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "heldout_r_by_mode.csv"
    fieldnames = [
        "heldout_stimulus_id",
        "r_none",
        "r_schira",
        "r_camera",
        "r_all",
        "best_mode",
    ]
    rows_out = []
    means = {mode: 0.0 for mode in paths}
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sid in ids:
            values = {
                "none": _r(tables["none"].get(sid)),
                "schira": _r(tables["schira"].get(sid)),
                "camera": _r(tables["camera"].get(sid)),
                "all": _r(tables["all"].get(sid)),
            }
            for mode, value in values.items():
                means[mode] += value
            best = max(values, key=values.get)
            row = {
                "heldout_stimulus_id": sid,
                "r_none": f"{values['none']:.6f}",
                "r_schira": f"{values['schira']:.6f}",
                "r_camera": f"{values['camera']:.6f}",
                "r_all": f"{values['all']:.6f}",
                "best_mode": best,
            }
            w.writerow(row)
            rows_out.append((sid, values, best))
        n = len(ids)
        mean_row = {
            "heldout_stimulus_id": "MEAN",
            "r_none": f"{means['none'] / n:.6f}",
            "r_schira": f"{means['schira'] / n:.6f}",
            "r_camera": f"{means['camera'] / n:.6f}",
            "r_all": f"{means['all'] / n:.6f}",
            "best_mode": max(means, key=means.get),
        }
        w.writerow(mean_row)
    print(f"Wrote {out_csv}")
    print(
        f"mean r  none={means['none']/n:.3f}  schira={means['schira']/n:.3f}  "
        f"camera={means['camera']/n:.3f}  all={means['all']/n:.3f}"
    )


if __name__ == "__main__":
    main()
