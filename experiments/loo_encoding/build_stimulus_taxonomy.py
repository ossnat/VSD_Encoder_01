#!/usr/bin/env python3
"""Build stimulus taxonomy table for LOO encoding.

Writes:
  experiments/loo_encoding/stimulus_taxonomy.yaml
  experiments/loo_encoding/stimulus_taxonomy.csv

Reuses roi_review inventory when present; otherwise rebuilds from encoding pairs
+ stimulus catalog fields.

Usage:
  scripts/py experiments/loo_encoding/build_stimulus_taxonomy.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.roi_mask import list_roi_stimulus_ids
from src.loo.folds import load_heldout_list
from src.paths import project_root
from src.stimuli.identity import attach_stimulus_ids

OUT_DIR = Path("experiments/loo_encoding")
REVIEW_INV_CSV = OUT_DIR / "roi_review" / "stimulus_inventory.csv"
REVIEW_INV_JSON = OUT_DIR / "roi_review" / "stimulus_inventory.json"
HELDOUT_YAML = OUT_DIR / "heldout_list.yaml"


def _load_inventory(repo: Path) -> pd.DataFrame:
    csv_path = repo / REVIEW_INV_CSV
    json_path = repo / REVIEW_INV_JSON
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    if json_path.is_file():
        with json_path.open() as f:
            return pd.DataFrame(json.load(f))
    raise FileNotFoundError(
        f"Missing inventory at {csv_path} (or JSON). "
        "Run scripts/16_propose_stimulus_rois.py first, or pass --pairs."
    )


def _from_pairs(pairs_path: Path) -> pd.DataFrame:
    pairs = pd.read_parquet(pairs_path)
    pairs = attach_stimulus_ids(pairs)
    rows: list[dict] = []
    for sid, g in pairs.dropna(subset=["stimulus_id"]).groupby("stimulus_id"):
        sessions = (
            g[["date", "condition"]]
            .drop_duplicates()
            .sort_values(["date", "condition"])
        )
        split_counts = g["split"].value_counts().to_dict()
        letter = g["letter"].iloc[0] if "letter" in g.columns else None
        if letter is not None and pd.isna(letter):
            letter = None
        rows.append(
            {
                "stimulus_id": sid,
                "shape_type": str(g["shape_type"].iloc[0]),
                "stimulus_text_example": str(g["stimulus_text"].iloc[0]),
                "color": str(g["color"].iloc[0]),
                "size_deg": float(g["size_deg"].iloc[0])
                if pd.notna(g["size_deg"].iloc[0])
                else None,
                "letter": None if letter is None else str(letter),
                "n_sessions": int(len(sessions)),
                "n_trials": int(len(g)),
                "n_train": int(split_counts.get("train", 0)),
                "n_val": int(split_counts.get("val", 0)),
                "n_test": int(split_counts.get("test", 0)),
                "dates_conditions": ";".join(
                    f"{r.date}/{r.condition}" for r in sessions.itertuples(index=False)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_taxonomy(
    inv: pd.DataFrame,
    *,
    heldout_ids: list[str],
    roi_ids: list[str],
) -> pd.DataFrame:
    held = set(heldout_ids)
    rois = set(roi_ids)
    df = inv.copy()
    df["heldout_candidate"] = df["stimulus_id"].astype(str).isin(held)
    # Preserve older inventory flag name if present.
    if "candidate_held_out" in df.columns:
        df["heldout_candidate"] = df["heldout_candidate"] | df[
            "candidate_held_out"
        ].astype(bool)
    df["has_roi"] = df["stimulus_id"].astype(str).isin(rois)
    df["roi_status"] = df["has_roi"].map(
        lambda x: "accepted" if x else "missing"
    )
    prefer = [
        "stimulus_id",
        "shape_type",
        "color",
        "size_deg",
        "letter",
        "n_sessions",
        "n_trials",
        "n_train",
        "n_val",
        "n_test",
        "dates_conditions",
        "heldout_candidate",
        "has_roi",
        "roi_status",
        "stimulus_text_example",
    ]
    cols = [c for c in prefer if c in df.columns] + [
        c for c in df.columns if c not in prefer
    ]
    out = df[cols].sort_values(
        ["heldout_candidate", "shape_type", "stimulus_id"],
        ascending=[False, True, True],
    )
    return out.reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="Optional encoding-pairs parquet (rebuild inventory from pairs)",
    )
    p.add_argument(
        "--heldout",
        type=Path,
        default=None,
        help="Held-out list YAML (default: experiments/loo_encoding/heldout_list.yaml)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    if args.pairs is not None:
        inv = _from_pairs(args.pairs if args.pairs.is_absolute() else repo / args.pairs)
    else:
        inv = _load_inventory(repo)

    heldout_path = args.heldout or (repo / HELDOUT_YAML)
    heldout_ids = load_heldout_list(heldout_path)
    roi_ids = list_roi_stimulus_ids(repo=repo)
    tax = build_taxonomy(inv, heldout_ids=heldout_ids, roi_ids=roi_ids)

    out_dir = repo / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stimulus_taxonomy.csv"
    yaml_path = out_dir / "stimulus_taxonomy.yaml"
    tax.to_csv(csv_path, index=False)
    payload = {
        "n_stimuli": int(len(tax)),
        "n_heldout_candidates": int(tax["heldout_candidate"].sum()),
        "n_with_roi": int(tax["has_roi"].sum()),
        "heldout_list": heldout_ids,
        "stimuli": tax.to_dict(orient="records"),
    }
    with yaml_path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    print(f"Wrote {csv_path.relative_to(repo)} ({len(tax)} stimuli)")
    print(f"Wrote {yaml_path.relative_to(repo)}")
    print(
        f"heldout_candidates={int(tax['heldout_candidate'].sum())} "
        f"with_roi={int(tax['has_roi'].sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
