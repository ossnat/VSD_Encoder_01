#!/usr/bin/env python3
"""Interactive megapixel VSD explorer (local Mac GUI).

Usage (from repo root)::

    scripts/py tools/space_conv/run_megapixel_explorer.py
"""

from __future__ import annotations

import argparse
import sys

from src.data.averaging import (
    NORMALIZATION_BASELINE_ZSCORE,
    NORMALIZATION_NONE,
    resolve_normalization,
)
from src.paths import project_root

from tools.space_conv.discovery import (
    condition_labels_with_stimuli,
    describe_workspace,
    list_conditions,
    list_monkeys,
    list_session_h5_files,
    list_trials_for_condition,
    resolve_session_h5_by_date,
    session_date_from_h5,
    stimulus_description_for_condition,
)
from tools.space_conv.explorer_gui import show_mosaic
from tools.space_conv.megapixel import (
    BASELINE_END,
    BASELINE_START,
    build_megapixel_stack,
)


def _prompt(text: str, default: str | None = None) -> str:
    if default is None:
        raw = input(f"{text}: ").strip()
        return raw
    raw = input(f"{text} [{default}]: ").strip()
    return raw if raw else default


def _choose_index(labels: list[str], prompt: str) -> int:
    if not labels:
        raise SystemExit(f"Nothing to choose for: {prompt}")
    width = len(str(len(labels)))
    for i, label in enumerate(labels, start=1):
        print(f"  {i:>{width}}. {label}")
    while True:
        raw = _prompt(prompt)
        try:
            idx = int(raw)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= idx <= len(labels):
            return idx - 1
        print(f"Choose 1..{len(labels)}")


def _choose_trial_or_all(trial_labels: list[str]) -> int | None:
    """Return trial index, or None for ALL."""
    labels = ["ALL trials in condition"] + trial_labels
    choice = _choose_index(labels, "Select trial (or ALL)")
    if choice == 0:
        return None
    return choice - 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Explore VSD megapixel (10×10 block-mean) traces locally."
    )
    p.add_argument("--monkey", default=None, help="Monkey name (default: gandalf)")
    p.add_argument("--session", default=None, help="Session date tag, e.g. 270618b")
    p.add_argument("--condition", default=None, help="Condition label, e.g. condAN1")
    p.add_argument(
        "--trial",
        default=None,
        help="Trial index in condition (0-based) or 'all'",
    )
    p.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Analysis window start (inclusive). Default 35.",
    )
    p.add_argument(
        "--end-frame-inclusive",
        type=int,
        default=None,
        help="Analysis window end (inclusive). Default 45 → code uses [start, end+1).",
    )
    p.add_argument(
        "--normalization",
        choices=["none", "zscore", "baseline_zscore", "raw", "zscore_baseline"],
        default=None,
        help="none | zscore (baseline_zscore). Default: prompt.",
    )
    p.add_argument(
        "--list-only",
        action="store_true",
        help="Print discovery info and exit (no GUI).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    print(describe_workspace(repo=repo))
    print()

    monkeys = list_monkeys(repo=repo)
    if not monkeys:
        print("No monkeys with session_*.h5 under ProcessedData.", file=sys.stderr)
        return 1

    monkey = args.monkey or _prompt("Monkey name", "gandalf")
    if monkey not in monkeys:
        print(f"Unknown monkey {monkey!r}. Available: {', '.join(monkeys)}")
        if args.monkey:
            return 1
        monkey = monkeys[_choose_index(monkeys, "Select monkey")]

    sessions = list_session_h5_files(monkey, repo=repo)
    if not sessions:
        print(f"No session_*.h5 under ProcessedData/{monkey}", file=sys.stderr)
        return 1

    session_labels = [f"{session_date_from_h5(p)}  ({p.name})" for p in sessions]
    if args.session:
        h5_path = resolve_session_h5_by_date(monkey, args.session, repo=repo)
        if h5_path is None:
            print(f"Session {args.session!r} not found.")
            return 1
    else:
        print(f"\nSessions for {monkey}:")
        sess_i = _choose_index(session_labels, "Select session")
        h5_path = sessions[sess_i]
    date = session_date_from_h5(h5_path)
    print(f"Using {h5_path}")

    conditions = list_conditions(h5_path)
    if not conditions:
        print("No conditions in H5 metadata.", file=sys.stderr)
        return 1
    if args.condition:
        if args.condition not in conditions:
            print(f"Condition {args.condition!r} not in {conditions}")
            return 1
        condition = args.condition
    else:
        print("\nConditions:")
        cond_labels = condition_labels_with_stimuli(
            h5_path, conditions, monkey=monkey, repo=repo
        )
        condition = conditions[_choose_index(cond_labels, "Select condition")]
    stim_desc = stimulus_description_for_condition(
        monkey, date, condition, repo=repo
    )
    if stim_desc:
        print(f"Stimulus: {stim_desc}")

    trials = list_trials_for_condition(h5_path, condition)
    if not trials:
        print(f"No trials for {condition}", file=sys.stderr)
        return 1
    trial_labels = [
        f"idx={t.trial_index_in_condition}  global_id={t.trial_global_id}  ({t.dataset_name})"
        for t in trials
    ]

    if args.list_only:
        stim_bit = f" ({stim_desc})" if stim_desc else ""
        print(f"\n{monkey} / {date} / {condition}{stim_bit}: {len(trials)} trials")
        for lab in trial_labels:
            print(f"  {lab}")
        return 0

    if args.trial is not None:
        if str(args.trial).lower() in {"all", "*"}:
            selected = trials
            trial_desc = "ALL"
        else:
            ti = int(args.trial)
            if not (0 <= ti < len(trials)):
                print(f"Trial index {ti} out of range 0..{len(trials)-1}")
                return 1
            selected = [trials[ti]]
            trial_desc = f"trial_index={ti}"
    else:
        print(f"\nTrials in {condition} ({len(trials)}):")
        choice = _choose_trial_or_all(trial_labels)
        if choice is None:
            selected = trials
            trial_desc = "ALL"
        else:
            selected = [trials[choice]]
            trial_desc = f"trial_index={trials[choice].trial_index_in_condition}"

    start = (
        args.start_frame
        if args.start_frame is not None
        else int(_prompt("Analysis start_frame (inclusive)", "35"))
    )
    end_inclusive = (
        args.end_frame_inclusive
        if args.end_frame_inclusive is not None
        else int(_prompt("Analysis end_frame (inclusive)", "45"))
    )
    end_exclusive = end_inclusive + 1
    print(f"Analysis window in code: [{start}, {end_exclusive})")

    if args.normalization is not None:
        norm_raw = args.normalization
    else:
        print("\nNormalization:")
        print("  1. None (raw)")
        print("  2. Z-scored (baseline [5, 26) per-pixel, then full trial)")
        nchoice = _prompt("Select normalization", "1")
        norm_raw = "none" if nchoice.strip() in {"1", ""} else "baseline_zscore"

    if norm_raw in {"zscore", "Z", "z"}:
        norm_raw = NORMALIZATION_BASELINE_ZSCORE
    normalization = resolve_normalization(norm_raw)
    if normalization not in {NORMALIZATION_NONE, NORMALIZATION_BASELINE_ZSCORE}:
        print(f"Unsupported normalization {normalization!r}")
        return 1

    print(
        f"\nLoading {len(selected)} trial(s), building megapixel stack "
        f"(normalization={normalization}, baseline=[{BASELINE_START}, {BASELINE_END})) ..."
    )
    stack = build_megapixel_stack(
        h5_path,
        selected,
        normalization=normalization,
    )
    if stim_desc:
        title = f"{monkey} / {date} / {condition} ({stim_desc}) / {trial_desc}"
    else:
        title = f"{monkey} / {date} / {condition} / {trial_desc}"
    print(
        f"Stack ready: grid={stack.grid_shape}, frames={stack.n_frames}, "
        f"mode={stack.mode}. Opening mosaic…"
    )
    show_mosaic(
        stack,
        start_frame=start,
        end_frame=end_exclusive,
        title=title,
        h5_path=h5_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
