"""Hourly minimal situations pipeline - step A (CPU, no LLM).

Builds situations straight from the dataset with a MINIMAL structure and expands
each base situation into 24 hourly copies (one per hour, `time_text`). The point
of the hourly expansion is that the same activity at 3am vs 2pm should let the
person decide differently later.

To keep LLM use to a minimum, this build invents NOTHING: no urgency, sensitivity,
user_busy, conditions or context_flags (the trainable model is assumed not to use
them). It only lays out the data structure:

  - action_input      : {action_text (human action; robot action after translate), activity}
  - context_input     : location, objects, time_of_day (per hour), user_state
  - time_text / hour  : the hour of the day
  - structured_task_features : {kind, quiet_hours}  (both free/deterministic)

Reuses the parsing helpers from build_situations_from_external (no duplication).
Default source is Charades; EPIC is available with --sources epic charades.

Next steps: translate_situations.py (robot actions) then
generate_training_data_hourly.py (free decision).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from iri_tfg_program import PROJECT_ROOT
from typing import Any, Iterator

from iri_tfg_program.situations.build_from_external import (
    DOMAIN_FEATURES,
    clean_charades_label,
    infer_activity,
    infer_user_state,
    load_charades_classes,
    normalize_scene,
    parse_all_nouns,
)
from iri_tfg_program.simulation.human_sim_preference_data import normalize_action_input, normalize_context_input

PROGRAM_DIR = PROJECT_ROOT
DEFAULT_RAW = PROGRAM_DIR / "data/external"
DEFAULT_OUTPUT = (
    PROGRAM_DIR / "data/generated" / "situations_hourly" / "situations_hourly.jsonl"
)
ALL_HOURS = list(range(24))


def hour_to_text(hour: int) -> str:
    suffix = "am" if hour < 12 else "pm"
    twelve = hour % 12 or 12
    return f"{twelve} {suffix}"


def time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def domain_kind(activity: str) -> str:
    return DOMAIN_FEATURES.get(activity, DOMAIN_FEATURES["daily living support"])[0]


def expand_hour(base: dict[str, Any], hour: int) -> dict[str, Any]:
    return {
        "situation_id": f"{base['base_id']}:h{hour:02d}",
        "source_dataset": base["source_dataset"],
        "hour": hour,                       # the time signal (top-level)
        "time_text": hour_to_text(hour),
        "action_input": normalize_action_input(
            {"action_text": base["action_text"], "activity": base["activity"]}
        ),
        # Slim context: only what the model conditions on.
        "context_input": {
            "location_current": base["location"] or "unknown",
            "objects_nearby": [str(o).strip() for o in base["objects"] if str(o).strip()],
            "user_state": list(base["user_state"]),
        },
        "scenario_rationale": base["scenario_rationale"],
        "source_metadata": base["source_metadata"],
    }


def iter_base_charades(
    charades_dir: Path, splits: list[str], classes: dict[str, str]
) -> Iterator[dict[str, Any]]:
    for split in splits:
        path = charades_dir / f"Charades_v1_{split}.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                scene = normalize_scene(str(row.get("scene", "")))
                objects = [
                    o.strip()
                    for o in str(row.get("objects", "")).split(";")
                    if o.strip() and o.strip().lower() != "none"
                ]
                script = str(row.get("script", "")).strip()
                actions_raw = str(row.get("actions", "")).strip()
                if not actions_raw:
                    continue
                seen: set[str] = set()
                for segment in actions_raw.split(";"):
                    parts = segment.split()
                    if not parts or parts[0] in seen:
                        continue
                    code = parts[0]
                    seen.add(code)
                    label = classes.get(code)
                    if not label:
                        continue
                    action_text = clean_charades_label(label)
                    yield {
                        "base_id": f"ext:charades:{row.get('id')}:{code}",
                        "source_dataset": "charades",
                        "action_text": action_text,
                        "activity": infer_activity(label),
                        "location": scene,
                        "objects": objects,
                        "user_state": infer_user_state(label),
                        "scenario_rationale": script or f"Home activity: {action_text}.",
                        "source_metadata": {
                            "video_id": row.get("id"),
                            "scene": row.get("scene"),
                            "charades_class": code,
                            "charades_label": label,
                            "objects": objects,
                            "split": split,
                        },
                    }


def iter_base_epic(epic_dir: Path, splits: list[str]) -> Iterator[dict[str, Any]]:
    for split in splits:
        path = epic_dir / f"EPIC_100_{split}.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                narration = str(row.get("narration", "")).strip()
                if not narration:
                    continue
                nouns = parse_all_nouns(str(row.get("all_nouns", "[]")))
                activity = infer_activity(" ".join([narration, " ".join(nouns)]))
                if activity == "daily living support":
                    activity = "meal support"
                yield {
                    "base_id": f"ext:epic:{row.get('narration_id')}",
                    "source_dataset": "epic_kitchens_100",
                    "action_text": narration,
                    "activity": activity,
                    "location": "kitchen",
                    "objects": nouns,
                    "user_state": [],
                    "scenario_rationale": f"Kitchen activity '{narration}'.",
                    "source_metadata": {
                        "narration_id": row.get("narration_id"),
                        "video_id": row.get("video_id"),
                        "verb": row.get("verb"),
                        "noun": row.get("noun"),
                        "all_nouns": nouns,
                        "split": split,
                    },
                }


def base_dedup_key(base: dict[str, Any]) -> tuple:
    return (
        base["source_dataset"],
        base["action_text"],
        base["activity"],
        base["location"],
        tuple(sorted(base["objects"])),
    )


def collect_bases(
    *,
    sources: list[str],
    raw_dir: Path,
    max_per_source: int | None,
    dedup: bool,
) -> tuple[list[dict[str, Any]], int]:
    epic_dir = raw_dir / "epic"
    charades_root = raw_dir / "charades"
    charades_dir = charades_root / "Charades"
    if not (charades_dir / "Charades_v1_train.csv").exists():
        charades_dir = charades_root

    bases: list[dict[str, Any]] = []
    dropped = 0
    for source in sources:
        if source == "charades":
            classes = load_charades_classes(charades_dir / "Charades_v1_classes.txt")
            iterator = iter_base_charades(charades_dir, ["train", "test"], classes)
        else:
            iterator = iter_base_epic(epic_dir, ["train", "validation"])
        seen: set[tuple] = set()
        kept = 0
        for base in iterator:
            if max_per_source is not None and kept >= max_per_source:
                break
            if dedup:
                key = base_dedup_key(base)
                if key in seen:
                    dropped += 1
                    continue
                seen.add(key)
            bases.append(base)
            kept += 1
    return bases, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build minimal situations expanded to 24 hours (no LLM)."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--sources", nargs="+", choices=["epic", "charades"], default=["charades"]
    )
    parser.add_argument(
        "--hours", type=int, nargs="+", default=ALL_HOURS,
        help="Hours (0-23) to expand each situation into (default: all 24).",
    )
    parser.add_argument(
        "--max-per-source", type=int, default=None,
        help="Cap base situations per source BEFORE the 24x hourly expansion.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hours = [h for h in args.hours if 0 <= h <= 23]
    if not hours:
        raise ValueError("--hours must contain values in 0..23.")
    raw_dir = args.raw_dir.resolve()

    bases, dropped = collect_bases(
        sources=args.sources,
        raw_dir=raw_dir,
        max_per_source=args.max_per_source,
        dedup=not args.no_dedup,
    )
    total = len(bases) * len(hours)

    print("")
    print("=" * 72)
    print("Build hourly minimal situations")
    print("=" * 72)
    print(f"Sources         : {', '.join(args.sources)}")
    print(f"Base situations : {len(bases)} (dropped dupes {dropped})")
    print(f"Hours per base  : {len(hours)} -> {[hour_to_text(h) for h in hours]}")
    print(f"Total situations: {total}")
    print(f"Output          : {args.output}")
    print(f"Fields dropped  : urgency, sensitivity, user_busy, conditions, context_flags")
    print("=" * 72)

    if args.dry_run:
        return
    if not bases:
        raise FileNotFoundError(f"No base situations found under {raw_dir}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    per_source: dict[str, int] = {}
    per_activity: dict[str, int] = {}
    written = 0
    with args.output.open("w", encoding="utf-8") as out:
        for base in bases:
            for hour in hours:
                situation = expand_hour(base, hour)
                out.write(json.dumps(situation, ensure_ascii=False) + "\n")
                written += 1
                per_source[base["source_dataset"]] = per_source.get(base["source_dataset"], 0) + 1
                act = base["activity"]
                per_activity[act] = per_activity.get(act, 0) + 1

    print("")
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Situations written : {written}")
    print("Per source:")
    for source, count in per_source.items():
        print(f"  {source:<12} : {count}")
    print("By activity domain:")
    for act, count in sorted(per_activity.items(), key=lambda kv: -kv[1]):
        print(f"  {act:<30} : {count}")
    print(f"Output : {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
