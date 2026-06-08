"""Build a situations dataset from external public datasets.

Maps real-world activity annotations into the *situation* side of the pipeline
schema (everything about the situation; nothing about the user). Situations are
emitted **as-is, with no preference anchoring** - they are not steered toward any
decision. The synthetic human decides freely later, in generate_training_data.py.

Typical flow:
  build_situations_from_external.py  (this script, CPU)  -> raw situations
  translate_situations.py            (Qwen/GPU)          -> robot-action situations
  generate_training_data.py          (Qwen/GPU)          -> training samples

Sources (only lightweight, openly-downloadable text annotations are used; no
video/image pixels):

  * EPIC-Kitchens-100  - 77k kitchen action segments (verb/noun/objects).
      https://github.com/epic-kitchens/epic-kitchens-100-annotations  (CC BY-NC 4.0)
  * Charades           - ~9.8k home videos, 157 daily-living activity classes,
      real room labels and object lists, across the whole house.
      https://prior.allenai.org/projects/charades  (Charades license, non-commercial)

What is intentionally NOT emitted: preference_snapshot, label_action, any
anchoring or profile/personality. Those are user-side. The ``action_text`` here
is still the *human* activity; translate_situations.py rewrites it into a robot
assistance action.

Reuses v1 normalizers so the emitted inputs are byte-compatible with the rest of
the pipeline.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterator

from human_sim_preference_data import (
    normalize_action_input,
    normalize_context_input,
    normalize_structured_features,
)

PROGRAM_DIR = Path(__file__).resolve().parent

# Self-contained activity-domain keyword map (the project taxonomy no longer
# ships DOMAIN_KEYWORDS). Ordered: first matching domain wins, so the most
# specific / safety-relevant domains are listed first.
ACTIVITY_KEYWORDS: list[tuple[str, set[str]]] = [
    ("medication support", {"medicine", "medication", "pill", "pills", "tablet"}),
    ("meal support", {
        "food", "meal", "breakfast", "lunch", "dinner", "cook", "cooking", "stove",
        "sandwich", "eat", "eating", "drink", "drinking", "cup", "glass", "bottle",
        "mug", "dish", "dishes", "plate", "bowl", "spoon", "fork", "pour", "pouring",
        "fridge", "refrigerator", "groceries", "kitchen",
    }),
    ("cognitive or leisure support", {
        "book", "read", "reading", "laptop", "computer", "paper", "notebook",
        "work", "working", "study", "game", "puzzle", "draw", "sketch", "write",
        "writing",
    }),
    ("social support", {
        "phone", "camera", "call", "talking", "talk", "conversation", "friend",
        "family", "guest", "guests", "laughing", "laugh", "television", "tv",
        "smiling", "smile", "picture", "photo",
    }),
    ("mobility support", {
        "walk", "walking", "doorway", "stairs", "run", "running", "stand",
        "standing", "sit", "sitting", "lying", "lie", "exercise", "stretch",
        "mobility", "doorknob", "awakening", "dressing", "undressing", "shoes",
    }),
    ("housekeeping", {
        "clean", "cleaning", "tidy", "tidying", "laundry", "wash", "washing",
        "broom", "vacuum", "sweep", "sweeping", "towel", "dust", "dusting",
        "closet", "cabinet", "shelf", "organize", "organizing", "fold", "folding",
        "blanket", "pillow", "mirror", "window", "door",
    }),
]
DEFAULT_RAW = PROGRAM_DIR / "external_datasets" / "raw"
DEFAULT_OUTPUT = (
    PROGRAM_DIR / "generated_data" / "situations_external" / "external_situations.jsonl"
)

# Map an inferred activity domain to coarse structured task features. The robot
# can intervene in all of these; sensitivity is highest for medication.
DOMAIN_FEATURES = {
    "meal support":                 ("meal_support",       "medium", "low"),
    "housekeeping":                 ("housekeeping",       "low",    "low"),
    "medication support":           ("medication_support", "medium", "high"),
    "mobility support":             ("mobility_support",   "medium", "medium"),
    "social support":               ("social_support",     "low",    "low"),
    "cognitive or leisure support": ("cognitive_support",  "low",    "low"),
    "daily living support":         ("daily_living_support", "medium", "medium"),
}

# Light situational body-state cues (these describe the situation, not a
# preference). Keyed by substrings found in the activity text.
USER_STATE_CUES = {
    "lying": "lying_down",
    "sitting on the floor": "on_floor",
    "lying on the floor": "on_floor",
    "in a bed": "in_bed",
    "in bed": "in_bed",
    "on a bed": "in_bed",
    "awakening": "waking_up",
    "sitting": "seated",
    "standing": "standing",
    "walking": "walking",
    "running": "moving_fast",
    "cooking": "active",
    "dressing": "dressing",
    "undressing": "dressing",
}


# ──────────────────────────────────────────────────────────────────────────────
# Activity / feature inference (shared)
# ──────────────────────────────────────────────────────────────────────────────
_WORD = re.compile(r"[a-z]+")


def _stem(word: str) -> str:
    """Crude singular form so 'tablets'->'tablet', 'dishes'->'dishe' etc."""
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def infer_activity(text: str) -> str:
    # Token (word-boundary) matching, NOT substring: avoids 'pill' in 'pillow'.
    tokens = {_stem(word) for word in _WORD.findall(text.lower())}
    for domain, keywords in ACTIVITY_KEYWORDS:
        if any(_stem(keyword) in tokens for keyword in keywords):
            return domain
    return "daily living support"


def features_for(activity: str) -> dict[str, Any]:
    kind, urgency, sensitivity = DOMAIN_FEATURES.get(
        activity, DOMAIN_FEATURES["daily living support"]
    )
    return normalize_structured_features(
        {
            "kind": kind,
            "urgency": urgency,
            "sensitivity": sensitivity,
            "user_busy": False,
            "quiet_hours": False,
            "conditions": [],
            "context_flags": {},
        }
    )


def infer_user_state(text: str) -> list[str]:
    lower = text.lower()
    states: list[str] = []
    for cue, state in USER_STATE_CUES.items():
        if cue in lower and state not in states:
            states.append(state)
    return states


def make_situation(
    *,
    situation_id: str,
    source_dataset: str,
    action_text: str,
    activity: str,
    location: str,
    objects: list[str],
    user_state: list[str],
    scenario_rationale: str,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    context = normalize_context_input(
        {
            "location_current": location or None,
            "objects_nearby": objects,
            "available_objects": sorted(set(objects)),
            "raw_conditions": [],
            "time_of_day": "unknown",
            "weekday": "unknown",
            "user_state": user_state,
            "environment_flags": [],
        },
        day="00",
    )
    return {
        "situation_id": situation_id,
        "source_dataset": source_dataset,
        "action_input": normalize_action_input(
            {"action_text": action_text, "activity": activity}
        ),
        "context_input": context,
        "structured_task_features": features_for(activity),
        "scenario_rationale": scenario_rationale.strip(),
        "source_metadata": source_metadata,
    }


# ──────────────────────────────────────────────────────────────────────────────
# EPIC-Kitchens-100
# ──────────────────────────────────────────────────────────────────────────────
def parse_all_nouns(raw: str) -> list[str]:
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [token.strip() for token in raw.strip("[]").replace("'", "").split(",") if token.strip()]


def iter_epic(epic_dir: Path, splits: list[str]) -> Iterator[dict[str, Any]]:
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
                verb = str(row.get("verb", "")).strip()
                # EPIC is kitchen-only: default unmatched actions to meal support.
                activity = infer_activity(" ".join([narration, " ".join(nouns)]))
                if activity == "daily living support":
                    activity = "meal support"
                yield make_situation(
                    situation_id=f"ext:epic:{row.get('narration_id')}",
                    source_dataset="epic_kitchens_100",
                    action_text=narration,
                    activity=activity,
                    location="kitchen",
                    objects=nouns,
                    user_state=[],
                    scenario_rationale=(
                        f"Kitchen activity '{narration}' ({verb}); a home robot could "
                        "assist, remind, or stay passive depending on the user."
                    ),
                    source_metadata={
                        "narration_id": row.get("narration_id"),
                        "video_id": row.get("video_id"),
                        "verb": verb,
                        "noun": row.get("noun"),
                        "all_nouns": nouns,
                        "start_timestamp": row.get("start_timestamp"),
                        "stop_timestamp": row.get("stop_timestamp"),
                        "split": split,
                    },
                )


# ──────────────────────────────────────────────────────────────────────────────
# Charades
# ──────────────────────────────────────────────────────────────────────────────
def load_charades_classes(classes_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not classes_path.exists():
        return mapping
    for line in classes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        code, _, label = line.partition(" ")
        mapping[code] = label.strip()
    return mapping


def clean_charades_label(label: str) -> str:
    text = label.strip()
    for prefix in ("Someone is ", "Someone ", "A person ", "Person "):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    # Lowercase fully so Charades action_text matches EPIC's lowercase style.
    return text.lower()


def iter_charades(
    charades_dir: Path,
    splits: list[str],
    classes: dict[str, str],
) -> Iterator[dict[str, Any]]:
    for split in splits:
        path = charades_dir / f"Charades_v1_{split}.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                scene = str(row.get("scene", "")).strip()
                objects = [
                    obj.strip()
                    for obj in str(row.get("objects", "")).split(";")
                    if obj.strip() and obj.strip().lower() != "none"
                ]
                script = str(row.get("script", "")).strip()
                actions_raw = str(row.get("actions", "")).strip()
                if not actions_raw:
                    continue
                seen_classes: set[str] = set()
                for seg_idx, segment in enumerate(actions_raw.split(";")):
                    parts = segment.split()
                    if not parts:
                        continue
                    code = parts[0]
                    if code in seen_classes:  # one situation per distinct activity/video
                        continue
                    seen_classes.add(code)
                    label = classes.get(code)
                    if not label:
                        continue
                    action_text = clean_charades_label(label)
                    # Infer the domain from the activity label ONLY; the video's
                    # full object list would contaminate unrelated segments.
                    activity = infer_activity(label)
                    yield make_situation(
                        situation_id=f"ext:charades:{row.get('id')}:{code}",
                        source_dataset="charades",
                        action_text=action_text,
                        activity=activity,
                        location=normalize_scene(scene),
                        objects=objects,
                        user_state=infer_user_state(label),
                        scenario_rationale=(
                            (script or f"Home activity: {action_text}.")
                            + " A home robot could assist, remind, or stay passive "
                            "depending on the user."
                        ),
                        source_metadata={
                            "video_id": row.get("id"),
                            "scene": scene,
                            "charades_class": code,
                            "charades_label": label,
                            "objects": objects,
                            "split": split,
                        },
                    )


def normalize_scene(scene: str) -> str:
    text = scene.strip().lower()
    if not text:
        return "unknown"
    text = text.split("(")[0]   # drop parenthetical description
    text = text.split("/")[0]   # take the first alternative
    return text.strip() or "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a grounded situations dataset (situation-only fields) from "
            "EPIC-Kitchens-100 and Charades annotations."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["epic", "charades"],
        default=["epic", "charades"],
    )
    parser.add_argument("--epic-splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--charades-splits", nargs="+", default=["train", "test"])
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Optional cap on situations emitted per source.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Keep exact-duplicate situations (default removes them).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dedup_key(situation: dict[str, Any]) -> tuple:
    action = situation["action_input"]
    context = situation["context_input"]
    return (
        situation["source_dataset"],
        action.get("action_text"),
        action.get("activity"),
        context.get("location_current"),
        tuple(sorted(context.get("available_objects", []))),
    )


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    epic_dir = raw_dir / "epic"
    charades_root = raw_dir / "charades"
    # Charades.zip extracts into a nested "Charades/" folder.
    charades_dir = charades_root / "Charades"
    if not (charades_dir / "Charades_v1_train.csv").exists():
        charades_dir = charades_root

    print("")
    print("=" * 72)
    print("Build external situations dataset")
    print("=" * 72)
    print(f"Raw dir   : {raw_dir}")
    print(f"Sources   : {', '.join(args.sources)}")
    print(f"Output    : {args.output}")
    print(f"Dedup     : {not args.no_dedup}")
    print(f"Max/source: {args.max_per_source}")
    print("=" * 72)

    if args.dry_run:
        return

    iterators: list[tuple[str, Iterator[dict[str, Any]]]] = []
    if "epic" in args.sources:
        iterators.append(("epic", iter_epic(epic_dir, args.epic_splits)))
    if "charades" in args.sources:
        classes = load_charades_classes(charades_dir / "Charades_v1_classes.txt")
        iterators.append(
            ("charades", iter_charades(charades_dir, args.charades_splits, classes))
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple] = set()
    per_source: dict[str, int] = {}
    activity_counts: dict[str, int] = {}
    location_counts: dict[str, int] = {}
    dropped_dupes = 0
    total = 0

    with args.output.open("w", encoding="utf-8") as out:
        for source, iterator in iterators:
            emitted = 0
            for situation in iterator:
                if args.max_per_source is not None and emitted >= args.max_per_source:
                    break
                if not args.no_dedup:
                    key = dedup_key(situation)
                    if key in seen:
                        dropped_dupes += 1
                        continue
                    seen.add(key)
                out.write(json.dumps(situation, ensure_ascii=False) + "\n")
                emitted += 1
                total += 1
                activity = situation["action_input"]["activity"] or "unknown"
                location = situation["context_input"]["location_current"] or "unknown"
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
                location_counts[location] = location_counts.get(location, 0) + 1
            per_source[source] = emitted

    print("")
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Total situations : {total}")
    print(f"Dropped dupes    : {dropped_dupes}")
    print(f"Output           : {args.output}")
    print("")
    print("Per source:")
    for source, count in per_source.items():
        print(f"  {source:<10} : {count}")
    print("")
    print("By activity domain:")
    for activity, count in sorted(activity_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {activity:<30} : {count}")
    print("")
    print("Top locations:")
    for location, count in sorted(location_counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {location:<20} : {count}")
    print("=" * 72)


if __name__ == "__main__":
    main()
