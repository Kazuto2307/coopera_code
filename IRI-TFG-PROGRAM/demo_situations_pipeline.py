"""Demo: walk a few examples through the situation-generation pipeline.

Shows, for a handful of examples, the full transformation:

   RAW dataset annotation  (EPIC / Charades row)
        -> EXTRACTED situation   (build_situations_from_external logic)
        -> ROBOT ACTION          (translate_situations, with --translate / Qwen)

It reuses the REAL pipeline functions, so what you see is exactly what the full
run produces - just on a few, readable examples.

  # extraction only (CPU, no Qwen) - see RAW -> situation
  python IRI-TFG-PROGRAM/demo_situations_pipeline.py --per-source 4

  # full flow incl. robot actions (needs a GPU)
  python IRI-TFG-PROGRAM/demo_situations_pipeline.py --per-source 4 --translate
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterator

from build_situations_from_external import (
    iter_charades,
    iter_epic,
    load_charades_classes,
)

PROGRAM_DIR = Path(__file__).resolve().parent
DEFAULT_RAW = PROGRAM_DIR / "external_datasets" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo the situation-generation pipeline on a few examples.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--sources", nargs="+", choices=["epic", "charades"], default=["charades"],
                        help="Datasets to demo (default: charades only).")
    parser.add_argument("--per-source", type=int, default=4, help="Examples to show per dataset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-cap", type=int, default=4000, help="How many rows to sample examples from.")
    parser.add_argument("--translate", action="store_true", help="Also show robot actions (loads Qwen / GPU).")
    parser.add_argument("--output", type=Path, default=None, help="Optionally write the shown examples as JSONL.")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--qwen-temperature", type=float, default=0.2)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=512)
    return parser.parse_args()


def collect(iterator: Iterator[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for situation in iterator:
        pool.append(situation)
        if len(pool) >= cap:
            break
    return pool


def pick_diverse(pool: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Pick n examples, preferring distinct activity domains for a varied demo."""
    shuffled = pool[:]
    rng.shuffle(shuffled)
    chosen: list[dict[str, Any]] = []
    seen_domains: set[str] = set()
    for situation in shuffled:
        domain = situation["action_input"]["activity"]
        if domain not in seen_domains:
            seen_domains.add(domain)
            chosen.append(situation)
            if len(chosen) >= n:
                return chosen
    for situation in shuffled:  # fill remaining slots
        if situation not in chosen:
            chosen.append(situation)
            if len(chosen) >= n:
                break
    return chosen[:n]


def raw_view(situation: dict[str, Any]) -> list[tuple[str, str]]:
    meta = situation.get("source_metadata", {})
    if situation.get("source_dataset") == "epic_kitchens_100":
        return [
            ("dataset", "EPIC-Kitchens-100  (egocentric, unscripted, kitchen)"),
            ("narration_id", str(meta.get("narration_id"))),
            ("narration", f'"{meta.get("verb", "")}" -> {meta.get("noun", "")}  '
                          f'| all_nouns={meta.get("all_nouns")}'),
        ]
    return [
        ("dataset", "Charades  (scripted, whole-home, 3rd person)"),
        ("video / class", f'{meta.get("video_id")} / {meta.get("charades_class")}'),
        ("class label", f'"{meta.get("charades_label")}"'),
        ("scene / objects", f'{meta.get("scene")} | objects={meta.get("objects")}'),
    ]


def print_example(idx: int, situation: dict[str, Any], robot_action: str | None) -> None:
    ai = situation["action_input"]
    ci = situation["context_input"]
    feat = situation["structured_task_features"]
    print("\n" + "-" * 72)
    print(f"Example {idx}  [{situation['source_dataset']}]")
    print("-" * 72)

    print("1) RAW dataset annotation:")
    for key, val in raw_view(situation):
        print(f"     {key:<16}: {val}")

    print("\n2) EXTRACTED situation (build_situations_from_external):")
    print(f"     situation_id    : {situation['situation_id']}")
    print(f"     action_text     : {ai['action_text']!r}   <- HUMAN action")
    print(f"     activity (domain): {ai['activity']}")
    print(f"     location        : {ci['location_current']}")
    print(f"     objects_nearby  : {ci['objects_nearby']}")
    print(f"     user_state      : {ci['user_state']}")
    print(f"     task_features   : kind={feat['kind']} urgency={feat['urgency']} "
          f"sensitivity={feat['sensitivity']}")

    print("\n3) ROBOT ACTION (translate_situations):")
    if robot_action is None:
        print("     (run with --translate on a GPU to fill this)")
    else:
        print(f"     action_text     : {robot_action!r}   <- ROBOT action")


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    epic_dir = raw_dir / "epic"
    charades_dir = raw_dir / "charades" / "Charades"
    if not (charades_dir / "Charades_v1_train.csv").exists():
        charades_dir = raw_dir / "charades"
    rng = random.Random(args.seed)

    print("=" * 72)
    print("SITUATION-GENERATION PIPELINE DEMO")
    print("=" * 72)
    print(f"Raw dir     : {raw_dir}")
    print(f"Per source  : {args.per_source}    seed: {args.seed}")
    print(f"Translate   : {args.translate}")

    examples: list[dict[str, Any]] = []
    if "epic" in args.sources:
        epic_pool = collect(iter_epic(epic_dir, ["train"]), args.pool_cap)
        if epic_pool:
            examples += pick_diverse(epic_pool, args.per_source, rng)
    if "charades" in args.sources:
        classes = load_charades_classes(charades_dir / "Charades_v1_classes.txt")
        charades_pool = collect(iter_charades(charades_dir, ["train"], classes), args.pool_cap)
        if charades_pool:
            examples += pick_diverse(charades_pool, args.per_source, rng)

    if not examples:
        raise FileNotFoundError(
            f"No raw annotations found under {raw_dir}. Download them first "
            "(see build_situations_from_external.py)."
        )

    robot_actions: list[str | None] = [None] * len(examples)
    if args.translate:
        from translate_situations import translate_batch
        from qwen_labeler import QwenDecisionLabeler

        print("\nLoading Qwen for translation ...")
        generator = QwenDecisionLabeler(
            model_name=args.qwen_model,
            temperature=args.qwen_temperature,
            max_new_tokens=args.qwen_max_new_tokens,
        )
        try:
            phrases = [s["action_input"]["action_text"] for s in examples]
            robot_actions = list(translate_batch(generator=generator, phrases=phrases))
        finally:
            generator.close()

    for idx, (situation, robot) in enumerate(zip(examples, robot_actions), start=1):
        print_example(idx, situation, robot)
        if robot is not None:
            situation.setdefault("source_metadata", {})["original_action_text"] = (
                situation["action_input"]["action_text"]
            )
            situation["action_input"]["action_text"] = robot

    print("\n" + "=" * 72)
    print(f"Shown {len(examples)} examples "
          f"({sum(1 for s in examples if s['source_dataset'] == 'epic_kitchens_100')} EPIC, "
          f"{sum(1 for s in examples if s['source_dataset'] == 'charades')} Charades).")
    if not args.translate:
        print("Tip: add --translate (on a GPU) to also see the robot actions.")
    print("=" * 72)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for situation in examples:
                f.write(json.dumps(situation, ensure_ascii=False) + "\n")
        print(f"\nExamples written to {args.output}")


if __name__ == "__main__":
    main()
