"""Translate situation actions into robot-assistance actions (Qwen / GPU).

The situations produced by ``build_situations_from_external.py`` carry the
*human* activity as ``action_text`` ("consume pills", "make breakfast"). This
script rewrites each one into the candidate **robot** action the home robot
could perform or offer in that situation:

  * already robot-doable  -> kept as the robot doing it   ("make breakfast" -> "prepare breakfast")
  * human-only self action -> reframed as assist / offer   ("consume pills"  -> "offer the pills")
  * passive / rest         -> gentle optional assistance   ("lying on a bed" -> "check if the user needs anything")

It does NOT decide whether the robot should act (that is the free decision in
generate_training_data.py); it only produces the candidate action.

Efficiency: actions are deduplicated and translated in batches, and the
{human_action -> robot_action} map is cached to disk so re-runs are cheap and
resumable (``--reuse-map``).

Runs Qwen, so use a GPU/server. ``--dry-run`` shows the plan without loading it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from qwen_labeler import QwenDecisionLabeler

PROGRAM_DIR = Path(__file__).resolve().parent
SITU_DIR = PROGRAM_DIR / "generated_data" / "situations_external"
DEFAULT_INPUT = SITU_DIR / "external_situations.jsonl"
DEFAULT_OUTPUT = SITU_DIR / "external_situations_robot.jsonl"
DEFAULT_MAP = SITU_DIR / "translation_map.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite situation action_text into robot-assistance actions with Qwen."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--map-output", type=Path, default=DEFAULT_MAP)
    parser.add_argument(
        "--reuse-map",
        action="store_true",
        help="Load an existing translation map and only translate new phrases.",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--max-unique",
        type=int,
        default=None,
        help="Cap the number of unique phrases translated (debug/testing).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Translate only N RANDOM situations (for quick tests); the output file "
            "then contains just those N. Tip: point --output to a test path so you "
            "do not overwrite the full dataset."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for --sample.")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--qwen-temperature", type=float, default=0.2)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1024)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Situations input not found: {args.input}")

    situations = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total_loaded = len(situations)
    if args.sample is not None and args.sample < total_loaded:
        situations = random.Random(args.seed).sample(situations, args.sample)
    unique_actions = sorted({
        str((s.get("action_input") or {}).get("action_text", "")).strip()
        for s in situations
        if str((s.get("action_input") or {}).get("action_text", "")).strip()
    })

    cached: dict[str, str] = {}
    if args.reuse_map and args.map_output.exists():
        cached = json.loads(args.map_output.read_text(encoding="utf-8"))
    todo = [action for action in unique_actions if action not in cached]
    if args.max_unique is not None:
        todo = todo[: args.max_unique]

    print("")
    print("=" * 72)
    print("Translate situation actions -> robot actions")
    print("=" * 72)
    print(f"Input situations   : {total_loaded}")
    if args.sample is not None:
        print(f"Random sample      : {len(situations)} situations (seed {args.seed})")
    print(f"Unique actions     : {len(unique_actions)}")
    print(f"Already cached      : {len(cached)}")
    print(f"To translate        : {len(todo)}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Planned Qwen calls  : {(len(todo) + args.batch_size - 1) // max(args.batch_size, 1)}")
    print(f"Input               : {args.input}")
    print(f"Output              : {args.output}")
    print(f"Map                 : {args.map_output}")
    print("=" * 72)

    if args.dry_run:
        return

    mapping = dict(cached)
    if todo:
        generator = QwenDecisionLabeler(
            model_name=args.qwen_model,
            temperature=args.qwen_temperature,
            max_new_tokens=args.qwen_max_new_tokens,
        )
        try:
            batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
            for batch_idx, batch in enumerate(batches, start=1):
                robots = translate_batch(generator=generator, phrases=batch)
                for phrase, robot in zip(batch, robots):
                    mapping[phrase] = robot
                if not args.no_progress and (batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(batches)):
                    done = min(batch_idx * args.batch_size, len(todo))
                    print(f"  translated {done}/{len(todo)} unique actions "
                          f"(batch {batch_idx}/{len(batches)})")
                # Persist the growing map so an interrupt keeps progress.
                _write_json(args.map_output, mapping)
        finally:
            generator.close()
    _write_json(args.map_output, mapping)

    # Rewrite every situation with its robot action (stream to disk).
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rewritten = 0
    with args.output.open("w", encoding="utf-8") as out:
        for situation in situations:
            action_input = situation.get("action_input") or {}
            original = str(action_input.get("action_text", "")).strip()
            robot = mapping.get(original)
            if robot:
                action_input["action_text"] = robot
                meta = situation.setdefault("source_metadata", {})
                meta["original_action_text"] = original
                situation["action_input"] = action_input
                rewritten += 1
            out.write(json.dumps(situation, ensure_ascii=False) + "\n")

    print("")
    print("=" * 72)
    print("Translation summary")
    print("=" * 72)
    print(f"Situations written : {len(situations)}")
    print(f"Actions rewritten  : {rewritten}")
    print(f"Map entries        : {len(mapping)}")
    print(f"Output             : {args.output}")
    print("")
    print("Examples:")
    for phrase in list(mapping)[:8]:
        print(f"  {phrase!r:<45} -> {mapping[phrase]!r}")
    print("=" * 72)


def translate_batch(
    *,
    generator: QwenDecisionLabeler,
    phrases: list[str],
) -> list[str]:
    """Translate a batch; fall back to per-phrase, then identity, on failure."""
    try:
        robots = _translate_call(generator=generator, phrases=phrases)
        if len(robots) == len(phrases):
            return [_clean(robot, fallback=phrase) for robot, phrase in zip(robots, phrases)]
    except Exception:
        pass

    # Misaligned or failed batch: translate each phrase on its own.
    out: list[str] = []
    for phrase in phrases:
        try:
            single = _translate_call(generator=generator, phrases=[phrase])
            out.append(_clean(single[0] if single else "", fallback=phrase))
        except Exception:
            out.append(phrase)  # last resort: keep the original
    return out


def _translate_call(
    *,
    generator: QwenDecisionLabeler,
    phrases: list[str],
) -> list[str]:
    payload, _ = generator.generate_json(
        system=(
            "You convert human daily-living actions into short robot assistance "
            "actions for a home assistive robot. Return only valid JSON."
        ),
        user=build_translation_prompt(phrases),
    )
    robots = payload.get("robot_actions")
    if not isinstance(robots, list):
        raise ValueError("Response missing 'robot_actions' list.")
    return [str(item) for item in robots]


def build_translation_prompt(phrases: list[str]) -> str:
    numbered = "\n".join(f"{i}. {phrase}" for i, phrase in enumerate(phrases, start=1))
    return (
        "For each human action below, write the SHORT robot assistance action that "
        "a home assistive robot could perform or offer in that situation.\n\n"
        "Rules:\n"
        "- Robot's perspective, imperative, concise (2-6 words).\n"
        "- If the robot can do the task itself, keep it as the robot doing it "
        "(\"make breakfast\" -> \"prepare breakfast\"; \"open door\" -> \"open the door\").\n"
        "- If it is something only the person does to themselves, reframe it as the "
        "robot ASSISTING or OFFERING (\"consume pills\" -> \"offer the pills\"; "
        "\"drink water\" -> \"bring a glass of water\"; \"eat a sandwich\" -> "
        "\"serve the sandwich\").\n"
        "- If it is passive or rest (\"lying on a bed\"), use a gentle optional "
        "assistance (\"check if the user needs anything\").\n"
        "- Do NOT decide whether the robot should act; only describe the candidate "
        "action. Do not add explanations.\n\n"
        "Return JSON exactly as: {\"robot_actions\": [\"...\", \"...\"]} with one "
        "entry per input action, in the SAME order and the SAME count.\n\n"
        f"Human actions:\n{numbered}"
    )


def _clean(robot: str, *, fallback: str) -> str:
    text = " ".join(str(robot).strip().split())
    text = text.strip().strip(".").strip()
    return text or fallback


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
