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
from iri_tfg_program import PROJECT_ROOT
from typing import Any

from iri_tfg_program.labeling.qwen_labeler import QwenDecisionLabeler

PROGRAM_DIR = PROJECT_ROOT
SITU_DIR = PROGRAM_DIR / "data/generated" / "situations_external"
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

    cached: dict[str, dict[str, Any]] = {}
    if args.reuse_map and args.map_output.exists():
        raw_cached = json.loads(args.map_output.read_text(encoding="utf-8"))
        cached = {k: _coerce_cached(v, k) for k, v in raw_cached.items()}
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
                entries = translate_batch(generator=generator, phrases=batch)
                for phrase, entry in zip(batch, entries):
                    mapping[phrase] = entry
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
            entry = mapping.get(original)
            if entry:
                action_input["action_text"] = entry["robot_action"]
                situation["action_input"] = action_input
                meta = situation.setdefault("source_metadata", {})
                meta["original_action_text"] = original
                # robot_autonomous -> top-level (the hourly sleep rule reads it).
                if entry.get("robot_autonomous") is not None:
                    situation["robot_autonomous"] = entry["robot_autonomous"]
                # Qwen-extracted user_state replaces the heuristic dataset one.
                if entry.get("user_state") is not None:
                    ctx = situation.setdefault("context_input", {})
                    ctx["user_state"] = entry["user_state"]
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
        e = mapping[phrase]
        print(f"  {phrase!r:<38} -> {e.get('robot_action')!r} "
              f"[autonomous={e.get('robot_autonomous')}, user_state={e.get('user_state')}]")
    print("=" * 72)


def translate_batch(
    *,
    generator: QwenDecisionLabeler,
    phrases: list[str],
) -> list[dict[str, Any]]:
    """Translate+tag a batch. Each entry: {robot_action, robot_autonomous, user_state}.

    Falls back to per-phrase, then identity, on failure.
    """
    try:
        items = _translate_call(generator=generator, phrases=phrases)
        if len(items) == len(phrases):
            return [_normalize_item(item, phrase) for item, phrase in zip(items, phrases)]
    except Exception:
        pass

    out: list[dict[str, Any]] = []
    for phrase in phrases:
        try:
            items = _translate_call(generator=generator, phrases=[phrase])
            out.append(_normalize_item(items[0] if items else {}, phrase))
        except Exception:
            out.append(_identity_item(phrase))
    return out


def _translate_call(
    *,
    generator: QwenDecisionLabeler,
    phrases: list[str],
) -> list[Any]:
    payload, _ = generator.generate_json(
        system=(
            "You convert human daily-living actions into robot assistance actions and "
            "tag them. Return only valid JSON."
        ),
        user=build_translation_prompt(phrases),
    )
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Response missing 'items' list.")
    return items


def build_translation_prompt(phrases: list[str]) -> str:
    numbered = "\n".join(f"{i}. {phrase}" for i, phrase in enumerate(phrases, start=1))
    return (
        "For each human action below, return THREE things about the robot's candidate "
        "assistance in that situation.\n\n"
        "1) robot_action: the SHORT robot assistance action (robot's perspective, "
        "imperative, 2-6 words).\n"
        "   - If the robot can do the task itself, keep it as the robot doing it "
        "(\"make breakfast\" -> \"prepare breakfast\").\n"
        "   - If only the person can do it to themselves, reframe as assist/offer "
        "(\"consume pills\" -> \"offer the pills\"; \"drink water\" -> \"bring a glass of water\").\n"
        "   - If passive/rest (\"lying on a bed\"), a gentle optional assistance "
        "(\"check if the user needs anything\").\n"
        "2) robot_autonomous: true if the robot can carry out this action ON ITS OWN, "
        "WITHOUT the person present or awake (e.g. tidy the room, water the plants, mop "
        "the floor); false if it needs the person there/awake (offer/remind/help-walk, "
        "or anything the person does to themselves).\n"
        "3) user_state: short list of the person's physical state implied by the action "
        "(e.g. [\"seated\"], [\"lying_down\",\"in_bed\"], [\"walking\"], [\"standing\"]); "
        "use [] if none is implied.\n\n"
        "Do NOT decide whether the robot SHOULD act; only describe the candidate action "
        "and the two tags.\n\n"
        "Return JSON exactly as: {\"items\": [{\"robot_action\": \"...\", "
        "\"robot_autonomous\": true, \"user_state\": [\"...\"]}, ...]} with one entry per "
        "input action, in the SAME order and the SAME count.\n\n"
        f"Human actions:\n{numbered}"
    )


def _normalize_item(item: Any, phrase: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        return _identity_item(phrase)
    robot = _clean(str(item.get("robot_action", "")), fallback=phrase)
    user_state = item.get("user_state")
    if isinstance(user_state, list):
        states: list[str] | None = [str(s).strip() for s in user_state if str(s).strip()]
    elif user_state:
        states = [str(user_state).strip()]
    else:
        states = []
    return {
        "robot_action": robot,
        "robot_autonomous": _coerce_bool(item.get("robot_autonomous")),
        "user_state": states,
    }


def _identity_item(phrase: str) -> dict[str, Any]:
    return {"robot_action": phrase, "robot_autonomous": None, "user_state": None}


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def _coerce_cached(value: Any, phrase: str) -> dict[str, Any]:
    if isinstance(value, dict) and "robot_action" in value:
        return value
    if isinstance(value, str):  # old map format: {phrase: robot_action_str}
        return {"robot_action": value, "robot_autonomous": None, "user_state": None}
    return _identity_item(phrase)


def _clean(robot: str, *, fallback: str) -> str:
    text = " ".join(str(robot).strip().split())
    text = text.strip().strip(".").strip()
    return text or fallback


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
