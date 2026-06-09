"""Hourly minimal pipeline - final step: free decision over hourly situations.

Same idea as generate_training_data.py (Qwen decides freely what the person would
want the robot to do, new taxonomy, no forcing), but it builds a MINIMAL training
sample: it passes through the situation's minimal structure and does NOT re-add
urgency/sensitivity/user_busy/conditions/context_flags. It keeps the hour and
time so the decision can depend on time of day.

Reuses the decision prompt + reflection from generate_training_data.py.

Typical flow:
  build_situations_hourly.py        -> situations_hourly.jsonl        (human action, minimal, 24h)
  translate_situations.py           -> situations_hourly_robot.jsonl  (robot action)
  generate_training_data_hourly.py  -> training JSONL                 (free decision)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from generate_training_data import (
    assemble_sample,
    load_profiles,
    load_situations,
    memory_entry,
    reflect_free_decision,
)
from human_sim_preference_data import (
    ProgressDisplay,
    print_counter,
    summarize,
)
from qwen_labeler import QwenDecisionLabeler

SITU_DIR = Path(__file__).resolve().parent / "generated_data" / "situations_hourly"
DEFAULT_SITUATIONS = SITU_DIR / "situations_hourly_robot.jsonl"

# How many random picks to try before giving up on finding a non-forbidden
# (profile, situation) pair for a sample (sleep rule).
SLEEP_MAX_TRIES = 25


def _hour_in_window(hour: int, start: int, end: int) -> bool:
    """True if `hour` is in the sleep window [start, end) on a 24h clock (wraps)."""
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def apply_sleep_rule(
    profile: dict[str, Any], situation: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Apply the sleep rule for an (profile, hourly-situation) pair.

    Returns (verdict, situation_to_use):
    - "ok"        -> hour outside sleep (or no hour / no sleep_hours): use as-is.
    - "asleep"    -> hour in sleep AND the action is robot-autonomous: a copy with
                     context_input.user_state = ["asleep"].
    - "forbidden" -> hour in sleep AND not robot-autonomous (needs the person, or
                     unknown autonomy): the caller must resample.
    """
    hour = situation.get("hour")
    sleep = profile.get("sleep_hours") or {}
    try:
        start = int(sleep["start"]) % 24
        end = int(sleep["end"]) % 24
    except (KeyError, TypeError, ValueError):
        return "ok", situation
    if hour is None or not _hour_in_window(int(hour), start, end):
        return "ok", situation
    if situation.get("robot_autonomous") is True:
        asleep_sit = dict(situation)
        asleep_sit["context_input"] = {
            **(situation.get("context_input") or {}),
            "user_state": ["asleep"],
        }
        return "asleep", asleep_sit
    return "forbidden", situation


def parse_args() -> argparse.Namespace:
    program_dir = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_profiles_dir = program_dir / "generated_data" / "profiles"
    default_output = program_dir / "generated_data" / f"preference_training_hourly_{stamp}.jsonl"
    default_summary = program_dir / "generated_data" / f"preference_training_hourly_{stamp}_summary.json"

    parser = argparse.ArgumentParser(
        description="Assemble training JSONL (minimal samples) by free decision over hourly situations."
    )
    parser.add_argument("--profiles-dir", type=Path, default=default_profiles_dir)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--situations", type=Path, default=None,
                     help="Hourly situations JSONL (default: situations_hourly_robot.jsonl).")
    src.add_argument("--situations-dir", type=Path, default=None)
    parser.add_argument("--profile-indices", type=int, nargs="+", default=None)
    parser.add_argument("--target-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary-output", type=Path, default=default_summary)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--qwen-temperature", type=float, default=0.35)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1200)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles_dir = args.profiles_dir.resolve()
    profiles = load_profiles(profiles_dir, args.profile_indices)
    situations, situations_source = load_situations(
        situations_path=args.situations,
        situations_dir=args.situations_dir,
        default_path=DEFAULT_SITUATIONS,
    )

    plan = {
        "step": "generate_training_data_hourly (free decision, minimal samples)",
        "profiles_dir": str(profiles_dir),
        "num_profiles": len(profiles),
        "profile_ids": [str(p.get("human_id")) for p in profiles],
        "situations_source": situations_source,
        "num_situations": len(situations),
        "target_samples": args.target_samples,
        "planned_qwen_calls": args.target_samples,
        "output": str(args.output),
        "summary_output": str(args.summary_output),
    }
    print("")
    print("=" * 72)
    print("Generate training data (hourly, minimal, free decision)")
    print("=" * 72)
    print(f"Profiles          : {plan['num_profiles']} -> {plan['profile_ids']}")
    print(f"Situations source : {plan['situations_source']}")
    print(f"Situations loaded : {plan['num_situations']}")
    print(f"Target samples    : {plan['target_samples']}")
    print(f"Output            : {plan['output']}")
    print("=" * 72)

    if args.dry_run:
        return
    if not profiles:
        raise FileNotFoundError(f"No profile JSON files found in: {profiles_dir}")
    if not situations:
        raise FileNotFoundError(f"No situations loaded from: {situations_source}")

    generator = QwenDecisionLabeler(
        model_name=args.qwen_model,
        temperature=args.qwen_temperature,
        max_new_tokens=args.qwen_max_new_tokens,
    )
    rng = random.Random(args.seed)
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    memory_by_human: dict[str, list[dict[str, Any]]] = {}
    sleep_counts: Counter[str] = Counter()
    seen_profiles: set[Any] = set()
    progress = ProgressDisplay(
        enabled=not args.no_progress,
        total_profiles=len(profiles),
        total_samples=args.target_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_handle = args.output.open("w", encoding="utf-8")
    progress.start()

    try:
        for index in range(args.target_samples):
            profile = profiles[index % len(profiles)]
            human_id = str(profile.get("human_id"))
            if profile.get("profile_index") not in seen_profiles:
                seen_profiles.add(profile.get("profile_index"))
                progress.profile_start(human_id=human_id, profile_index=profile.get("profile_index"))
                progress.profile_done()

            # Sleep rule: forbidden pairs (asleep + non-autonomous action) are
            # resampled; asleep + autonomous gets user_state = ["asleep"].
            situation = None
            verdict = "ok"
            for _ in range(SLEEP_MAX_TRIES):
                candidate = rng.choice(situations)
                verdict, used = apply_sleep_rule(profile, candidate)
                if verdict != "forbidden":
                    situation = used
                    break
            if situation is None:
                sleep_counts["forbidden_no_valid_situation"] += 1
                continue
            sleep_counts[verdict] += 1

            memory = memory_by_human.setdefault(human_id, [])
            progress.stage("free_decision", human_id=human_id, time_text=situation.get("time_text"))
            try:
                decision_payload, _ = reflect_free_decision(
                    generator=generator,
                    profile_summary=profile.get("profile_summary") or {},
                    preference_profile=profile.get("preference_profile") or {},
                    situation=situation,
                    memory=memory,
                )
                sample = assemble_sample(
                    profile=profile,
                    situation=situation,
                    decision_payload=decision_payload,
                    index=index,
                )
            except Exception as exc:
                errors.append({"human_id": human_id, "stage": "free_decision",
                               "error": f"{type(exc).__name__}: {exc}"})
                progress.error(human_id=human_id, stage="free_decision",
                               message=f"{type(exc).__name__}: {exc}")
                continue

            sample["source_metadata"]["asleep"] = verdict == "asleep"
            samples.append(sample)
            output_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            output_handle.flush()
            memory.append(memory_entry(sample))
            progress.sample(label=sample["label_action"])
    finally:
        progress.close()
        generator.close()
        output_handle.close()

    summary = summarize(samples=samples, errors=errors, args=args, plan=plan)
    summary["label_distribution"] = dict(Counter(s["label_action"] for s in samples))
    summary["time_text_distribution"] = dict(
        Counter(s["source_metadata"].get("time_text") for s in samples)
    )
    summary["sleep_rule"] = dict(sleep_counts)
    summary["asleep_samples"] = sum(
        1 for s in samples if s["source_metadata"].get("asleep")
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("=" * 72)
    print("Hourly training data summary")
    print("=" * 72)
    print(f"Samples generated : {summary['num_samples']}")
    print(f"Errors            : {summary['num_errors']}")
    print(f"Output JSONL      : {summary['output']}")
    print("\nLabel distribution (natural, unforced):")
    print_counter(summary.get("label_distribution", {}))
    print("\nTop preference signals:")
    print_counter(summary.get("top_preference_signals", {}), limit=12)
    print("=" * 72)


if __name__ == "__main__":
    main()
