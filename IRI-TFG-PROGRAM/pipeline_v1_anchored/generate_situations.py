"""Step 2 of the split preference-data pipeline: generate anchored situations.

This script generates domestic assistive-robot situations that are *anchored* to
a taxonomy signal. The key idea: each situation is designed so that the robot's
best ``label_action`` flips depending on whether the human HAS the anchored
preference signal or not. The signal is the "lever".

Situations are agnostic to any concrete human profile (they describe the world,
not the person) but sensitive to the taxonomy: ``generate_training_data.py``
later pairs each situation with profiles whose stable preferences include the
anchored signal.

For every selected signal, Qwen is asked to (a) infer the two labels the signal
differentiates between and (b) craft a plausible, non-forced situation whose
``scenario_rationale`` makes that contrast explicit. The differentiating label
pairs are NOT hardcoded; Qwen infers them from the signal name.

Reuses the v1 normalizers (``normalize_action_input`` /
``normalize_context_input`` / ``normalize_structured_features``) so the emitted
inputs match the rest of the pipeline. Nothing loads Qwen under ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from human_sim_preference_data import (
    ProgressDisplay,
    normalize_action_input,
    normalize_context_input,
    normalize_structured_features,
    write_jsonl,
)
from preference_taxonomy import PREFERENCE_SIGNALS, VALID_LABELS
from qwen_labeler import QwenDecisionLabeler


# Morning / afternoon / evening keeps time-sensitive signals (quiet hours,
# nighttime notifications, time-of-day preferences) meaningful without exploding
# the grid the way v1's 13 hourly slots would.
DEFAULT_SITUATION_TIMES = ["9 am", "2 pm", "9 pm"]

# Used when neither --signals nor --all-signals is given. These are exactly the
# five worked examples in the design brief, so a no-arg run produces a useful,
# diverse starter set.
DEFAULT_SIGNALS = [
    "prefer_confirmation_before_action",
    "avoid_interrupt_during_quiet_hours",
    "prefer_proactive_assistance",
    "prefer_on_demand_reminders",
    "prefer_high_robot_autonomy",
]


def parse_args() -> argparse.Namespace:
    program_dir = Path(__file__).resolve().parent
    default_output_dir = program_dir / "generated_data" / "situations"

    parser = argparse.ArgumentParser(
        description=(
            "Step 2/3: generate domestic assistive-robot situations anchored to "
            "taxonomy signals (the signal is the lever that flips the decision)."
        )
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--signals",
        nargs="+",
        default=None,
        help="Subset of taxonomy signals to anchor situations to.",
    )
    selection.add_argument(
        "--all-signals",
        action="store_true",
        help="Use the whole taxonomy (all 50 signals).",
    )

    parser.add_argument(
        "--situations-per-signal",
        type=int,
        default=3,
        help=(
            "Number of situations to generate per signal (default: 3). They are "
            "distributed round-robin over the (day x time) grid."
        ),
    )
    parser.add_argument(
        "--times",
        nargs="*",
        default=None,
        help='Times of day, e.g. --times "9 am" "2 pm" "9 pm".',
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of synthetic days (default: 3).",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Explicit path to a single combined JSONL. If omitted, one JSONL per "
            "signal is written under --output-dir."
        ),
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--qwen-temperature", type=float, default=0.6)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1200)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generation plan without loading Qwen or writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signals = select_signals(signals_arg=args.signals, all_signals=args.all_signals)
    times = args.times or DEFAULT_SITUATION_TIMES
    if args.days <= 0:
        raise ValueError("--days must be positive.")
    if args.situations_per_signal <= 0:
        raise ValueError("--situations-per-signal must be positive.")
    days = [str(i).zfill(2) for i in range(1, args.days + 1)]
    grid = [(day, time_text) for day in days for time_text in times]
    if not grid:
        raise ValueError("Empty (day x time) grid. Check --days and --times.")

    total_situations = len(signals) * args.situations_per_signal
    plan = {
        "step": "2/3 generate_situations",
        "num_signals": len(signals),
        "signals": signals,
        "situations_per_signal": args.situations_per_signal,
        "days": days,
        "times": times,
        "total_situations": total_situations,
        "output_dir": str(args.output_dir.resolve()),
        "output": str(args.output) if args.output is not None else None,
    }
    print_situations_plan(plan)

    if args.dry_run:
        return

    generator = QwenDecisionLabeler(
        model_name=args.qwen_model,
        temperature=args.qwen_temperature,
        max_new_tokens=args.qwen_max_new_tokens,
    )

    situations_by_signal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, str]] = []
    progress = ProgressDisplay(
        enabled=not args.no_progress,
        total_profiles=len(signals),
        total_samples=total_situations,
    )
    progress.start()

    try:
        for signal_index, signal in enumerate(signals):
            description = describe_signal(signal)
            progress.profile_start(human_id=signal, profile_index=signal_index)
            for variant in range(1, args.situations_per_signal + 1):
                day, time_text = grid[(variant - 1) % len(grid)]
                progress.stage(
                    "situation",
                    human_id=signal,
                    day=day,
                    time_text=time_text,
                    scenario_idx=variant,
                )
                try:
                    situation = generate_situation(
                        generator=generator,
                        signal=signal,
                        signal_description=description,
                        day=day,
                        time_text=time_text,
                        variant=variant,
                    )
                    situation_id = (
                        f"sit:{signal}:day{day}:{compact_time(time_text)}:{variant:03d}"
                    )
                    situation = {"situation_id": situation_id, **situation}
                    situations_by_signal[signal].append(situation)
                    progress.sample(label=signal)
                except Exception as exc:
                    errors.append(
                        {
                            "signal": signal,
                            "day": day,
                            "time": time_text,
                            "variant": str(variant),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    progress.error(
                        human_id=signal,
                        stage="situation",
                        message=f"{type(exc).__name__}: {exc}",
                    )
            progress.profile_done()
    finally:
        progress.close()
        generator.close()

    written = write_situations(
        situations_by_signal=situations_by_signal,
        output=args.output,
        output_dir=args.output_dir.resolve(),
    )
    print_situations_summary(
        situations_by_signal=situations_by_signal,
        written=written,
        errors=errors,
    )


def select_signals(*, signals_arg: list[str] | None, all_signals: bool) -> list[str]:
    if all_signals:
        return sorted(PREFERENCE_SIGNALS)
    if signals_arg:
        unknown = [s for s in signals_arg if s not in PREFERENCE_SIGNALS]
        if unknown:
            raise ValueError(f"Unknown signals (not in taxonomy): {unknown}")
        ordered: list[str] = []
        seen: set[str] = set()
        for signal in signals_arg:
            if signal not in seen:
                seen.add(signal)
                ordered.append(signal)
        return ordered
    return list(DEFAULT_SIGNALS)


def generate_situation(
    *,
    generator: QwenDecisionLabeler,
    signal: str,
    signal_description: str,
    day: str,
    time_text: str,
    variant: int,
) -> dict[str, Any]:
    payload, _ = generator.generate_json(
        system=(
            "You design domestic assistive-robot situations anchored to a single "
            "preference signal. Return only valid JSON."
        ),
        user=build_situation_prompt(
            signal=signal,
            signal_description=signal_description,
            day=day,
            time_text=time_text,
            variant=variant,
        ),
    )
    differentiating_labels = validate_differentiating_labels(
        payload.get("differentiating_labels")
    )
    return {
        "anchored_signal": signal,
        "differentiating_labels": differentiating_labels,
        "day": day,
        "time_text": time_text,
        "action_input": normalize_action_input(payload.get("action_input")),
        "context_input": normalize_context_input(
            payload.get("context_input"), day=day
        ),
        "structured_task_features": normalize_structured_features(
            payload.get("structured_task_features")
        ),
        "scenario_rationale": str(payload.get("scenario_rationale", "")).strip(),
    }


def build_situation_prompt(
    *,
    signal: str,
    signal_description: str,
    day: str,
    time_text: str,
    variant: int,
) -> str:
    allowed_labels = json.dumps(sorted(VALID_LABELS))
    return (
        "Design ONE domestic assistive-robot situation for a synthetic dataset.\n\n"
        f"Anchored preference signal: {signal}\n"
        f"Meaning (inferred from the name): {signal_description}\n\n"
        "Goal: build a situation where THIS signal is the decisive lever. The "
        "robot's best action must change depending on whether the user has this "
        "preference or not. A user WITH the signal and a user WITHOUT it should "
        "lead to two DIFFERENT label_action outcomes for the very same situation.\n\n"
        "First infer the two labels this signal differentiates between, chosen "
        f"from: {allowed_labels}.\n"
        "Examples of how a signal can differentiate two labels (do NOT copy "
        "blindly; infer from the actual signal above):\n"
        "- prefer_confirmation_before_action -> remind vs do_now\n"
        "- avoid_interrupt_during_quiet_hours -> no_action vs do_now\n"
        "- prefer_proactive_assistance -> do_now vs do_later\n"
        "- prefer_on_demand_reminders -> no_action vs remind\n"
        "- prefer_high_robot_autonomy -> do_now vs remind\n\n"
        "Rules:\n"
        "- The situation must be domestic and assistive (home, daily living).\n"
        "- It must be plausible and natural, never forced or contrived.\n"
        "- Keep it AGNOSTIC to any personality: describe the world, not the user's "
        "preferences. The same situation is shown to everyone; only the anchored "
        "signal decides the outcome.\n"
        "- Do NOT include label_action or preference_snapshot.\n"
        "- Anchor the situation to the given day and time.\n"
        "- scenario_rationale MUST state which two labels differ and why this "
        "signal is what flips the decision between them.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "differentiating_labels": ["<label_if_user_has_signal>", "<label_if_not>"],\n'
        '  "action_input": {"action_text": "...", "activity": "..."},\n'
        '  "context_input": {"location_current": "...", "objects_nearby": [], "available_objects": [], "raw_conditions": [], "time_of_day": "...", "weekday": "synthetic_day_XX", "user_state": [], "environment_flags": []},\n'
        '  "structured_task_features": {"kind": "...", "urgency": "low|medium|high", "sensitivity": "low|medium|high", "user_busy": false, "quiet_hours": false, "conditions": [], "context_flags": {}},\n'
        '  "scenario_rationale": "which two labels differ and why this signal is the lever"\n'
        "}\n\n"
        f"Day: {day}\n"
        f"Time: {time_text}\n"
        f"Variant: {variant}\n"
    )


def describe_signal(signal: str) -> str:
    """Human-readable gloss inferred purely from the signal name."""
    if signal.startswith("avoid_"):
        body = signal[len("avoid_"):].replace("_", " ")
        return f"the user prefers to avoid {body}"
    if signal.startswith("prefer_"):
        body = signal[len("prefer_"):].replace("_", " ")
        return f"the user prefers {body}"
    return signal.replace("_", " ")


def validate_differentiating_labels(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("differentiating_labels must be a list.")
    labels: list[str] = []
    for item in value:
        label = str(item).strip()
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid differentiating label: {label!r}")
        if label not in labels:
            labels.append(label)
    if len(labels) != 2:
        raise ValueError(
            f"differentiating_labels must contain exactly 2 distinct labels, got {labels}"
        )
    return labels


def compact_time(text: str) -> str:
    """'9 am' -> '9am' for use inside situation_id."""
    return "".join(str(text).lower().split())


def write_situations(
    *,
    situations_by_signal: dict[str, list[dict[str, Any]]],
    output: Path | None,
    output_dir: Path,
) -> list[Path]:
    if output is not None:
        all_situations = [
            situation
            for situations in situations_by_signal.values()
            for situation in situations
        ]
        write_jsonl(output, all_situations)
        return [output]

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for signal, situations in situations_by_signal.items():
        path = output_dir / f"situations_{signal}.jsonl"
        write_jsonl(path, situations)
        written.append(path)
    return written


def print_situations_plan(plan: dict[str, Any]) -> None:
    print("")
    print("=" * 72)
    print("Step 2/3 - generate signal-anchored situations")
    print("=" * 72)
    print(f"Signals           : {plan['num_signals']}")
    print(f"Per signal        : {plan['situations_per_signal']}")
    print(f"Days              : {len(plan['days'])} -> {', '.join(plan['days'])}")
    print(f"Times             : {len(plan['times'])} -> {', '.join(plan['times'])}")
    print(f"Total situations  : {plan['total_situations']}")
    print("")
    if plan["output"]:
        print(f"Combined output   : {plan['output']}")
    else:
        print(f"Output dir        : {plan['output_dir']} (one JSONL per signal)")
    print("")
    print("Anchored signals:")
    for signal in plan["signals"]:
        print(f"  - {signal}")
    print("=" * 72)


def print_situations_summary(
    *,
    situations_by_signal: dict[str, list[dict[str, Any]]],
    written: list[Path],
    errors: list[dict[str, str]],
) -> None:
    total = sum(len(situations) for situations in situations_by_signal.values())
    print("")
    print("=" * 72)
    print("Situation generation summary")
    print("=" * 72)
    print(f"Situations generated : {total}")
    print(f"Errors               : {len(errors)}")
    print("")
    print("Per signal:")
    if situations_by_signal:
        width = max(len(signal) for signal in situations_by_signal)
        for signal, situations in situations_by_signal.items():
            print(f"  {signal.ljust(width)} : {len(situations)}")
    else:
        print("  none")
    print("")
    print("Files written:")
    for path in written:
        print(f"  - {path}")
    if errors:
        print("")
        print("Errors preview:")
        for item in errors[:5]:
            print(f"- {item['signal']} ({item['time']}): {item['error']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
