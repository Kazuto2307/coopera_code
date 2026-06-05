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
)
from preference_taxonomy import (
    PREFERENCE_SIGNALS,
    SIGNAL_SEMANTICS,
    describe_signal_semantics,
    differentiating_labels_for,
)
from qwen_labeler import QwenDecisionLabeler


# Morning / afternoon / evening keeps time-sensitive signals (context
# sensitivity, interruption sensitivity, immediacy) meaningful without exploding
# the grid the way v1's 13 hourly slots would.
DEFAULT_SITUATION_TIMES = ["9 am", "2 pm", "9 pm"]

# Used when neither --signals nor --all-signals is given. A diverse starter set
# spanning several subcategories of the new taxonomy.
DEFAULT_SIGNALS = [
    "autonomous_execution",
    "user_control",
    "action_immediacy",
    "interruption_sensitivity",
    "user_prompting",
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
        help="Use the whole taxonomy (all 12 signals).",
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
    # Stream each situation to disk as it is generated.
    writer = IncrementalSituationWriter(
        output=args.output,
        output_dir=args.output_dir.resolve(),
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
                    writer.write(signal, situation)
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
        writer.close()

    print_situations_summary(
        situations_by_signal=situations_by_signal,
        written=writer.paths,
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
    differentiating_labels = differentiating_labels_for(signal, "prefer")
    payload, _ = generator.generate_json(
        system=(
            "You design domestic assistive-robot situations anchored to a single "
            "preference signal. Return only valid JSON."
        ),
        user=build_situation_prompt(
            signal=signal,
            signal_description=signal_description,
            differentiating_labels=differentiating_labels,
            day=day,
            time_text=time_text,
            variant=variant,
        ),
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
    differentiating_labels: list[str],
    day: str,
    time_text: str,
    variant: int,
) -> str:
    semantics = SIGNAL_SEMANTICS.get(signal, {})
    label_if_has, label_if_not = differentiating_labels[0], differentiating_labels[1]
    context_hint = signal_context_hint(signal)
    return (
        "Design ONE domestic assistive-robot situation for a synthetic dataset.\n\n"
        f"Anchored preference signal: {signal}\n"
        f"Meaning: {signal_description}\n"
        f"- prefer (high) means: {semantics.get('prefer_means', '')}\n"
        f"- avoid (high) means: {semantics.get('avoid_means', '')}\n\n"
        "Goal: build a situation where THIS signal is the decisive lever. For the "
        "very same situation, the robot's best decision should be "
        f"'{label_if_has}' for a user who strongly holds this preference and "
        f"'{label_if_not}' for a user who does not.\n\n"
        "CRITICAL constraint on action_text:\n"
        "- action_text MUST be a CONCRETE domestic task that both a human and a "
        "robot could physically perform (a chore, a care task, or a daily-living "
        "activity). Examples: 'make breakfast', 'give medication to user', "
        "'vacuum the living room', 'clean the table', 'wash dishes', 'take out the "
        "trash', 'water the plants', 'prepare coffee', 'fold laundry'.\n"
        "- It must NOT describe the robot's response or framing. WRONG: 'offer to "
        "make breakfast', 'suggest the user takes medication', 'remind user to "
        "clean'. The robot's decision (do it now, tell the user, wait, etc.) is "
        "predicted separately and must not be baked into the action.\n\n"
        "Rules:\n"
        "- The situation must be domestic and assistive (home, daily living).\n"
        "- It must be plausible and natural, never forced or contrived.\n"
        "- Keep it AGNOSTIC to any personality: describe the world, not the user's "
        "preferences. The same situation is shown to everyone; only the anchored "
        "signal decides the outcome.\n"
        "- Shape the context (time_of_day, user_state, environment_flags, etc.) so "
        f"the signal makes a meaningful difference. {context_hint}\n"
        "- Do NOT include label_action, differentiating_labels, or preference_snapshot.\n"
        "- Anchor the situation to the given day and time.\n"
        "- scenario_rationale MUST explain why this signal flips the decision "
        f"between '{label_if_has}' and '{label_if_not}'.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "action_input": {"action_text": "...", "activity": "..."},\n'
        '  "context_input": {"location_current": "...", "objects_nearby": [], "available_objects": [], "raw_conditions": [], "time_of_day": "...", "weekday": "synthetic_day_XX", "user_state": [], "environment_flags": []},\n'
        '  "structured_task_features": {"kind": "...", "urgency": "low|medium|high", "sensitivity": "low|medium|high", "user_busy": false, "quiet_hours": false, "conditions": [], "context_flags": {}},\n'
        '  "scenario_rationale": "why this signal is the lever between the two labels"\n'
        "}\n\n"
        f"Day: {day}\n"
        f"Time: {time_text}\n"
        f"Variant: {variant}\n"
    )


# Per-signal hint nudging the context toward where the signal would matter.
_SIGNAL_CONTEXT_HINTS: dict[str, str] = {
    "interruption_sensitivity": "For example set user_state=['focused'] or environment_flags=['guests_present'].",
    "context_sensitivity": "For example use a nighttime time_of_day or user_state=['tired'].",
    "routine_adherence": "For example make the task fall inside or outside the user's usual routine slot.",
    "action_immediacy": "For example make the task something that could equally be done now or later.",
    "safety_priority": "For example introduce a mild safety concern (a spill, a hot stove left on).",
    "risk_caution": "For example make the action uncertain, sensitive, or potentially annoying.",
    "user_control": "For example make the task one a user might reasonably want to decide on themselves.",
    "user_prompting": "For example make the task something the user could easily do if told.",
    "autonomous_execution": "For example make the task safely automatable without supervision.",
    "robot_initiative": "For example leave the need unspoken so initiative is what differs.",
    "explanation_need": "For example make the action non-obvious so an explanation would help.",
}


def signal_context_hint(signal: str) -> str:
    return _SIGNAL_CONTEXT_HINTS.get(signal, "")


def describe_signal(signal: str) -> str:
    """Human-readable gloss for a signal, sourced from the taxonomy semantics."""
    return describe_signal_semantics(signal)


def compact_time(text: str) -> str:
    """'9 am' -> '9am' for use inside situation_id."""
    return "".join(str(text).lower().split())


class IncrementalSituationWriter:
    """Streams situations to disk as they are generated, one JSONL line each.

    Combined mode (``output`` set): a single JSONL with every situation.
    Per-signal mode (``output_dir``): one ``situations_<signal>.jsonl`` per
    signal, opened lazily on the first situation for that signal (so signals
    that produce nothing create no file, matching the previous batch behavior).
    Every line is flushed so the file can be tailed and an interrupt keeps all
    situations produced so far.
    """

    def __init__(self, *, output: Path | None, output_dir: Path) -> None:
        self.output = output
        self.output_dir = output_dir
        self.paths: list[Path] = []
        self._combined: Any = None
        self._per_signal: dict[str, Any] = {}
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            self._combined = output.open("w", encoding="utf-8")
            self.paths.append(output)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, signal: str, situation: dict[str, Any]) -> None:
        if self._combined is not None:
            handle = self._combined
        else:
            handle = self._per_signal.get(signal)
            if handle is None:
                path = self.output_dir / f"situations_{signal}.jsonl"
                handle = path.open("w", encoding="utf-8")
                self._per_signal[signal] = handle
                self.paths.append(path)
        handle.write(json.dumps(situation, ensure_ascii=False) + "\n")
        handle.flush()

    def close(self) -> None:
        if self._combined is not None:
            self._combined.close()
        for handle in self._per_signal.values():
            handle.close()


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
