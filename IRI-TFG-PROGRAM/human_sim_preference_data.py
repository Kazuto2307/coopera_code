from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from coopera_profile_loader import (
    load_latest_traits_summary,
    load_mypersonality_profiles,
    resolve_mypersonality_path,
)
from preference_taxonomy import (
    PREFERENCE_SIGNALS,
    SIGNAL_SEMANTICS,
    VALID_LABELS,
    signal_subcategory,
)
from qwen_labeler import QwenDecisionLabeler


# Default preference weight (1-10) used when Qwen omits it for an item.
DEFAULT_PREFERENCE_WEIGHT = 5.0


def taxonomy_signal_block() -> str:
    """Render the 12 signals with their prefer/avoid meaning for Qwen prompts."""
    lines: list[str] = []
    for signal in sorted(PREFERENCE_SIGNALS):
        semantics = SIGNAL_SEMANTICS.get(signal, {})
        lines.append(
            f"- {signal}: {semantics.get('description', '')} "
            f"prefer(high) => {semantics.get('prefer_means', '')} "
            f"avoid(high) => {semantics.get('avoid_means', '')}"
        )
    return "\n".join(lines)


DEFAULT_TIMES = [
    "9 am",
    "10 am",
    "11 am",
    "12 pm",
    "1 pm",
    "2 pm",
    "3 pm",
    "4 pm",
    "5 pm",
    "6 pm",
    "7 pm",
    "8 pm",
    "9 pm",
]


def parse_args() -> argparse.Namespace:
    program_dir = Path(__file__).resolve().parent
    coopera_root = program_dir.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = program_dir / "generated_data" / f"preference_human_sim_{stamp}.jsonl"
    default_summary = (
        program_dir / "generated_data" / f"preference_human_sim_{stamp}_summary.json"
    )
    default_intermediate = program_dir / "generated_data" / f"preference_human_sim_{stamp}_intermediate"

    parser = argparse.ArgumentParser(
        description=(
            "COOPERA-style human simulator for the preference model. It generates "
            "profile-grounded preference-training JSONL through staged Qwen calls."
        )
    )
    parser.add_argument("--coopera-root", type=Path, default=coopera_root)
    parser.add_argument("--mypersonality-path", type=Path, default=None)
    parser.add_argument(
        "--response-source",
        choices=["gpt_response", "llama_response"],
        default="gpt_response",
    )
    parser.add_argument("--profile-indices", type=int, nargs="+", default=None)
    parser.add_argument("--max-profiles", type=int, default=10)
    parser.add_argument("--max-days", type=int, default=1)
    parser.add_argument("--times", nargs="*", default=None)
    parser.add_argument("--samples-per-hour", type=int, default=1)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary-output", type=Path, default=default_summary)
    parser.add_argument("--intermediate-dir", type=Path, default=default_intermediate)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--qwen-temperature", type=float, default=0.25)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1800)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars and use plain terminal output.",
    )
    parser.add_argument(
        "--plan-json",
        action="store_true",
        help="Also print the full run plan as JSON at startup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned generation loops without loading Qwen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coopera_root = args.coopera_root.resolve()
    results_dir = coopera_root / "results"
    mypersonality_path = resolve_mypersonality_path(
        coopera_root=coopera_root,
        explicit_path=args.mypersonality_path,
    )
    profiles = load_mypersonality_profiles(mypersonality_path)
    selected_indices = select_profile_indices(
        total=len(profiles),
        explicit=args.profile_indices,
        max_profiles=args.max_profiles,
    )
    times = args.times or DEFAULT_TIMES
    days = [str(i).zfill(2) for i in range(args.max_days)]
    planned_qwen_calls = estimate_calls(
        profiles=profiles,
        selected_indices=selected_indices,
        results_dir=results_dir,
        response_source=args.response_source,
        days=days,
        times=times,
        samples_per_hour=args.samples_per_hour,
    )
    planned_samples = len(selected_indices) * len(days) * len(times) * args.samples_per_hour

    plan = {
        "pipeline": "profile_summary -> preference_profile -> scenario -> decision_reflection",
        "coopera_root": str(coopera_root),
        "mypersonality_path": str(mypersonality_path),
        "num_profiles_available": len(profiles),
        "profile_indices": selected_indices,
        "days": days,
        "times": times,
        "samples_per_hour": args.samples_per_hour,
        "planned_qwen_calls": planned_qwen_calls,
        "output": str(args.output),
        "summary_output": str(args.summary_output),
        "intermediate_dir": str(args.intermediate_dir),
    }
    print_run_plan(
        plan=plan,
        planned_samples=planned_samples,
        show_json=args.plan_json,
    )

    if args.dry_run:
        return
    if not profiles:
        raise FileNotFoundError(f"No COOPERA profiles found at: {mypersonality_path}")

    generator = QwenDecisionLabeler(
        model_name=args.qwen_model,
        temperature=args.qwen_temperature,
        max_new_tokens=args.qwen_max_new_tokens,
    )

    samples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    progress = ProgressDisplay(
        enabled=not args.no_progress,
        total_profiles=len(selected_indices),
        total_samples=planned_samples,
    )
    progress.start()
    try:
        for profile_index in selected_indices:
            profile = profiles[profile_index]
            human_id = str(profile_index).zfill(5)
            progress.profile_start(human_id=human_id, profile_index=profile_index)
            traits_summary = load_latest_traits_summary(
                results_dir=results_dir,
                response_source=args.response_source,
                human_id=human_id,
            )
            profile_context = {
                "human_id": human_id,
                "profile_index": profile_index,
                "mypersonality": profile,
                "traits_summary": traits_summary,
            }

            try:
                progress.stage("profile_summary", human_id=human_id)
                profile_summary, profile_summary_meta = get_or_generate_profile_summary(
                    generator=generator,
                    profile_context=profile_context,
                )
                progress.stage("preference_profile", human_id=human_id)
                preference_profile, preference_meta = generate_preference_profile(
                    generator=generator,
                    profile_summary=profile_summary,
                    profile_context=profile_context,
                )
                write_intermediate(
                    args.intermediate_dir,
                    human_id,
                    "profile_summary",
                    profile_summary,
                    profile_summary_meta,
                )
                write_intermediate(
                    args.intermediate_dir,
                    human_id,
                    "preference_profile",
                    preference_profile,
                    preference_meta,
                )
            except Exception as exc:
                errors.append(
                    {
                        "human_id": human_id,
                        "stage": "profile_setup",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                progress.error(
                    human_id=human_id,
                    stage="profile_setup",
                    message=f"{type(exc).__name__}: {exc}",
                )
                progress.profile_done()
                continue

            for day in days:
                memory: list[dict[str, Any]] = []
                for time_text in times:
                    try:
                        progress.stage(
                            "scenario",
                            human_id=human_id,
                            day=day,
                            time_text=time_text,
                        )
                        scenario_payload, scenario_meta = generate_assistance_scenarios(
                            generator=generator,
                            profile_summary=profile_summary,
                            preference_profile=preference_profile,
                            day=day,
                            time_text=time_text,
                            samples_per_hour=args.samples_per_hour,
                            memory=memory,
                        )
                        raw_scenarios = scenario_payload.get("scenarios", [])
                        if not isinstance(raw_scenarios, list):
                            raise ValueError("Scenario stage must return a 'scenarios' list.")

                        for scenario_idx, scenario in enumerate(raw_scenarios, start=1):
                            progress.stage(
                                "decision_reflection",
                                human_id=human_id,
                                day=day,
                                time_text=time_text,
                                scenario_idx=scenario_idx,
                            )
                            decision_payload, decision_meta = reflect_decision(
                                generator=generator,
                                profile_summary=profile_summary,
                                preference_profile=preference_profile,
                                scenario=scenario,
                                memory=memory,
                            )
                            sample = build_sample_from_stages(
                                profile_context=profile_context,
                                profile_summary=profile_summary,
                                preference_profile=preference_profile,
                                scenario=scenario,
                                scenario_metadata=scenario_meta,
                                decision_payload=decision_payload,
                                decision_metadata=decision_meta,
                                day=day,
                                time_text=time_text,
                                scenario_idx=scenario_idx,
                                existing_count=len(samples),
                            )
                            samples.append(sample)
                            memory.append(
                                {
                                    "time": time_text,
                                    "action_input": sample["action_input"],
                                    "context_input": sample["context_input"],
                                    "label_action": sample["label_action"],
                                    "preference_snapshot": sample["preference_snapshot"],
                                }
                            )
                            progress.sample(label=sample["label_action"])
                    except Exception as exc:
                        errors.append(
                            {
                                "human_id": human_id,
                                "day": day,
                                "time": time_text,
                                "stage": "scenario_or_decision",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        progress.error(
                            human_id=human_id,
                            stage="scenario_or_decision",
                            message=f"{type(exc).__name__}: {exc}",
                        )
            progress.profile_done()
    finally:
        progress.close()
        generator.close()

    write_jsonl(args.output, samples)
    summary = summarize(samples=samples, errors=errors, args=args, plan=plan)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_run_summary(summary)


def print_run_plan(
    *,
    plan: dict[str, Any],
    planned_samples: int,
    show_json: bool,
) -> None:
    print("")
    print("=" * 72)
    print("COOPERA preference synthetic data run")
    print("=" * 72)
    print(f"Pipeline          : {plan['pipeline']}")
    print(f"Profiles selected : {len(plan['profile_indices'])} / {plan['num_profiles_available']}")
    print(f"Days              : {len(plan['days'])} -> {', '.join(plan['days'])}")
    print(f"Hours per day     : {len(plan['times'])} -> {', '.join(plan['times'])}")
    print(f"Samples per hour  : {plan['samples_per_hour']}")
    print(f"Target samples    : {planned_samples}")
    print(f"Planned Qwen calls: {plan['planned_qwen_calls']}")
    print("")
    print(f"Profiles CSV      : {plan['mypersonality_path']}")
    print(f"Output JSONL      : {plan['output']}")
    print(f"Summary JSON      : {plan['summary_output']}")
    print(f"Intermediates     : {plan['intermediate_dir']}")
    print("=" * 72)
    if show_json:
        print("")
        print("Full plan JSON:")
        print(json.dumps(plan, ensure_ascii=False, indent=2))


def print_run_summary(summary: dict[str, Any]) -> None:
    print("")
    print("=" * 72)
    print("Generation summary")
    print("=" * 72)
    print(f"Samples generated : {summary['num_samples']}")
    print(f"Errors            : {summary['num_errors']}")
    print(f"Output JSONL      : {summary['output']}")
    print(f"Summary JSON      : {summary['summary_output']}")
    print("")
    print("Labels:")
    print_counter(summary.get("labels", {}))
    print("")
    print("Activities:")
    print_counter(summary.get("activities", {}), limit=10)
    print("")
    print("Top preference signals:")
    print_counter(summary.get("top_preference_signals", {}), limit=12)
    if summary.get("errors_preview"):
        print("")
        print("Errors preview:")
        for item in summary["errors_preview"][:5]:
            print(f"- {item}")
    print("=" * 72)


def print_counter(values: dict[str, int], *, limit: int | None = None) -> None:
    if not values:
        print("  none")
        return
    rows = list(values.items())
    if limit is not None:
        rows = rows[:limit]
    width = max(len(str(key)) for key, _ in rows)
    for key, count in rows:
        print(f"  {str(key).ljust(width)} : {count}")


class ProgressDisplay:
    def __init__(
        self,
        *,
        enabled: bool,
        total_profiles: int,
        total_samples: int,
    ) -> None:
        self.enabled = enabled
        self.total_profiles = total_profiles
        self.total_samples = total_samples
        self.generated_samples = 0
        self.generated_profiles = 0
        self.error_count = 0
        self.label_counts: Counter[str] = Counter()
        self._tqdm: Any = None
        self._profiles_bar: Any = None
        self._samples_bar: Any = None

    def start(self) -> None:
        if not self.enabled:
            print("")
            print("Progress bars disabled.")
            return
        try:
            from tqdm import tqdm
        except Exception:
            self.enabled = False
            print("")
            print("tqdm is not installed. Using simple progress messages.")
            return
        self._tqdm = tqdm
        print("")
        self._profiles_bar = tqdm(
            total=self.total_profiles,
            desc="Profiles",
            unit="profile",
            position=0,
            leave=True,
        )
        self._samples_bar = tqdm(
            total=self.total_samples,
            desc="Samples",
            unit="sample",
            position=1,
            leave=True,
        )

    def profile_start(self, *, human_id: str, profile_index: int) -> None:
        text = f"Starting human {human_id} (profile_index={profile_index})"
        if self._profiles_bar is not None:
            self._profiles_bar.set_description(f"Profiles human={human_id}")
            self._write(text)
        else:
            print("")
            print(text)

    def profile_done(self) -> None:
        self.generated_profiles += 1
        if self._profiles_bar is not None:
            self._profiles_bar.update(1)
            self._profiles_bar.set_postfix(
                samples=self.generated_samples,
                errors=self.error_count,
            )

    def stage(
        self,
        stage: str,
        *,
        human_id: str,
        day: str | None = None,
        time_text: str | None = None,
        scenario_idx: int | None = None,
    ) -> None:
        parts = [stage, f"human={human_id}"]
        if day is not None:
            parts.append(f"day={day}")
        if time_text is not None:
            parts.append(f"time={time_text}")
        if scenario_idx is not None:
            parts.append(f"scenario={scenario_idx}")
        text = " | ".join(parts)
        if self._samples_bar is not None:
            self._samples_bar.set_postfix(
                stage=stage,
                human=human_id,
                errors=self.error_count,
                refresh=False,
            )
        elif not self.enabled:
            return
        else:
            print(text)

    def sample(self, *, label: str) -> None:
        self.generated_samples += 1
        self.label_counts[label] += 1
        if self._samples_bar is not None:
            self._samples_bar.update(1)
            self._samples_bar.set_postfix(
                last_label=label,
                errors=self.error_count,
            )
            return
        if self.generated_samples == 1 or self.generated_samples % 25 == 0:
            self._print_plain_sample(label)
        elif self.generated_samples == self.total_samples:
            self._print_plain_sample(label)

    def error(self, *, human_id: str, stage: str, message: str) -> None:
        self.error_count += 1
        self._write(f"ERROR human={human_id} stage={stage}: {message}")
        if self._samples_bar is not None:
            self._samples_bar.set_postfix(errors=self.error_count)

    def close(self) -> None:
        if self._samples_bar is not None:
            self._samples_bar.close()
        if self._profiles_bar is not None:
            self._profiles_bar.close()

    def _write(self, text: str) -> None:
        if self._tqdm is not None:
            self._tqdm.write(text)
        else:
            print(text)

    def _print_plain_sample(self, label: str) -> None:
        total = self.total_samples or "?"
        print(
            f"Samples: {self.generated_samples}/{total} "
            f"| last_label={label} | errors={self.error_count}"
        )


def get_or_generate_profile_summary(
    *,
    generator: QwenDecisionLabeler,
    profile_context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    traits = profile_context.get("traits_summary")
    if isinstance(traits, dict) and str(traits.get("res", "")).strip():
        return (
            {
                "source": "existing_coopera_traits_summary",
                "summary": str(traits.get("res", "")).strip(),
                "traits_summary_path": traits.get("path"),
                "big_five": (profile_context.get("mypersonality") or {}).get("big_five"),
            },
            {"stage": "profile_summary", "mode": "loaded_existing"},
        )

    payload, meta = generator.generate_json(
        system="You summarize COOPERA human profiles. Return only valid JSON.",
        user=build_profile_summary_prompt(profile_context),
    )
    summary = payload.get("summary")
    if isinstance(summary, str):
        summary = summary.strip()
    if not summary:
        raise ValueError("Profile summary stage returned empty summary.")
    return (
        {
            "source": "generated_from_mypersonality",
            "summary": summary,
            "big_five": (profile_context.get("mypersonality") or {}).get("big_five"),
            "profile_evidence": payload.get("profile_evidence", []),
        },
        {"stage": "profile_summary", **meta},
    )


def generate_preference_profile(
    *,
    generator: QwenDecisionLabeler,
    profile_summary: dict[str, Any],
    profile_context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, meta = generator.generate_json(
        system="You infer stable assistive-robot preferences. Return only valid JSON.",
        user=build_preference_profile_prompt(
            profile_summary=profile_summary,
            profile_context=profile_context,
        ),
    )
    stable_preferences = compact_snapshot(
        validate_snapshot(payload.get("stable_preferences", [])),
        max_items=8,
    )
    return (
        {
            "stable_preferences": stable_preferences,
            "profile_level_rationale": payload.get("profile_level_rationale", ""),
            "uncertain_or_omitted": payload.get("uncertain_or_omitted", []),
        },
        {"stage": "preference_profile", **meta},
    )


def generate_assistance_scenarios(
    *,
    generator: QwenDecisionLabeler,
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
    day: str,
    time_text: str,
    samples_per_hour: int,
    memory: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return generator.generate_json(
        system="You generate assistive-robot situations. Return only valid JSON.",
        user=build_scenario_prompt(
            profile_summary=profile_summary,
            preference_profile=preference_profile,
            day=day,
            time_text=time_text,
            samples_per_hour=samples_per_hour,
            memory=memory,
        ),
    )


def reflect_decision(
    *,
    generator: QwenDecisionLabeler,
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
    scenario: dict[str, Any],
    memory: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, meta = generator.generate_json(
        system="You reflect on robot decisions for a simulated human. Return only valid JSON.",
        user=build_decision_reflection_prompt(
            profile_summary=profile_summary,
            preference_profile=preference_profile,
            scenario=scenario,
            memory=memory,
        ),
    )
    label = str(payload.get("label_action", "")).strip()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label_action={label!r}")
    payload["preference_snapshot"] = compact_snapshot(
        validate_snapshot(payload.get("preference_snapshot", [])),
        max_items=5,
    )
    return payload, {"stage": "decision_reflection", **meta}


def build_profile_summary_prompt(profile_context: dict[str, Any]) -> str:
    profile = profile_context.get("mypersonality") or {}
    compact = {
        "human_id": profile_context.get("human_id"),
        "big_five": profile.get("big_five"),
        "profile_text_excerpt": str(profile.get("profile_text", ""))[:5000],
    }
    return (
        "Summarize this COOPERA human profile for later synthetic simulation.\n"
        "Focus on stable personality implications relevant to how an assistive "
        "robot should behave: tolerance for robot initiative and autonomous "
        "execution, desire to keep control, action immediacy vs routine adherence, "
        "sensitivity to interruption and context, need for prompting/explanation, "
        "and attitude to safety/risk. Do not create concrete robot tasks yet.\n\n"
        "Use cautious language. Big Five values around 2.5-3.5 are moderate, not low "
        "or high. Do not make clinical claims or strong claims that are not directly "
        "supported by the profile.\n\n"
        "Return JSON with keys: summary, profile_evidence. The summary value must be "
        "a JSON object, not a string.\n\n"
        f"PROFILE:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def build_preference_profile_prompt(
    *,
    profile_summary: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    allowed = sorted(PREFERENCE_SIGNALS)
    compact = {
        "human_id": profile_context.get("human_id"),
        "profile_summary": profile_summary,
        "big_five": (profile_context.get("mypersonality") or {}).get("big_five"),
    }
    return (
        "Infer a stable assistive-robot preference profile for this human.\n\n"
        "The taxonomy has 12 behavioral preference signals. Each preference is "
        "expressed with a polarity (prefer or avoid) and a weight from 1 to 10 "
        "(how strongly this person holds it). Signal names never contain 'prefer' "
        "or 'avoid' - polarity is a separate field.\n\n"
        "Signals and their meaning:\n"
        f"{taxonomy_signal_block()}\n\n"
        "Rules:\n"
        "- Use ONLY evidence from the profile summary and Big Five.\n"
        "- Do NOT infer preferences from a hypothetical current task or time.\n"
        "- Only assign signals clearly supported by the Big Five profile; not all "
        "12 need to be present.\n"
        "- Copy signal_name values exactly from the allowed list.\n"
        "- polarity is 'prefer' or 'avoid'; weight is an integer 1-10.\n"
        "- rationale is a one-sentence justification grounded in the personality traits.\n"
        "- Return at most 8 stable_preferences.\n"
        "- Omit weak or unsupported preferences; put them in uncertain_or_omitted.\n"
        "- Do not assign two signals from the same subcategory with opposite "
        "polarities (no contradictions on the same axis).\n\n"
        "Allowed signal_name values:\n"
        f"{json.dumps(allowed, ensure_ascii=False)}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "stable_preferences": [{"signal_name": "...", "polarity": "prefer|avoid", "weight": 7, "rationale": "..."}],\n'
        '  "profile_level_rationale": "...",\n'
        '  "uncertain_or_omitted": ["..."]\n'
        "}\n\n"
        f"HUMAN:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def build_scenario_prompt(
    *,
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
    day: str,
    time_text: str,
    samples_per_hour: int,
    memory: list[dict[str, Any]],
) -> str:
    return (
        "Generate assistive-robot decision situations for this simulated human.\n\n"
        "Rules:\n"
        "- Generate plausible situations, not preference labels.\n"
        "- action_text MUST be a concrete domestic task that both a human and a "
        "robot could physically perform (e.g. 'make breakfast', 'give medication "
        "to user', 'vacuum the living room', 'water the plants'). It must NOT "
        "describe the robot's response or framing (NOT 'offer to...', 'remind the "
        "user to...', 'suggest...'). The robot's decision is predicted separately.\n"
        "- Do NOT include preference_snapshot or label_action here.\n"
        "- Do NOT introduce strong context flags such as user_asleep, guests_present, "
        "adverse_weather, quiet_hours, or user_in_rush unless the profile summary "
        "or recent memory gives a reason.\n"
        "- Keep context internally consistent.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "scenarios": [\n'
        "    {\n"
        '      "action_input": {"action_text": "...", "activity": "..."},\n'
        '      "context_input": {"location_current": "...", "objects_nearby": [], "available_objects": [], "raw_conditions": [], "time_of_day": "...", "weekday": "synthetic_day_XX", "user_state": [], "environment_flags": []},\n'
        '      "structured_task_features": {"kind": "...", "urgency": "low|medium|high", "sensitivity": "low|medium|high", "user_busy": false, "quiet_hours": false, "conditions": [], "context_flags": {}} ,\n'
        '      "scenario_rationale": "why this situation is plausible for the profile"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Generate {samples_per_hour} scenario(s) for day {day}, time {time_text}.\n\n"
        f"PROFILE SUMMARY:\n{json.dumps(profile_summary, ensure_ascii=False, indent=2)}\n\n"
        f"STABLE PREFERENCES:\n{json.dumps(preference_profile, ensure_ascii=False, indent=2)}\n\n"
        f"RECENT MEMORY:\n{json.dumps(memory[-8:], ensure_ascii=False, indent=2)}"
    )


def build_decision_reflection_prompt(
    *,
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
    scenario: dict[str, Any],
    memory: list[dict[str, Any]],
) -> str:
    return (
        "Reflect on the robot decision for this scenario.\n\n"
        "Rules:\n"
        "- Choose label_action from: do_now, do_later, tell_the_user, no_action.\n"
        "  - do_now: robot should execute/help now.\n"
        "  - do_later: robot should postpone the action.\n"
        "  - tell_the_user: robot should tell or ask the user rather than executing.\n"
        "  - no_action: robot should stay passive.\n"
        "- preference_snapshot must be a subset of the stable preferences unless "
        "there is direct profile evidence in the summary.\n"
        "- Do not create new context conditions.\n"
        "- Each preference_snapshot item keeps the polarity and weight (1-10) it "
        "had in the stable preferences.\n"
        "- Copy signal_name values exactly from the stable preferences or allowed "
        "taxonomy.\n"
        "- Include at most 5 preference_snapshot items, only those directly relevant "
        "to the scenario decision.\n"
        "- If the scenario conflicts with the profile or lacks enough evidence, "
        "prefer conservative labels such as tell_the_user or no_action.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "preference_snapshot": [{"signal_name": "...", "polarity": "prefer|avoid", "weight": 7}],\n'
        '  "label_action": "do_now|do_later|tell_the_user|no_action",\n'
        '  "decision_rationale": "...",\n'
        '  "consistency_checks": ["..."]\n'
        "}\n\n"
        f"PROFILE SUMMARY:\n{json.dumps(profile_summary, ensure_ascii=False, indent=2)}\n\n"
        f"STABLE PREFERENCES:\n{json.dumps(preference_profile, ensure_ascii=False, indent=2)}\n\n"
        f"SCENARIO:\n{json.dumps(scenario, ensure_ascii=False, indent=2)}\n\n"
        f"RECENT MEMORY:\n{json.dumps(memory[-8:], ensure_ascii=False, indent=2)}"
    )


def build_sample_from_stages(
    *,
    profile_context: dict[str, Any],
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
    scenario: dict[str, Any],
    scenario_metadata: dict[str, Any],
    decision_payload: dict[str, Any],
    decision_metadata: dict[str, Any],
    day: str,
    time_text: str,
    scenario_idx: int,
    existing_count: int,
) -> dict[str, Any]:
    human_id = str(profile_context["human_id"])
    user_id = int(human_id) + 1 if human_id.isdigit() else existing_count + 1
    label = str(decision_payload.get("label_action", "")).strip()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label_action={label!r}")
    snapshot = validate_snapshot(decision_payload.get("preference_snapshot", []))
    sample_id = (
        f"preference_human_sim:{human_id}:{day}:{safe_id(time_text)}:"
        f"{existing_count + scenario_idx}"
    )
    return {
        "sample_id": sample_id,
        "user_id": user_id,
        "user_external_id": f"coopera_human_{human_id}",
        "label_action": label,
        "action_input": normalize_action_input(scenario.get("action_input")),
        "context_input": normalize_context_input(scenario.get("context_input"), day=day),
        "structured_task_features": normalize_structured_features(
            scenario.get("structured_task_features")
        ),
        "preference_snapshot": snapshot,
        "source_metadata": {
            "source": "preference_human_sim_multistage_qwen",
            "human_id": human_id,
            "profile_index": profile_context.get("profile_index"),
            "day": day,
            "time_text": time_text,
            "scenario_rationale": scenario.get("scenario_rationale"),
            "decision_rationale": decision_payload.get("decision_rationale"),
            "consistency_checks": decision_payload.get("consistency_checks", []),
            "traits_summary_path": profile_summary.get("traits_summary_path"),
            "mypersonality_authid": (
                (profile_context.get("mypersonality") or {}).get("authid")
                if isinstance(profile_context.get("mypersonality"), dict)
                else None
            ),
            "stable_preference_profile": preference_profile,
            "scenario_metadata": scenario_metadata,
            "decision_metadata": decision_metadata,
        },
        "data_provenance": {
            "action_input": "synthetic_qwen_scenario_from_coopera_profile",
            "context_input": "synthetic_qwen_scenario_from_coopera_profile",
            "structured_task_features": "synthetic_qwen_scenario_from_coopera_profile",
            "preference_snapshot": "synthetic_qwen_decision_reflection_from_stable_profile",
            "label_action": "synthetic_qwen_decision_reflection_from_stable_profile",
        },
    }


def normalize_action_input(value: Any) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    return {
        "action_text": str(value.get("action_text", "")).strip(),
        "activity": str(value.get("activity", "")).strip() or None,
    }


def normalize_context_input(value: Any, *, day: str) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    return {
        "location_current": _optional_str(value.get("location_current")),
        "objects_nearby": _as_str_list(value.get("objects_nearby")),
        "available_objects": _as_str_list(value.get("available_objects")),
        "raw_conditions": _as_str_list(value.get("raw_conditions")),
        "time_of_day": str(value.get("time_of_day", "unknown")).strip() or "unknown",
        "weekday": str(value.get("weekday", f"synthetic_day_{day}")).strip()
        or f"synthetic_day_{day}",
        "user_state": _as_str_list(value.get("user_state")),
        "environment_flags": _as_str_list(value.get("environment_flags")),
    }


def normalize_structured_features(value: Any) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    return {
        "kind": str(value.get("kind", "daily_living_support")).strip()
        or "daily_living_support",
        "urgency": str(value.get("urgency", "medium")).strip() or "medium",
        "sensitivity": str(value.get("sensitivity", "medium")).strip() or "medium",
        "user_busy": bool(value.get("user_busy", False)),
        "quiet_hours": bool(value.get("quiet_hours", False)),
        "conditions": _as_str_list(value.get("conditions")),
        "context_flags": value.get("context_flags")
        if isinstance(value.get("context_flags"), dict)
        else {},
    }


def validate_snapshot(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("preference_snapshot must be a list.")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        signal = canonical_signal_name(str(item.get("signal_name", "")).strip())
        polarity = str(item.get("polarity", "prefer")).strip().lower()
        if signal not in PREFERENCE_SIGNALS:
            raise ValueError(f"Unknown preference signal: {signal!r}")
        if polarity not in {"prefer", "avoid"}:
            raise ValueError(f"Invalid polarity for {signal!r}: {polarity!r}")
        if signal in seen:
            continue
        seen.add(signal)
        row: dict[str, Any] = {
            "signal_name": signal,
            "polarity": polarity,
            "weight": coerce_weight(item.get("weight")),
        }
        rationale = str(item.get("rationale", "")).strip()
        if rationale:
            row["rationale"] = rationale
        out.append(row)
    return out


def coerce_weight(value: Any) -> float:
    """Parse a preference weight into [1, 10], defaulting when missing/invalid."""
    if value is None or value == "":
        return DEFAULT_PREFERENCE_WEIGHT
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return DEFAULT_PREFERENCE_WEIGHT
    return max(1.0, min(10.0, weight))


def compact_snapshot(
    snapshot: list[dict[str, Any]],
    *,
    max_items: int,
) -> list[dict[str, Any]]:
    """Keep the training signal focused when Qwen over-selects taxonomy entries."""
    compact: list[dict[str, Any]] = []
    used_axes: set[str] = set()
    for item in snapshot:
        axis = preference_axis(item["signal_name"])
        if axis in used_axes:
            continue
        used_axes.add(axis)
        compact.append(item)
        if len(compact) >= max_items:
            break
    return compact


def preference_axis(signal: str) -> str:
    """Preference axis for a signal = its taxonomy subcategory (e.g. 'control')."""
    return signal_subcategory(signal) or signal


def canonical_signal_name(signal: str) -> str:
    """Normalize an LLM-provided signal name against the closed taxonomy."""
    normalized = signal.strip().lower()
    if normalized in PREFERENCE_SIGNALS:
        return normalized
    return signal


def estimate_calls(
    *,
    profiles: list[dict[str, Any]],
    selected_indices: list[int],
    results_dir: Path,
    response_source: str,
    days: list[str],
    times: list[str],
    samples_per_hour: int,
) -> int:
    calls = 0
    for profile_index in selected_indices:
        human_id = str(profile_index).zfill(5)
        traits = load_latest_traits_summary(
            results_dir=results_dir,
            response_source=response_source,
            human_id=human_id,
        )
        if not traits:
            calls += 1  # profile_summary
        calls += 1  # preference_profile
        calls += len(days) * len(times)  # scenario proposal per hour
        calls += len(days) * len(times) * samples_per_hour  # decision reflection
    return calls


def select_profile_indices(
    *,
    total: int,
    explicit: list[int] | None,
    max_profiles: int,
) -> list[int]:
    if explicit is not None:
        return [idx for idx in explicit if 0 <= idx < total]
    return list(range(min(total, max_profiles)))


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_intermediate(
    base_dir: Path,
    human_id: str,
    name: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    target = base_dir / human_id
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{name}.json").write_text(
        json.dumps({"payload": payload, "metadata": metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize(
    *,
    samples: list[dict[str, Any]],
    errors: list[dict[str, str]],
    args: argparse.Namespace,
    plan: dict[str, Any],
) -> dict[str, Any]:
    labels = Counter(str(sample.get("label_action")) for sample in samples)
    users = Counter(str(sample.get("user_external_id")) for sample in samples)
    activities = Counter(
        str((sample.get("action_input") or {}).get("activity")) for sample in samples
    )
    signals = Counter(
        row.get("signal_name")
        for sample in samples
        for row in sample.get("preference_snapshot", [])
    )
    return {
        "num_samples": len(samples),
        "num_errors": len(errors),
        "errors_preview": errors[:20],
        "labels": dict(labels),
        "users": dict(users),
        "activities": dict(activities),
        "top_preference_signals": dict(signals.most_common(20)),
        "plan": plan,
        "output": str(args.output),
        "summary_output": str(args.summary_output),
    }


def safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text))


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":
    main()
