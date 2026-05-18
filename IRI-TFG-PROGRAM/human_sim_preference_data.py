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
from preference_taxonomy import PREFERENCE_SIGNALS, VALID_LABELS
from qwen_labeler import QwenDecisionLabeler


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
    print(json.dumps(plan, ensure_ascii=False, indent=2))

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
    try:
        for profile_index in selected_indices:
            profile = profiles[profile_index]
            human_id = str(profile_index).zfill(5)
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
                profile_summary, profile_summary_meta = get_or_generate_profile_summary(
                    generator=generator,
                    profile_context=profile_context,
                )
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
                continue

            for day in days:
                memory: list[dict[str, Any]] = []
                for time_text in times:
                    try:
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
    finally:
        generator.close()

    write_jsonl(args.output, samples)
    summary = summarize(samples=samples, errors=errors, args=args, plan=plan)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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
    summary = str(payload.get("summary", "")).strip()
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
    stable_preferences = validate_snapshot(payload.get("stable_preferences", []))
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
    payload["preference_snapshot"] = validate_snapshot(payload.get("preference_snapshot", []))
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
        "Focus on stable personality implications: social style, routine style, "
        "autonomy/control, intrusiveness tolerance, reminder style, likely support "
        "domains, and uncertainty. Do not create concrete robot tasks yet.\n\n"
        "Return JSON with keys: summary, profile_evidence.\n\n"
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
        "Rules:\n"
        "- Use ONLY evidence from the profile summary and Big Five.\n"
        "- Do NOT infer preferences from a hypothetical current task or time.\n"
        "- For signals whose name starts with avoid_, use polarity='prefer' when "
        "the human prefers that avoidance rule.\n"
        "- Copy signal_name values exactly from the allowed list. Do not create "
        "variants such as prefer_prefer_* or avoid_avoid_*.\n"
        "- Omit weak or unsupported preferences.\n\n"
        "Allowed signal_name values:\n"
        f"{json.dumps(allowed, ensure_ascii=False)}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "stable_preferences": [{"signal_name": "...", "polarity": "prefer|avoid"}],\n'
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
        "- Choose label_action from: do_now, do_later, remind, no_action.\n"
        "- preference_snapshot must be a subset of the stable preferences unless "
        "there is direct profile evidence in the summary.\n"
        "- Do not create new context conditions.\n"
        "- For avoid_* signals, polarity should usually be 'prefer' when the user "
        "prefers that avoidance rule.\n"
        "- Copy signal_name values exactly from the stable preferences or allowed "
        "taxonomy. Do not add extra prefixes.\n"
        "- If the scenario conflicts with the profile or lacks enough evidence, "
        "prefer conservative labels such as remind or no_action.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "preference_snapshot": [{"signal_name": "...", "polarity": "prefer|avoid"}],\n'
        '  "label_action": "do_now|do_later|remind|no_action",\n'
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


def validate_snapshot(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("preference_snapshot must be a list.")
    out: list[dict[str, str]] = []
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
        if signal.startswith("avoid_") and polarity == "avoid":
            polarity = "prefer"
        if signal in seen:
            continue
        seen.add(signal)
        out.append({"signal_name": signal, "polarity": polarity})
    return out


def canonical_signal_name(signal: str) -> str:
    """Repair common LLM near-misses while keeping the taxonomy closed."""
    candidates = [signal]
    if signal.startswith("prefer_prefer_"):
        candidates.append(signal.replace("prefer_prefer_", "prefer_", 1))
    if signal.startswith("avoid_avoid_"):
        candidates.append(signal.replace("avoid_avoid_", "avoid_", 1))
    if signal.startswith("prefer_avoid_"):
        candidates.append(signal.replace("prefer_avoid_", "avoid_", 1))
    if signal.startswith("avoid_prefer_"):
        candidates.append(signal.replace("avoid_prefer_", "prefer_", 1))
    for candidate in candidates:
        if candidate in PREFERENCE_SIGNALS:
            return candidate
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
