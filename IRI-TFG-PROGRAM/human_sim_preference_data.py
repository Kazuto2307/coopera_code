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
    default_output = (
        program_dir
        / "generated_data"
        / f"preference_human_sim_{stamp}.jsonl"
    )
    default_summary = (
        program_dir
        / "generated_data"
        / f"preference_human_sim_{stamp}_summary.json"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Human-sim equivalent for the preference model: simulate humans from "
            "COOPERA personalities and directly generate decision-training JSONL."
        )
    )
    parser.add_argument("--coopera-root", type=Path, default=coopera_root)
    parser.add_argument(
        "--response-source",
        choices=["gpt_response", "llama_response"],
        default="gpt_response",
        help="Where to look for optional COOPERA traits_summary files.",
    )
    parser.add_argument("--profile-indices", type=int, nargs="+", default=None)
    parser.add_argument("--max-profiles", type=int, default=10)
    parser.add_argument("--max-days", type=int, default=1)
    parser.add_argument("--times", nargs="*", default=None)
    parser.add_argument("--samples-per-hour", type=int, default=1)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary-output", type=Path, default=default_summary)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--qwen-temperature", type=float, default=0.25)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1800)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned generation loops without loading Qwen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coopera_root = args.coopera_root.resolve()
    results_dir = coopera_root / "results"
    mypersonality_path = (
        coopera_root
        / "data"
        / "humanoids"
        / "humanoid_data"
        / "mypersonality_final.csv"
    )
    profiles = load_mypersonality_profiles(mypersonality_path)
    selected_indices = select_profile_indices(
        total=len(profiles),
        explicit=args.profile_indices,
        max_profiles=args.max_profiles,
    )
    times = args.times or DEFAULT_TIMES
    days = [str(i).zfill(2) for i in range(args.max_days)]

    plan = {
        "coopera_root": str(coopera_root),
        "mypersonality_path": str(mypersonality_path),
        "num_profiles_available": len(profiles),
        "profile_indices": selected_indices,
        "days": days,
        "times": times,
        "samples_per_hour": args.samples_per_hour,
        "planned_qwen_calls": len(selected_indices) * len(days) * len(times),
        "output": str(args.output),
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

            for day in days:
                memory: list[dict[str, Any]] = []
                for time_text in times:
                    try:
                        payload, metadata = generator.generate_json(
                            system=(
                                "You generate synthetic training data for an "
                                "assistive-robot decision personalization model. "
                                "Return only valid JSON."
                            ),
                            user=build_generation_prompt(
                                profile_context=profile_context,
                                day=day,
                                time_text=time_text,
                                samples_per_hour=args.samples_per_hour,
                                memory=memory,
                            ),
                        )
                        new_samples = normalize_generated_samples(
                            payload=payload,
                            metadata=metadata,
                            profile_context=profile_context,
                            day=day,
                            time_text=time_text,
                            existing_count=len(samples),
                        )
                        samples.extend(new_samples)
                        memory.extend(
                            {
                                "time": time_text,
                                "action_input": sample["action_input"],
                                "context_input": sample["context_input"],
                                "label_action": sample["label_action"],
                                "preference_snapshot": sample["preference_snapshot"],
                            }
                            for sample in new_samples
                        )
                    except Exception as exc:
                        errors.append(
                            {
                                "human_id": human_id,
                                "day": day,
                                "time": time_text,
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


def build_generation_prompt(
    *,
    profile_context: dict[str, Any],
    day: str,
    time_text: str,
    samples_per_hour: int,
    memory: list[dict[str, Any]],
) -> str:
    allowed_signals = sorted(PREFERENCE_SIGNALS)
    compact_profile = {
        "human_id": profile_context["human_id"],
        "mypersonality": profile_context.get("mypersonality"),
        "traits_summary": profile_context.get("traits_summary"),
    }
    memory_tail = memory[-8:]
    return (
        "Simulate this COOPERA human as a potential user of an assistive robot. "
        "Generate model-ready training samples directly, not simulator actions.\n\n"
        "CRITICAL RULES:\n"
        "- Personal preferences must come from the human profile, Big Five scores, "
        "traits summary, and simulated personal history. Do not infer preferences "
        "just because of the current task, object, room, or time.\n"
        "- Example: breakfast at 9 am does not imply prefer_morning_tasks unless "
        "the profile supports that preference.\n"
        "- The robot decision target must reflect what this specific human would "
        "want the robot to do in the situation.\n"
        "- Use only the allowed preference signal names.\n"
        "- Return valid JSON only.\n\n"
        "Allowed label_action values: do_now, do_later, remind, no_action.\n"
        "Allowed preference signal_name values:\n"
        f"{json.dumps(allowed_signals, ensure_ascii=False)}\n\n"
        "Output schema:\n"
        "{\n"
        '  "samples": [\n'
        "    {\n"
        '      "action_input": {"action_text": "...", "activity": "..."},\n'
        '      "context_input": {\n'
        '        "location_current": "...",\n'
        '        "objects_nearby": ["..."],\n'
        '        "available_objects": ["..."],\n'
        '        "raw_conditions": ["..."],\n'
        '        "time_of_day": "morning|afternoon|evening|night|unknown",\n'
        '        "weekday": "synthetic_day_XX",\n'
        '        "user_state": ["busy|in_rush|asleep|nearby|injured_or_disabled"],\n'
        '        "environment_flags": ["quiet_hours|guests_present|weekend|adverse_weather"]\n'
        "      },\n"
        '      "structured_task_features": {\n'
        '        "kind": "routine_reminder|safety|daily_living_support|social_support|cognitive_support",\n'
        '        "urgency": "low|medium|high",\n'
        '        "sensitivity": "low|medium|high",\n'
        '        "user_busy": false,\n'
        '        "quiet_hours": false,\n'
        '        "conditions": ["..."],\n'
        '        "context_flags": {"user_in_rush": false, "user_asleep": false, "guests_present": false}\n'
        "      },\n"
        '      "preference_snapshot": [{"signal_name": "...", "polarity": "prefer|avoid"}],\n'
        '      "label_action": "do_now|do_later|remind|no_action",\n'
        '      "rationale": "short profile-grounded explanation"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Generate {samples_per_hour} sample(s) for day {day}, time {time_text}.\n\n"
        "HUMAN PROFILE CONTEXT:\n"
        f"{json.dumps(compact_profile, ensure_ascii=False, indent=2)}\n\n"
        "RECENT SYNTHETIC HISTORY FOR THIS HUMAN TODAY:\n"
        f"{json.dumps(memory_tail, ensure_ascii=False, indent=2)}"
    )


def normalize_generated_samples(
    *,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    profile_context: dict[str, Any],
    day: str,
    time_text: str,
    existing_count: int,
) -> list[dict[str, Any]]:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("Qwen response must contain a top-level 'samples' list.")

    out: list[dict[str, Any]] = []
    human_id = str(profile_context["human_id"])
    user_id = int(human_id) + 1 if human_id.isdigit() else existing_count + 1
    for idx, raw in enumerate(raw_samples, start=1):
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label_action", "")).strip()
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label_action={label!r}")
        snapshot = validate_snapshot(raw.get("preference_snapshot", []))
        sample_id = (
            f"preference_human_sim:{human_id}:{day}:"
            f"{safe_id(time_text)}:{existing_count + idx}"
        )
        action_input = normalize_action_input(raw.get("action_input"))
        context_input = normalize_context_input(raw.get("context_input"), day=day)
        structured = normalize_structured_features(raw.get("structured_task_features"))
        out.append(
            {
                "sample_id": sample_id,
                "user_id": user_id,
                "user_external_id": f"coopera_human_{human_id}",
                "label_action": label,
                "action_input": action_input,
                "context_input": context_input,
                "structured_task_features": structured,
                "preference_snapshot": snapshot,
                "source_metadata": {
                    "source": "preference_human_sim_qwen_profile",
                    "human_id": human_id,
                    "profile_index": profile_context.get("profile_index"),
                    "day": day,
                    "time_text": time_text,
                    "rationale": raw.get("rationale"),
                    "traits_summary_path": (
                        (profile_context.get("traits_summary") or {}).get("path")
                        if isinstance(profile_context.get("traits_summary"), dict)
                        else None
                    ),
                    "mypersonality_authid": (
                        (profile_context.get("mypersonality") or {}).get("authid")
                        if isinstance(profile_context.get("mypersonality"), dict)
                        else None
                    ),
                    "qwen_metadata": metadata,
                },
                "data_provenance": {
                    "action_input": "synthetic_qwen_from_coopera_profile",
                    "context_input": "synthetic_qwen_from_coopera_profile",
                    "structured_task_features": "synthetic_qwen_from_coopera_profile",
                    "preference_snapshot": "synthetic_qwen_from_coopera_profile",
                    "label_action": "synthetic_qwen_from_coopera_profile",
                },
            }
        )
    return out


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
        signal = str(item.get("signal_name", "")).strip()
        polarity = str(item.get("polarity", "prefer")).strip().lower()
        if signal not in PREFERENCE_SIGNALS:
            raise ValueError(f"Unknown preference signal: {signal!r}")
        if polarity not in {"prefer", "avoid"}:
            raise ValueError(f"Invalid polarity for {signal!r}: {polarity!r}")
        if signal in seen:
            continue
        seen.add(signal)
        out.append({"signal_name": signal, "polarity": polarity})
    return out


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

