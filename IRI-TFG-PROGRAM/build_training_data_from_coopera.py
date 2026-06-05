from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from coopera_profile_loader import (
    load_human_profile_context,
    profile_context_is_available,
)
from coopera_plan_parser import (
    CooperaPlan,
    CooperaTask,
    iter_predicates_reflection_files,
    parse_predicates_reflection_file,
)
from preference_taxonomy import DOMAIN_KEYWORDS, PREFERENCE_SIGNALS, VALID_LABELS


def parse_args() -> argparse.Namespace:
    program_dir = Path(__file__).resolve().parent
    coopera_root = program_dir.parent
    default_results = coopera_root / "results"
    default_output = program_dir / "generated_data" / "training_samples_from_coopera.jsonl"
    default_summary = program_dir / "generated_data" / "training_samples_from_coopera_summary.json"

    parser = argparse.ArgumentParser(
        description=(
            "Build decision-personalization JSONL samples from COOPERA synthetic "
            "human plans."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument(
        "--mypersonality-path",
        type=Path,
        default=None,
        help="Optional explicit path to mypersonality_final.csv.",
    )
    parser.add_argument(
        "--response-source",
        choices=["gpt_response", "llama_response"],
        default="gpt_response",
        help="COOPERA result branch to read. The name is historical; current code may use Qwen locally.",
    )
    parser.add_argument("--collab-type", type=int, choices=[1, 2], default=1)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary-output", type=Path, default=default_summary)
    parser.add_argument("--human-ids", nargs="*", default=None)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--days", nargs="*", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--first-task-only",
        action="store_true",
        help="Use only task 1 from each COOPERA hour, matching the current renderer behavior.",
    )
    parser.add_argument(
        "--generation-strategy",
        choices=["qwen_profile", "rules_debug"],
        default="qwen_profile",
        help=(
            "qwen_profile grounds preference_snapshot and label_action in the "
            "COOPERA human profile. rules_debug is only for parser/debug checks."
        ),
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--qwen-temperature", type=float, default=0.2)
    parser.add_argument(
        "--allow-missing-profile",
        action="store_true",
        help="Allow qwen_profile to run even if traits_summary/mypersonality is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_ids = _normalize_human_ids(args.human_ids)
    days = _normalize_days(args.days)
    scene_ids = set(args.scene_ids) if args.scene_ids else None

    profile_cache: dict[str, dict[str, Any]] = {}
    generator = None
    if args.generation_strategy == "qwen_profile":
        from qwen_labeler import QwenDecisionLabeler

        generator = QwenDecisionLabeler(
            model_name=args.qwen_model,
            temperature=args.qwen_temperature,
        )

    samples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    file_paths = list(
        iter_predicates_reflection_files(
            results_dir=args.results_dir,
            response_source=args.response_source,
            collab_type=args.collab_type,
            human_ids=human_ids,
            scene_ids=scene_ids,
            days=days,
            max_files=args.max_files,
        )
    )

    for json_path in file_paths:
        try:
            plan = parse_predicates_reflection_file(
                json_path,
                collab_type=args.collab_type,
            )
            profile_context = profile_cache.get(plan.human_id)
            if profile_context is None:
                profile_context = load_human_profile_context(
                    coopera_root=Path(__file__).resolve().parent.parent,
                    results_dir=args.results_dir,
                    response_source=args.response_source,
                    human_id=plan.human_id,
                    mypersonality_path=args.mypersonality_path,
                )
                profile_cache[plan.human_id] = profile_context

            if (
                args.generation_strategy == "qwen_profile"
                and not args.allow_missing_profile
                and not profile_context_is_available(profile_context)
            ):
                raise ValueError(
                    f"Missing COOPERA profile context for human_id={plan.human_id}. "
                    "Run human_sim.py traits generation first, or pass --allow-missing-profile."
                )

            plan_tasks = plan.tasks[:1] if args.first_task_only else plan.tasks
            for task in plan_tasks:
                sample = build_sample_from_task(
                    plan,
                    task,
                    collab_type=args.collab_type,
                    generation_strategy=args.generation_strategy,
                )
                if generator is not None:
                    snapshot, label, label_metadata = generator.synthesize_preferences_and_label(
                        sample=sample,
                        human_profile_context=profile_context,
                    )
                    sample["preference_snapshot"] = snapshot
                    sample["label_action"] = label
                    sample["source_metadata"]["generation_strategy"] = "qwen_profile"
                    sample["source_metadata"]["profile_context"] = {
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
                    }
                    sample["source_metadata"]["qwen_profile_metadata"] = label_metadata
                    sample["data_provenance"]["preference_snapshot"] = "synthetic_qwen_from_coopera_profile"
                    sample["data_provenance"]["label_action"] = "synthetic_qwen_from_coopera_profile"
                samples.append(sample)
                if args.max_samples is not None and len(samples) >= args.max_samples:
                    break
        except Exception as exc:
            errors.append({"path": str(json_path), "error": f"{type(exc).__name__}: {exc}"})
        if args.max_samples is not None and len(samples) >= args.max_samples:
            break

    if generator is not None:
        generator.close()

    write_jsonl(args.output, samples)
    summary = summarize(samples=samples, errors=errors, args=args, files_seen=len(file_paths))
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_sample_from_task(
    plan: CooperaPlan,
    task: CooperaTask,
    *,
    collab_type: int,
    generation_strategy: str,
) -> dict[str, Any]:
    user_num = _human_id_to_user_id(plan.human_id)
    time_of_day = time_bucket(plan.time_text)
    room = infer_location(plan, task, collab_type=collab_type)
    objects = infer_objects(task, collab_type=collab_type)
    action_text = action_text_from_task(task, collab_type=collab_type)
    activity = infer_activity(
        text=" ".join([action_text, plan.intention, task.thought, room, " ".join(objects)]),
    )
    features = infer_structured_features(
        plan=plan,
        task=task,
        activity=activity,
        time_of_day=time_of_day,
    )
    if generation_strategy == "rules_debug":
        preference_snapshot = infer_preference_snapshot_rules_debug(
            plan=plan,
            task=task,
            activity=activity,
            time_of_day=time_of_day,
            features=features,
        )
        label = infer_label_action_rules_debug(
            activity=activity,
            time_of_day=time_of_day,
            features=features,
            preference_snapshot=preference_snapshot,
            text=" ".join([action_text, plan.intention, task.thought]).lower(),
        )
    else:
        preference_snapshot = []
        label = "no_action"

    return {
        "sample_id": (
            f"coopera:{plan.human_id}:{plan.scene_id}:{plan.day}:"
            f"{_safe_id(plan.hour_folder)}:task{task.index}"
        ),
        "user_id": user_num,
        "user_external_id": f"coopera_human_{plan.human_id}",
        "label_action": label,
        "action_input": {
            "action_text": action_text,
            "activity": activity,
        },
        "context_input": {
            "location_current": room or "unknown",
            "objects_nearby": objects,
            "available_objects": sorted(set(objects)),
            "raw_conditions": infer_raw_conditions(plan=plan, task=task, features=features),
            "time_of_day": time_of_day,
            "weekday": f"synthetic_day_{plan.day}",
            "user_state": infer_user_state(features),
            "environment_flags": infer_environment_flags(features),
        },
        "structured_task_features": features,
        "preference_snapshot": preference_snapshot,
        "source_metadata": {
            "source": "coopera_synthetic_llm",
            "source_path": str(plan.source_path),
            "response_source": _response_source_from_path(plan.source_path),
            "collab_type": collab_type,
            "human_id": plan.human_id,
            "scene_id": plan.scene_id,
            "day": plan.day,
            "hour_folder": plan.hour_folder,
            "time_text": plan.time_text,
            "coopera_intention": plan.intention,
            "coopera_thought": task.thought,
            "coopera_act": task.act,
            "generation_strategy": generation_strategy,
        },
        "data_provenance": {
            "action_input": "derived_from_coopera_act_and_thought",
            "context_input": "derived_from_coopera_time_scene_objects",
            "structured_task_features": "synthetic_rules_from_coopera_text",
            "preference_snapshot": (
                "synthetic_rules_debug" if generation_strategy == "rules_debug" else "pending_qwen_profile"
            ),
            "label_action": (
                "synthetic_rules_debug" if generation_strategy == "rules_debug" else "pending_qwen_profile"
            ),
        },
    }


def action_text_from_task(task: CooperaTask, *, collab_type: int) -> str:
    if collab_type == 1:
        static_name = str(task.act[1])
        dynamic_name = str(task.act[3])
        return f"place {dynamic_name} on {static_name}"
    action_type = int(task.act[0])
    inter_name = str(task.act[2])
    inhand_name = str(task.act[3])
    motion = str(task.act[4])
    if action_type == 1:
        return f"support human motion '{motion}' near {inter_name}"
    if action_type == 2:
        return f"pick up {inter_name}"
    if action_type == 3:
        return f"place or interact with {inter_name}"
    return f"assist with {motion or inter_name}"


def infer_objects(task: CooperaTask, *, collab_type: int) -> list[str]:
    if collab_type == 1:
        return [str(task.act[1]), str(task.act[3])]
    values = [str(task.act[2]), str(task.act[3]), str(task.act[4])]
    return [value for value in values if value and value.lower() not in {"none", "null"}]


def infer_location(plan: CooperaPlan, task: CooperaTask, *, collab_type: int) -> str:
    text = " ".join([plan.intention, task.thought, " ".join(map(str, task.act))]).lower()
    room_keywords = {
        "kitchen": "kitchen",
        "living room": "living room",
        "bedroom": "bedroom",
        "bathroom": "bathroom",
        "garage": "garage",
        "laundry": "laundryroom",
        "hallway": "hallway",
    }
    for key, value in room_keywords.items():
        if key in text:
            return value
    return "unknown"


def infer_activity(*, text: str) -> str:
    lower = text.lower()
    for signal, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            if signal == "prefer_meal_support":
                return "meal support"
            if signal == "prefer_housekeeping_support":
                return "housekeeping"
            if signal == "prefer_medication_support":
                return "medication support"
            if signal == "prefer_mobility_support":
                return "mobility support"
            if signal == "prefer_social_activity_support":
                return "social support"
            if signal == "prefer_cognitive_training_support":
                return "cognitive or leisure support"
    return "daily living support"


def infer_structured_features(
    *,
    plan: CooperaPlan,
    task: CooperaTask,
    activity: str,
    time_of_day: str,
) -> dict[str, Any]:
    text = " ".join([plan.intention, task.thought, activity]).lower()
    quiet_hours = time_of_day == "night"
    user_busy = any(word in text for word in ["rush", "busy", "late", "urgent"])
    safety = any(word in text for word in ["fall", "danger", "risk", "injury", "medicine"])

    if safety:
        kind = "safety"
        urgency = "high"
        sensitivity = "high"
    elif "remind" in text or "plan" in text or "planning" in text:
        kind = "routine_reminder"
        urgency = "medium"
        sensitivity = "medium"
    elif activity in {"meal support", "housekeeping", "mobility support"}:
        kind = "routine_reminder"
        urgency = "medium"
        sensitivity = "low"
    else:
        kind = "daily_living_support"
        urgency = "medium"
        sensitivity = "medium"

    if user_busy:
        urgency = "high"

    context_flags = {
        "adverse_weather": "weather" in text or "rain" in text,
        "early_morning": time_of_day == "morning",
        "guests_present": "guest" in text or "guests" in text,
        "user_asleep": "sleep" in text or "asleep" in text,
        "user_in_rush": user_busy,
        "user_injured_or_disabled": "injur" in text or "disabled" in text,
        "user_nearby": True,
        "weekend": False,
    }

    return {
        "kind": kind,
        "urgency": urgency,
        "sensitivity": sensitivity,
        "user_busy": user_busy,
        "quiet_hours": quiet_hours,
        "conditions": _conditions_from_flags(context_flags),
        "context_flags": context_flags,
    }


def infer_preference_snapshot_rules_debug(
    *,
    plan: CooperaPlan,
    task: CooperaTask,
    activity: str,
    time_of_day: str,
    features: dict[str, Any],
) -> list[dict[str, str]]:
    text = " ".join([plan.intention, task.thought, activity, " ".join(map(str, task.act))]).lower()
    signals: list[tuple[str, str]] = []

    if time_of_day == "morning":
        signals.append(("prefer_morning_tasks", "prefer"))
    elif time_of_day == "afternoon":
        signals.append(("prefer_afternoon_tasks", "prefer"))
    elif time_of_day == "evening":
        signals.append(("prefer_evening_tasks", "prefer"))
    elif time_of_day == "night":
        signals.append(("avoid_nighttime_notifications", "prefer"))

    for signal_name, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            signals.append((signal_name, "prefer"))

    if features.get("user_busy"):
        signals.append(("prefer_assistance_when_user_busy", "prefer"))
    if features.get("quiet_hours"):
        signals.append(("avoid_interrupt_during_quiet_hours", "prefer"))

    if "cozy" in text or "quiet" in text or "slow" in text:
        signals.append(("prefer_non_intrusive_assistance", "prefer"))
    if "creative" in text or "playful" in text or "flexible" in text:
        signals.append(("prefer_flexible_routine", "prefer"))
        signals.append(("prefer_explain_alternatives", "prefer"))
    if "conscientious" in text or "routine" in text:
        signals.append(("prefer_proactive_reminders", "prefer"))

    if not any(signal == "prefer_proactive_assistance" for signal, _ in signals):
        signals.append(("prefer_proactive_assistance", "prefer"))

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for signal, polarity in signals:
        if signal in seen or signal not in PREFERENCE_SIGNALS:
            continue
        seen.add(signal)
        deduped.append({"signal_name": signal, "polarity": polarity})
    return deduped


def infer_label_action_rules_debug(
    *,
    activity: str,
    time_of_day: str,
    features: dict[str, Any],
    preference_snapshot: list[dict[str, str]],
    text: str,
) -> str:
    signals = {row["signal_name"] for row in preference_snapshot}
    if features.get("context_flags", {}).get("user_asleep"):
        return "no_action"
    if features.get("quiet_hours") or time_of_day == "night":
        if "avoid_nighttime_notifications" in signals:
            return "do_later"
        return "no_action"
    if features.get("kind") == "safety":
        return "do_now"
    if "remind" in text and "prefer_proactive_reminders" in signals:
        return "tell_the_user"
    if features.get("user_busy") and "prefer_assistance_when_user_busy" in signals:
        return "do_now"
    if activity in {"meal support", "housekeeping", "mobility support"}:
        return "do_now"
    if activity == "cognitive or leisure support":
        return "tell_the_user"
    return "do_now"


def infer_raw_conditions(
    *,
    plan: CooperaPlan,
    task: CooperaTask,
    features: dict[str, Any],
) -> list[str]:
    conditions = list(features.get("conditions", []))
    lowered = f"{plan.intention} {task.thought}".lower()
    for keyword in ["cozy", "quiet", "creative", "playful", "low-effort", "slow"]:
        if keyword in lowered:
            conditions.append(keyword)
    return sorted(set(conditions))


def infer_user_state(features: dict[str, Any]) -> list[str]:
    state = []
    flags = features.get("context_flags", {})
    if features.get("user_busy"):
        state.append("busy")
    if flags.get("user_in_rush"):
        state.append("in_rush")
    if flags.get("user_asleep"):
        state.append("asleep")
    if flags.get("user_injured_or_disabled"):
        state.append("injured_or_disabled")
    if flags.get("user_nearby"):
        state.append("nearby")
    return sorted(set(state))


def infer_environment_flags(features: dict[str, Any]) -> list[str]:
    env = []
    flags = features.get("context_flags", {})
    if flags.get("adverse_weather"):
        env.append("adverse_weather")
    if flags.get("guests_present"):
        env.append("guests_present")
    if flags.get("weekend"):
        env.append("weekend")
    if features.get("quiet_hours"):
        env.append("quiet_hours")
    return sorted(set(env))


def time_bucket(time_text: str) -> str:
    text = time_text.lower().strip()
    match = None
    import re

    match = re.search(r"(\d{1,2})\s*(am|pm)", text)
    if not match:
        return "unknown"
    hour = int(match.group(1))
    meridiem = match.group(2)
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


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
    files_seen: int,
) -> dict[str, Any]:
    labels = Counter(str(sample.get("label_action")) for sample in samples)
    users = Counter(str(sample.get("user_external_id")) for sample in samples)
    activities = Counter(str((sample.get("action_input") or {}).get("activity")) for sample in samples)
    signals = Counter(
        row.get("signal_name")
        for sample in samples
        for row in sample.get("preference_snapshot", [])
    )
    return {
        "num_samples": len(samples),
        "num_files_seen": files_seen,
        "num_errors": len(errors),
        "errors_preview": errors[:20],
        "labels": dict(labels),
        "users": dict(users),
        "activities": dict(activities),
        "top_preference_signals": dict(signals.most_common(20)),
        "output": str(args.output),
        "generation_strategy": args.generation_strategy,
        "response_source": args.response_source,
        "collab_type": args.collab_type,
    }


def _conditions_from_flags(flags: dict[str, bool]) -> list[str]:
    mapping = {
        "adverse_weather": "adverse weather",
        "early_morning": "early morning",
        "guests_present": "guests present",
        "user_asleep": "user asleep",
        "user_in_rush": "user is in a rush",
        "user_injured_or_disabled": "user injured or disabled",
        "user_nearby": "user nearby",
        "weekend": "weekend",
    }
    return [text for flag, text in mapping.items() if flags.get(flag)]


def _normalize_human_ids(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    out = set()
    for value in values:
        text = str(value).strip()
        out.add(text if text.startswith("0") and len(text) == 5 else text.zfill(5))
    return out


def _normalize_days(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(value).strip().zfill(2) for value in values}


def _human_id_to_user_id(human_id: str) -> int:
    try:
        return int(human_id) + 1
    except ValueError:
        return abs(hash(human_id)) % 100000 + 1


def _safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _response_source_from_path(path: Path) -> str:
    parts = set(path.parts)
    if "gpt_response" in parts:
        return "gpt_response"
    if "llama_response" in parts:
        return "llama_response"
    return "unknown"


if __name__ == "__main__":
    main()
