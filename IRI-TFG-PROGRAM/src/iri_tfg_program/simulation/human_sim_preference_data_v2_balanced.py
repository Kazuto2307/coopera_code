from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from iri_tfg_program import PROJECT_ROOT
from typing import Any

from iri_tfg_program.profiles.coopera_profile_loader import (
    load_latest_traits_summary,
    load_mypersonality_profiles,
    resolve_mypersonality_path,
)
from iri_tfg_program.simulation.human_sim_preference_data import (
    DEFAULT_TIMES,
    ProgressDisplay,
    build_sample_from_stages,
    compact_snapshot,
    generate_preference_profile,
    get_or_generate_profile_summary,
    normalize_action_input,
    normalize_context_input,
    normalize_structured_features,
    print_counter,
    select_profile_indices,
    summarize,
    validate_snapshot,
    write_intermediate,
    write_jsonl,
)
from iri_tfg_program.taxonomy.preference_taxonomy import VALID_LABELS
from iri_tfg_program.labeling.qwen_labeler import QwenDecisionLabeler


LABEL_ORDER = ["do_now", "do_later", "tell_the_user", "no_action"]
LABEL_GUIDANCE = {
    "do_now": (
        "Create a low-risk, useful, timely situation where immediate robot help is "
        "reasonable for this human."
    ),
    "do_later": (
        "Create a situation where the robot should postpone the action because "
        "timing, context, user state, privacy, interruption cost, or dependency "
        "makes immediate execution inappropriate."
    ),
    "tell_the_user": (
        "Create a situation where the robot should tell, notify, or ask the user "
        "rather than physically executing the task itself."
    ),
    "no_action": (
        "Create a situation where the robot should stay passive because help would "
        "be intrusive, unsupported by the profile, unnecessary, unsafe, private, or "
        "against the user's stable preferences."
    ),
}


def parse_args() -> argparse.Namespace:
    program_dir = PROJECT_ROOT
    coopera_root = program_dir.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = (
        program_dir / "data/generated" / f"preference_human_sim_balanced_{stamp}.jsonl"
    )
    default_summary = (
        program_dir
        / "data/generated"
        / f"preference_human_sim_balanced_{stamp}_summary.json"
    )
    default_intermediate = (
        program_dir
        / "data/generated"
        / f"preference_human_sim_balanced_{stamp}_intermediate"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Balanced V2 of the COOPERA-style preference data simulator. "
            "It keeps the original pipeline but schedules target labels to "
            "reduce do_now dominance."
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
    parser.add_argument("--max-profiles", type=int, default=20)
    parser.add_argument("--max-days", type=int, default=5)
    parser.add_argument("--times", nargs="*", default=None)
    parser.add_argument("--target-samples", type=int, default=1000)
    parser.add_argument(
        "--labels",
        nargs="+",
        choices=LABEL_ORDER,
        default=LABEL_ORDER,
        help="Labels to balance. Default: all four labels.",
    )
    parser.add_argument(
        "--label-weights",
        default=None,
        help=(
            "Optional JSON dict of label weights, e.g. "
            "'{\"do_now\": 0.25, \"do_later\": 0.25, \"tell_the_user\": 0.25, \"no_action\": 0.25}'."
        ),
    )
    parser.add_argument(
        "--max-attempts-per-sample",
        type=int,
        default=3,
        help="Retries when Qwen generates a label different from the scheduled target.",
    )
    parser.add_argument(
        "--accept-mismatch-if-useful",
        action="store_true",
        help=(
            "If all attempts miss the target, accept the last generated sample when "
            "its actual label still has remaining quota."
        ),
    )
    parser.add_argument(
        "--routine-consistency",
        choices=["enforce", "off"],
        default="enforce",
        help=(
            "When enabled, reject attempts where the same human/context/routine "
            "signature was previously accepted with a different label."
        ),
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary-output", type=Path, default=default_summary)
    parser.add_argument("--intermediate-dir", type=Path, default=default_intermediate)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--qwen-temperature", type=float, default=0.35)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1800)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--plan-json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
    label_targets = build_label_targets(
        labels=args.labels,
        target_samples=args.target_samples,
        weights=parse_label_weights(args.label_weights),
    )
    label_schedule = build_label_schedule(label_targets)
    slots = build_generation_slots(
        profile_indices=selected_indices,
        days=days,
        times=times,
    )
    planned_qwen_calls = estimate_balanced_calls(
        profiles=profiles,
        selected_indices=selected_indices,
        results_dir=results_dir,
        response_source=args.response_source,
        target_samples=len(label_schedule),
        max_attempts_per_sample=args.max_attempts_per_sample,
    )
    plan = {
        "pipeline": (
            "profile_summary -> preference_profile -> balanced_target_scenario "
            "-> target_aware_decision_reflection"
        ),
        "coopera_root": str(coopera_root),
        "mypersonality_path": str(mypersonality_path),
        "num_profiles_available": len(profiles),
        "profile_indices": selected_indices,
        "days": days,
        "times": times,
        "target_samples": len(label_schedule),
        "target_label_counts": dict(label_targets),
        "max_attempts_per_sample": args.max_attempts_per_sample,
        "accept_mismatch_if_useful": args.accept_mismatch_if_useful,
        "routine_consistency": args.routine_consistency,
        "planned_qwen_calls_max": planned_qwen_calls,
        "output": str(args.output),
        "summary_output": str(args.summary_output),
        "intermediate_dir": str(args.intermediate_dir),
    }
    print_balanced_plan(plan=plan, show_json=args.plan_json)

    if args.dry_run:
        return
    if not profiles:
        raise FileNotFoundError(f"No COOPERA profiles found at: {mypersonality_path}")
    if not slots:
        raise ValueError("No generation slots available. Check profiles, days, and times.")

    generator = QwenDecisionLabeler(
        model_name=args.qwen_model,
        temperature=args.qwen_temperature,
        max_new_tokens=args.qwen_max_new_tokens,
    )

    samples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    accepted_counts: Counter[str] = Counter()
    mismatch_counts: Counter[str] = Counter()
    consistency_counts: Counter[str] = Counter()
    routine_registry: dict[str, dict[str, Any]] = {}
    profile_cache: dict[int, dict[str, Any]] = {}
    memory_by_profile_day: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    progress = ProgressDisplay(
        enabled=not args.no_progress,
        total_profiles=len(selected_indices),
        total_samples=len(label_schedule),
    )
    progress.start()

    try:
        completed_profiles: set[int] = set()
        for sample_idx, target_label in enumerate(label_schedule):
            slot = slots[sample_idx % len(slots)]
            profile_index = int(slot["profile_index"])
            day = str(slot["day"])
            time_text = str(slot["time_text"])
            human_id = str(profile_index).zfill(5)

            try:
                profile_bundle = get_profile_bundle(
                    profile_index=profile_index,
                    profiles=profiles,
                    profile_cache=profile_cache,
                    generator=generator,
                    results_dir=results_dir,
                    response_source=args.response_source,
                    intermediate_dir=args.intermediate_dir,
                    progress=progress,
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
                continue

            if profile_index not in completed_profiles:
                completed_profiles.add(profile_index)
                progress.profile_done()

            memory_key = (human_id, day)
            memory = memory_by_profile_day[memory_key]
            sample, attempt_errors = generate_balanced_sample(
                generator=generator,
                profile_bundle=profile_bundle,
                day=day,
                time_text=time_text,
                target_label=target_label,
                label_targets=label_targets,
                accepted_counts=accepted_counts,
                memory=memory,
                existing_count=len(samples),
                max_attempts=args.max_attempts_per_sample,
                accept_mismatch_if_useful=args.accept_mismatch_if_useful,
                routine_consistency=args.routine_consistency,
                routine_registry=routine_registry,
                consistency_counts=consistency_counts,
                progress=progress,
            )
            errors.extend(attempt_errors)
            if sample is None:
                continue

            actual_label = sample["label_action"]
            samples.append(sample)
            accepted_counts[actual_label] += 1
            if actual_label != target_label:
                mismatch_counts[f"{target_label}->{actual_label}"] += 1
            register_routine(
                sample=sample,
                routine_registry=routine_registry,
                consistency_counts=consistency_counts,
            )
            memory.append(
                {
                    "time": time_text,
                    "action_input": sample["action_input"],
                    "context_input": sample["context_input"],
                    "label_action": sample["label_action"],
                    "preference_snapshot": sample["preference_snapshot"],
                }
            )
            progress.sample(label=actual_label)
    finally:
        progress.close()
        generator.close()

    write_jsonl(args.output, samples)
    summary = summarize(samples=samples, errors=errors, args=args, plan=plan)
    summary["target_label_counts"] = dict(label_targets)
    summary["accepted_label_counts"] = dict(accepted_counts)
    summary["mismatch_counts"] = dict(mismatch_counts)
    summary["routine_consistency"] = {
        "mode": args.routine_consistency,
        "unique_routines": len(routine_registry),
        **dict(consistency_counts),
    }
    summary["missing_label_counts"] = {
        label: max(label_targets[label] - accepted_counts[label], 0)
        for label in label_targets
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_balanced_summary(summary)


def get_profile_bundle(
    *,
    profile_index: int,
    profiles: list[dict[str, Any]],
    profile_cache: dict[int, dict[str, Any]],
    generator: QwenDecisionLabeler,
    results_dir: Path,
    response_source: str,
    intermediate_dir: Path,
    progress: ProgressDisplay,
) -> dict[str, Any]:
    if profile_index in profile_cache:
        return profile_cache[profile_index]

    profile = profiles[profile_index]
    human_id = str(profile_index).zfill(5)
    progress.profile_start(human_id=human_id, profile_index=profile_index)
    traits_summary = load_latest_traits_summary(
        results_dir=results_dir,
        response_source=response_source,
        human_id=human_id,
    )
    profile_context = {
        "human_id": human_id,
        "profile_index": profile_index,
        "mypersonality": profile,
        "traits_summary": traits_summary,
    }

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
        intermediate_dir,
        human_id,
        "profile_summary",
        profile_summary,
        profile_summary_meta,
    )
    write_intermediate(
        intermediate_dir,
        human_id,
        "preference_profile",
        preference_profile,
        preference_meta,
    )

    bundle = {
        "profile_context": profile_context,
        "profile_summary": profile_summary,
        "preference_profile": preference_profile,
    }
    profile_cache[profile_index] = bundle
    return bundle


def generate_balanced_sample(
    *,
    generator: QwenDecisionLabeler,
    profile_bundle: dict[str, Any],
    day: str,
    time_text: str,
    target_label: str,
    label_targets: Counter[str],
    accepted_counts: Counter[str],
    memory: list[dict[str, Any]],
    existing_count: int,
    max_attempts: int,
    accept_mismatch_if_useful: bool,
    routine_consistency: str,
    routine_registry: dict[str, dict[str, Any]],
    consistency_counts: Counter[str],
    progress: ProgressDisplay,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    profile_context = profile_bundle["profile_context"]
    profile_summary = profile_bundle["profile_summary"]
    preference_profile = profile_bundle["preference_profile"]
    human_id = str(profile_context["human_id"])
    attempt_errors: list[dict[str, str]] = []
    last_sample: dict[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            progress.stage(
                f"scenario_for_{target_label}",
                human_id=human_id,
                day=day,
                time_text=time_text,
                scenario_idx=attempt,
            )
            scenario_payload, scenario_meta = generate_targeted_scenario(
                generator=generator,
                profile_summary=profile_summary,
                preference_profile=preference_profile,
                day=day,
                time_text=time_text,
                target_label=target_label,
                memory=memory,
                attempt=attempt,
            )
            scenario = select_single_scenario(scenario_payload, day=day)
            progress.stage(
                f"decision_for_{target_label}",
                human_id=human_id,
                day=day,
                time_text=time_text,
                scenario_idx=attempt,
            )
            decision_payload, decision_meta = reflect_targeted_decision(
                generator=generator,
                profile_summary=profile_summary,
                preference_profile=preference_profile,
                scenario=scenario,
                target_label=target_label,
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
                scenario_idx=1,
                existing_count=existing_count,
            )
            sample["source_metadata"]["source"] = (
                "preference_human_sim_multistage_qwen_balanced_v2"
            )
            sample["source_metadata"]["target_label"] = target_label
            sample["source_metadata"]["balanced_attempt"] = attempt
            signature = routine_signature(sample)
            sample["source_metadata"]["routine_signature"] = signature
            if routine_consistency == "enforce":
                existing = routine_registry.get(signature)
                if existing and existing["label_action"] != sample["label_action"]:
                    consistency_counts["routine_label_conflicts_rejected"] += 1
                    attempt_errors.append(
                        {
                            "human_id": human_id,
                            "day": day,
                            "time": time_text,
                            "stage": "routine_label_conflict",
                            "error": (
                                "same routine signature already exists with "
                                f"label={existing['label_action']}; "
                                f"new_label={sample['label_action']}; "
                                f"target={target_label}; attempt={attempt}"
                            ),
                        }
                    )
                    continue
                if existing:
                    consistency_counts["same_label_duplicate_routines_seen"] += 1
            last_sample = sample
            if sample["label_action"] == target_label:
                return sample, attempt_errors
            attempt_errors.append(
                {
                    "human_id": human_id,
                    "day": day,
                    "time": time_text,
                    "stage": "label_mismatch",
                    "error": (
                        f"target={target_label}, actual={sample['label_action']}, "
                        f"attempt={attempt}"
                    ),
                }
            )
        except Exception as exc:
            attempt_errors.append(
                {
                    "human_id": human_id,
                    "day": day,
                    "time": time_text,
                    "stage": f"balanced_attempt_{attempt}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if (
        accept_mismatch_if_useful
        and last_sample is not None
        and accepted_counts[last_sample["label_action"]] < label_targets[last_sample["label_action"]]
    ):
        last_sample["source_metadata"]["accepted_despite_label_mismatch"] = True
        return last_sample, attempt_errors

    return None, attempt_errors


def register_routine(
    *,
    sample: dict[str, Any],
    routine_registry: dict[str, dict[str, Any]],
    consistency_counts: Counter[str],
) -> None:
    signature = sample.get("source_metadata", {}).get("routine_signature")
    if not signature:
        signature = routine_signature(sample)
        sample.setdefault("source_metadata", {})["routine_signature"] = signature
    if signature not in routine_registry:
        consistency_counts["unique_routines_registered"] += 1
        routine_registry[signature] = {
            "sample_id": sample.get("sample_id"),
            "label_action": sample.get("label_action"),
            "action_input": sample.get("action_input"),
            "context_input": sample.get("context_input"),
            "structured_task_features": sample.get("structured_task_features"),
            "user_external_id": sample.get("user_external_id"),
        }


def routine_signature(sample: dict[str, Any]) -> str:
    action = sample.get("action_input") or {}
    context = sample.get("context_input") or {}
    features = sample.get("structured_task_features") or {}
    key = {
        "user_external_id": sample.get("user_external_id"),
        "action_text": normalize_key_text(action.get("action_text")),
        "activity": normalize_key_text(action.get("activity")),
        "location_current": normalize_key_text(context.get("location_current")),
        "time_of_day": normalize_key_text(context.get("time_of_day")),
        "raw_conditions": normalize_key_list(context.get("raw_conditions")),
        "user_state": normalize_key_list(context.get("user_state")),
        "environment_flags": normalize_key_list(context.get("environment_flags")),
        "kind": normalize_key_text(features.get("kind")),
        "urgency": normalize_key_text(features.get("urgency")),
        "sensitivity": normalize_key_text(features.get("sensitivity")),
        "user_busy": bool(features.get("user_busy", False)),
        "quiet_hours": bool(features.get("quiet_hours", False)),
        "conditions": normalize_key_list(features.get("conditions")),
        "context_flags": normalize_key_dict(features.get("context_flags")),
    }
    return json.dumps(key, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def normalize_key_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().lower().split())
    return text or None


def normalize_key_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(
        item
        for item in (normalize_key_text(entry) for entry in value)
        if item
    )


def normalize_key_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key, item in sorted(value.items(), key=lambda row: str(row[0])):
        norm_key = normalize_key_text(key)
        if not norm_key:
            continue
        if isinstance(item, list):
            normalized[norm_key] = normalize_key_list(item)
        elif isinstance(item, dict):
            normalized[norm_key] = normalize_key_dict(item)
        elif isinstance(item, bool):
            normalized[norm_key] = item
        elif item is None:
            normalized[norm_key] = None
        else:
            normalized[norm_key] = normalize_key_text(item)
    return normalized


def generate_targeted_scenario(
    *,
    generator: QwenDecisionLabeler,
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
    day: str,
    time_text: str,
    target_label: str,
    memory: list[dict[str, Any]],
    attempt: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return generator.generate_json(
        system="You generate balanced assistive-robot situations. Return only valid JSON.",
        user=build_targeted_scenario_prompt(
            profile_summary=profile_summary,
            preference_profile=preference_profile,
            day=day,
            time_text=time_text,
            target_label=target_label,
            memory=memory,
            attempt=attempt,
        ),
    )


def reflect_targeted_decision(
    *,
    generator: QwenDecisionLabeler,
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
    scenario: dict[str, Any],
    target_label: str,
    memory: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, meta = generator.generate_json(
        system=(
            "You reflect on robot decisions for balanced synthetic data. "
            "Return only valid JSON."
        ),
        user=build_targeted_decision_prompt(
            profile_summary=profile_summary,
            preference_profile=preference_profile,
            scenario=scenario,
            target_label=target_label,
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
    return payload, {"stage": "target_aware_decision_reflection", **meta}


def build_targeted_scenario_prompt(
    *,
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
    day: str,
    time_text: str,
    target_label: str,
    memory: list[dict[str, Any]],
    attempt: int,
) -> str:
    return (
        "Generate one assistive-robot decision situation for this simulated human.\n\n"
        "This dataset is being balanced by label. Your scenario should naturally "
        f"lead to label_action='{target_label}' after reflection, without forcing "
        "an incoherent decision.\n\n"
        f"Target-label guidance: {LABEL_GUIDANCE[target_label]}\n\n"
        "Rules:\n"
        "- Generate the situation only; do not include label_action or preference_snapshot.\n"
        "- The situation must stay plausible for the human profile and the given time.\n"
        "- Use context flags only when needed to justify the target label.\n"
        "- For do_later/no_action/tell_the_user, create real reasons in the context "
        "rather than simply saying the robot should not act.\n"
        "- action_text MUST be a concrete domestic task that both a human and a robot "
        "could physically perform (e.g. 'make breakfast', 'give medication to user', "
        "'vacuum the living room'). It must NOT describe the robot's response or "
        "framing (NOT 'offer to...', 'remind the user to...', 'suggest...').\n"
        "- Do not invent medical emergencies unless the target requires a high-risk "
        "case and the profile/context supports it.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "scenarios": [\n'
        "    {\n"
        '      "action_input": {"action_text": "...", "activity": "..."},\n'
        '      "context_input": {"location_current": "...", "objects_nearby": [], "available_objects": [], "raw_conditions": [], "time_of_day": "...", "weekday": "synthetic_day_XX", "user_state": [], "environment_flags": []},\n'
        '      "structured_task_features": {"kind": "...", "urgency": "low|medium|high", "sensitivity": "low|medium|high", "user_busy": false, "quiet_hours": false, "conditions": [], "context_flags": {}},\n'
        '      "scenario_rationale": "why this scenario is plausible and why it should tend toward the target label"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Day: {day}\n"
        f"Time: {time_text}\n"
        f"Attempt: {attempt}\n\n"
        f"PROFILE SUMMARY:\n{json.dumps(profile_summary, ensure_ascii=False, indent=2)}\n\n"
        f"STABLE PREFERENCES:\n{json.dumps(preference_profile, ensure_ascii=False, indent=2)}\n\n"
        f"RECENT MEMORY:\n{json.dumps(memory[-8:], ensure_ascii=False, indent=2)}"
    )


def build_targeted_decision_prompt(
    *,
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
    scenario: dict[str, Any],
    target_label: str,
    memory: list[dict[str, Any]],
) -> str:
    return (
        "Reflect on the robot decision for this scenario.\n\n"
        f"The data generator intended this example to represent label_action='{target_label}'. "
        "Use that as a consistency target, but do not output it blindly. If the scenario "
        "does not justify the target label, choose the correct label and explain why.\n\n"
        "Allowed labels:\n"
        "- do_now: robot should execute/help now.\n"
        "- do_later: robot should postpone the action.\n"
        "- tell_the_user: robot should tell or ask the user, not execute directly.\n"
        "- no_action: robot should stay passive.\n\n"
        "Rules:\n"
        "- preference_snapshot must be a subset of the stable preferences unless "
        "there is direct profile evidence in the summary.\n"
        "- Include at most 5 preference_snapshot items.\n"
        "- Do not create new context conditions.\n"
        "- Each preference_snapshot item keeps the polarity and weight (1-10) it had "
        "in the stable preferences.\n"
        "- Copy signal_name values exactly.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "preference_snapshot": [{"signal_name": "...", "polarity": "prefer|avoid", "weight": 7}],\n'
        '  "label_action": "do_now|do_later|tell_the_user|no_action",\n'
        '  "decision_rationale": "...",\n'
        '  "target_label_consistency": "matched|mismatched",\n'
        '  "consistency_checks": ["..."]\n'
        "}\n\n"
        f"PROFILE SUMMARY:\n{json.dumps(profile_summary, ensure_ascii=False, indent=2)}\n\n"
        f"STABLE PREFERENCES:\n{json.dumps(preference_profile, ensure_ascii=False, indent=2)}\n\n"
        f"SCENARIO:\n{json.dumps(scenario, ensure_ascii=False, indent=2)}\n\n"
        f"RECENT MEMORY:\n{json.dumps(memory[-8:], ensure_ascii=False, indent=2)}"
    )


def select_single_scenario(payload: dict[str, Any], *, day: str) -> dict[str, Any]:
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario stage must return a non-empty 'scenarios' list.")
    scenario = scenarios[0]
    if not isinstance(scenario, dict):
        raise ValueError("Scenario must be a JSON object.")
    scenario["action_input"] = normalize_action_input(scenario.get("action_input"))
    scenario["context_input"] = normalize_context_input(
        scenario.get("context_input"),
        day=day,
    )
    scenario["structured_task_features"] = normalize_structured_features(
        scenario.get("structured_task_features")
    )
    return scenario


def build_generation_slots(
    *,
    profile_indices: list[int],
    days: list[str],
    times: list[str],
) -> list[dict[str, Any]]:
    return [
        {"profile_index": profile_index, "day": day, "time_text": time_text}
        for profile_index in profile_indices
        for day in days
        for time_text in times
    ]


def parse_label_weights(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--label-weights must be a JSON object.")
    weights: dict[str, float] = {}
    for label, weight in payload.items():
        if label not in VALID_LABELS:
            raise ValueError(f"Unknown label in --label-weights: {label!r}")
        weights[label] = float(weight)
    return weights


def build_label_targets(
    *,
    labels: list[str],
    target_samples: int,
    weights: dict[str, float] | None,
) -> Counter[str]:
    if target_samples <= 0:
        raise ValueError("--target-samples must be positive.")
    labels = [label for label in LABEL_ORDER if label in labels]
    if not labels:
        raise ValueError("At least one label must be selected.")

    if weights:
        total_weight = sum(max(weights.get(label, 0.0), 0.0) for label in labels)
        if total_weight <= 0:
            raise ValueError("--label-weights must contain positive weights.")
        raw_targets = {
            label: target_samples * max(weights.get(label, 0.0), 0.0) / total_weight
            for label in labels
        }
    else:
        raw_targets = {label: target_samples / len(labels) for label in labels}

    targets = Counter({label: int(raw_targets[label]) for label in labels})
    remainder = target_samples - sum(targets.values())
    ordered_by_fraction = sorted(
        labels,
        key=lambda label: raw_targets[label] - int(raw_targets[label]),
        reverse=True,
    )
    for label in ordered_by_fraction[:remainder]:
        targets[label] += 1
    return targets


def build_label_schedule(targets: Counter[str]) -> list[str]:
    remaining = Counter(targets)
    schedule: list[str] = []
    while sum(remaining.values()) > 0:
        for label in LABEL_ORDER:
            if remaining[label] > 0:
                schedule.append(label)
                remaining[label] -= 1
    return schedule


def estimate_balanced_calls(
    *,
    profiles: list[dict[str, Any]],
    selected_indices: list[int],
    results_dir: Path,
    response_source: str,
    target_samples: int,
    max_attempts_per_sample: int,
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
            calls += 1
        calls += 1
    calls += target_samples * max_attempts_per_sample * 2
    return calls


def print_balanced_plan(*, plan: dict[str, Any], show_json: bool) -> None:
    print("")
    print("=" * 72)
    print("COOPERA balanced preference synthetic data run - V2")
    print("=" * 72)
    print(f"Pipeline              : {plan['pipeline']}")
    print(f"Profiles selected     : {len(plan['profile_indices'])} / {plan['num_profiles_available']}")
    print(f"Days                  : {len(plan['days'])} -> {', '.join(plan['days'])}")
    print(f"Hours per day         : {len(plan['times'])} -> {', '.join(plan['times'])}")
    print(f"Target samples        : {plan['target_samples']}")
    print(f"Max Qwen calls        : {plan['planned_qwen_calls_max']}")
    print(f"Attempts per sample   : {plan['max_attempts_per_sample']}")
    print(f"Routine consistency   : {plan['routine_consistency']}")
    print("")
    print("Target label counts:")
    print_counter(plan["target_label_counts"])
    print("")
    print(f"Profiles CSV          : {plan['mypersonality_path']}")
    print(f"Output JSONL          : {plan['output']}")
    print(f"Summary JSON          : {plan['summary_output']}")
    print(f"Intermediates         : {plan['intermediate_dir']}")
    print("=" * 72)
    if show_json:
        print("")
        print("Full plan JSON:")
        print(json.dumps(plan, ensure_ascii=False, indent=2))


def print_balanced_summary(summary: dict[str, Any]) -> None:
    print("")
    print("=" * 72)
    print("Balanced generation summary - V2")
    print("=" * 72)
    print(f"Samples generated : {summary['num_samples']}")
    print(f"Errors            : {summary['num_errors']}")
    print(f"Output JSONL      : {summary['output']}")
    print(f"Summary JSON      : {summary['summary_output']}")
    print("")
    print("Target label counts:")
    print_counter(summary.get("target_label_counts", {}))
    print("")
    print("Accepted label counts:")
    print_counter(summary.get("accepted_label_counts", {}))
    print("")
    print("Missing label counts:")
    print_counter(summary.get("missing_label_counts", {}))
    if summary.get("mismatch_counts"):
        print("")
        print("Accepted mismatches:")
        print_counter(summary["mismatch_counts"])
    if summary.get("routine_consistency"):
        print("")
        print("Routine consistency:")
        print_counter(summary["routine_consistency"])
    print("")
    print("Top preference signals:")
    print_counter(summary.get("top_preference_signals", {}), limit=12)
    if summary.get("errors_preview"):
        print("")
        print("Errors preview:")
        for item in summary["errors_preview"][:5]:
            print(f"- {item}")
    print("=" * 72)


if __name__ == "__main__":
    main()
