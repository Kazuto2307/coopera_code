"""Step 3 of the split preference-data pipeline: assemble training JSONL.

This script ties the first two steps together. It loads the synthetic humans
written by ``generate_profiles.py`` and the signal-anchored situations written
by ``generate_situations.py``, then produces the final preference-training
JSONL.

It keeps the v2 anti-bias machinery intact:
  * a balanced ``label_schedule`` (``build_label_targets`` + ``build_label_schedule``),
  * target-aware decision reflection (``reflect_targeted_decision``),
  * retry logic (``--max-attempts-per-sample`` / ``--accept-mismatch-if-useful``),
  * a routine-consistency registry (``routine_signature`` / ``register_routine``).

The new piece is situation selection: for each scheduled target label we pick a
situation whose ``anchored_signal`` is present in the selected profile's
``stable_preferences`` (so the signal is actually a lever for that human); if no
such situation exists we fall back to any situation. Among matches we softly
prefer situations whose ``differentiating_labels`` include the target label,
which raises the target hit-rate without violating the matching rule.

Sample ids follow ``synth:<human_id>:<situation_id>:<attempt>``. Nothing loads
Qwen under ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from human_sim_preference_data import (
    ProgressDisplay,
    build_sample_from_stages,
    summarize,
)
from human_sim_preference_data_v2_balanced import (
    LABEL_ORDER,
    build_label_schedule,
    build_label_targets,
    parse_label_weights,
    print_balanced_summary,
    reflect_targeted_decision,
    register_routine,
    routine_signature,
)
from qwen_labeler import QwenDecisionLabeler


def parse_args() -> argparse.Namespace:
    program_dir = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_profiles_dir = program_dir / "generated_data" / "profiles"
    default_situations_dir = program_dir / "generated_data" / "situations"
    default_output = (
        program_dir / "generated_data" / f"preference_training_{stamp}.jsonl"
    )
    default_summary = (
        program_dir / "generated_data" / f"preference_training_{stamp}_summary.json"
    )
    default_intermediate = (
        program_dir / "generated_data" / f"preference_training_{stamp}_intermediate"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Step 3/3: assemble the final balanced preference-training JSONL from "
            "generated profiles and signal-anchored situations."
        )
    )
    parser.add_argument("--profiles-dir", type=Path, default=default_profiles_dir)

    situations = parser.add_mutually_exclusive_group()
    situations.add_argument(
        "--situations",
        type=Path,
        default=None,
        help="Path to a single situations JSONL.",
    )
    situations.add_argument(
        "--situations-dir",
        type=Path,
        default=None,
        help="Directory of situation JSONLs to combine (default if neither given).",
    )

    parser.add_argument(
        "--profile-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of profile_index values to use.",
    )
    parser.add_argument("--target-samples", type=int, default=1000)
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
        help="Retries when Qwen reflects a label different from the scheduled target.",
    )
    parser.add_argument(
        "--accept-mismatch-if-useful",
        action="store_true",
        help=(
            "If all attempts miss the target, accept the last sample when its actual "
            "label still has remaining quota."
        ),
    )
    parser.add_argument(
        "--routine-consistency",
        choices=["enforce", "off"],
        default="enforce",
        help=(
            "When enforced, reject attempts where the same human/context/routine "
            "signature was already accepted with a different label."
        ),
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary-output", type=Path, default=default_summary)
    parser.add_argument("--intermediate-dir", type=Path, default=default_intermediate)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--qwen-temperature", type=float, default=0.35)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1800)
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
        default_dir=Path(__file__).resolve().parent / "generated_data" / "situations",
    )
    situations_by_signal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for situation in situations:
        situations_by_signal[situation.get("anchored_signal")].append(situation)

    label_targets = build_label_targets(
        labels=LABEL_ORDER,
        target_samples=args.target_samples,
        weights=parse_label_weights(args.label_weights),
    )
    label_schedule = build_label_schedule(label_targets)

    plan = {
        "step": "3/3 generate_training_data",
        "profiles_dir": str(profiles_dir),
        "num_profiles": len(profiles),
        "profile_ids": [str(profile.get("human_id")) for profile in profiles],
        "situations_source": situations_source,
        "num_situations": len(situations),
        "anchored_signals": sorted(
            signal for signal in situations_by_signal if signal is not None
        ),
        "target_samples": len(label_schedule),
        "target_label_counts": dict(label_targets),
        "max_attempts_per_sample": args.max_attempts_per_sample,
        "accept_mismatch_if_useful": args.accept_mismatch_if_useful,
        "routine_consistency": args.routine_consistency,
        "planned_qwen_calls_max": len(label_schedule) * args.max_attempts_per_sample,
        "output": str(args.output),
        "summary_output": str(args.summary_output),
        "intermediate_dir": str(args.intermediate_dir),
    }
    print_training_plan(plan)

    if args.dry_run:
        return
    if not profiles:
        raise FileNotFoundError(f"No profile JSON files found in: {profiles_dir}")
    if not situations:
        raise FileNotFoundError(f"No situations loaded from: {situations_source}")

    write_intermediate_manifest(
        intermediate_dir=args.intermediate_dir,
        label_targets=label_targets,
        label_schedule=label_schedule,
        profiles=profiles,
        situations_by_signal=situations_by_signal,
    )

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
    memory_by_profile_day: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    selection_rotation: dict[tuple[str, str], int] = {}
    seen_profiles: set[Any] = set()
    progress = ProgressDisplay(
        enabled=not args.no_progress,
        total_profiles=len(profiles),
        total_samples=len(label_schedule),
    )

    # Stream each accepted sample to disk as soon as it is produced, so the JSONL
    # grows live and an interrupt keeps everything generated so far.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_handle = args.output.open("w", encoding="utf-8")
    progress.start()

    try:
        for sample_idx, target_label in enumerate(label_schedule):
            profile = profiles[sample_idx % len(profiles)]
            human_id = str(profile.get("human_id"))
            profile_index = profile.get("profile_index")
            if profile_index not in seen_profiles:
                seen_profiles.add(profile_index)
                progress.profile_start(human_id=human_id, profile_index=profile_index)
                progress.profile_done()

            sample, attempt_errors, chosen_day = generate_training_sample(
                generator=generator,
                profile=profile,
                situations_by_signal=situations_by_signal,
                all_situations=situations,
                target_label=target_label,
                label_targets=label_targets,
                accepted_counts=accepted_counts,
                memory_by_profile_day=memory_by_profile_day,
                existing_count=len(samples),
                max_attempts=args.max_attempts_per_sample,
                accept_mismatch_if_useful=args.accept_mismatch_if_useful,
                routine_consistency=args.routine_consistency,
                routine_registry=routine_registry,
                consistency_counts=consistency_counts,
                selection_rotation=selection_rotation,
                progress=progress,
            )
            errors.extend(attempt_errors)
            if sample is None:
                continue

            actual_label = sample["label_action"]
            samples.append(sample)
            output_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            output_handle.flush()
            accepted_counts[actual_label] += 1
            if actual_label != target_label:
                mismatch_counts[f"{target_label}->{actual_label}"] += 1
            register_routine(
                sample=sample,
                routine_registry=routine_registry,
                consistency_counts=consistency_counts,
            )
            time_text = str(sample["source_metadata"].get("time_text", ""))
            memory_by_profile_day[(human_id, str(chosen_day))].append(
                memory_entry(sample, time_text)
            )
            progress.sample(label=actual_label)
    finally:
        progress.close()
        generator.close()
        output_handle.close()

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
    summary["anchored_signal_counts"] = dict(
        Counter(
            sample["source_metadata"].get("anchored_signal") for sample in samples
        )
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_balanced_summary(summary)


def generate_training_sample(
    *,
    generator: QwenDecisionLabeler,
    profile: dict[str, Any],
    situations_by_signal: dict[str, list[dict[str, Any]]],
    all_situations: list[dict[str, Any]],
    target_label: str,
    label_targets: Counter[str],
    accepted_counts: Counter[str],
    memory_by_profile_day: dict[tuple[str, str], list[dict[str, Any]]],
    existing_count: int,
    max_attempts: int,
    accept_mismatch_if_useful: bool,
    routine_consistency: str,
    routine_registry: dict[str, dict[str, Any]],
    consistency_counts: Counter[str],
    selection_rotation: dict[tuple[str, str], int],
    progress: ProgressDisplay,
) -> tuple[dict[str, Any] | None, list[dict[str, str]], str | None]:
    human_id = str(profile.get("human_id"))
    profile_index = profile.get("profile_index")
    profile_summary = profile.get("profile_summary") or {}
    preference_profile = profile.get("preference_profile") or {}
    profile_context = {
        "human_id": human_id,
        "profile_index": profile_index,
        "mypersonality": {"big_five": profile.get("big_five")},
        "traits_summary": None,
    }
    stable_signals = [
        item.get("signal_name")
        for item in preference_profile.get("stable_preferences", [])
        if isinstance(item, dict) and item.get("signal_name")
    ]

    attempt_errors: list[dict[str, str]] = []
    last_sample: dict[str, Any] | None = None
    last_day: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            situation = select_situation(
                signal_pool=stable_signals,
                target_label=target_label,
                situations_by_signal=situations_by_signal,
                all_situations=all_situations,
                rotation=selection_rotation,
                rotation_key=(human_id, target_label),
                attempt=attempt,
            )
            if situation is None:
                raise ValueError("No situations available to select.")

            situation_id = str(situation.get("situation_id"))
            anchored_signal = situation.get("anchored_signal")
            day = str(situation.get("day") or "01")
            time_text = str(situation.get("time_text") or "")
            memory = memory_by_profile_day[(human_id, day)]

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
                scenario=situation,
                target_label=target_label,
                memory=memory,
            )
            sample = build_sample_from_stages(
                profile_context=profile_context,
                profile_summary=profile_summary,
                preference_profile=preference_profile,
                scenario=situation,
                scenario_metadata={
                    "situation_id": situation_id,
                    "anchored_signal": anchored_signal,
                    "differentiating_labels": situation.get("differentiating_labels"),
                    "scenario_rationale": situation.get("scenario_rationale"),
                    "day": day,
                    "time_text": time_text,
                },
                decision_payload=decision_payload,
                decision_metadata=decision_meta,
                day=day,
                time_text=time_text,
                scenario_idx=1,
                existing_count=existing_count,
            )

            # Override the id and provenance to reflect the split pipeline.
            sample["sample_id"] = f"synth:{human_id}:{situation_id}:{attempt}"
            meta = sample["source_metadata"]
            meta["source"] = "preference_training_data_from_situations"
            meta["anchored_signal"] = anchored_signal
            meta["human_id"] = human_id
            meta["situation_id"] = situation_id
            meta["target_label"] = target_label
            meta["balanced_attempt"] = attempt
            meta["differentiating_labels"] = situation.get("differentiating_labels")
            for key in ("action_input", "context_input", "structured_task_features"):
                sample["data_provenance"][key] = "synthetic_qwen_situation_anchored_to_signal"

            signature = routine_signature(sample)
            meta["routine_signature"] = signature
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
            last_day = day
            if sample["label_action"] == target_label:
                return sample, attempt_errors, day
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
                    "stage": f"training_attempt_{attempt}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if (
        accept_mismatch_if_useful
        and last_sample is not None
        and accepted_counts[last_sample["label_action"]]
        < label_targets[last_sample["label_action"]]
    ):
        last_sample["source_metadata"]["accepted_despite_label_mismatch"] = True
        return last_sample, attempt_errors, last_day

    return None, attempt_errors, None


def select_situation(
    *,
    signal_pool: list[str],
    target_label: str,
    situations_by_signal: dict[str, list[dict[str, Any]]],
    all_situations: list[dict[str, Any]],
    rotation: dict[tuple[str, str], int],
    rotation_key: tuple[str, str],
    attempt: int,
) -> dict[str, Any] | None:
    """Pick a situation anchored to one of the profile's stable signals.

    Primary rule (per the design brief): the situation's ``anchored_signal`` must
    be in the profile's stable preferences; otherwise fall back to any situation.
    Soft preference: among the chosen pool, favour situations whose
    ``differentiating_labels`` include the target label. Selection rotates per
    (human, target_label) and is offset by attempt for retry variety.
    """
    matching: list[dict[str, Any]] = []
    for signal in signal_pool:
        matching.extend(situations_by_signal.get(signal, []))
    pool = matching if matching else all_situations
    if not pool:
        return None

    targeted = [
        situation
        for situation in pool
        if target_label in (situation.get("differentiating_labels") or [])
    ]
    chosen_pool = targeted if targeted else pool

    base = rotation.get(rotation_key, 0)
    index = (base + attempt - 1) % len(chosen_pool)
    rotation[rotation_key] = base + 1
    return chosen_pool[index]


def memory_entry(sample: dict[str, Any], time_text: str) -> dict[str, Any]:
    return {
        "time": time_text,
        "action_input": sample["action_input"],
        "context_input": sample["context_input"],
        "label_action": sample["label_action"],
        "preference_snapshot": sample["preference_snapshot"],
    }


def load_profiles(
    profiles_dir: Path,
    profile_indices: list[int] | None,
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for path in sorted(profiles_dir.glob("human_*.json")):
        profiles.append(json.loads(path.read_text(encoding="utf-8")))
    if profile_indices is not None:
        wanted = set(profile_indices)
        profiles = [
            profile for profile in profiles if profile.get("profile_index") in wanted
        ]
    return profiles


def load_situations(
    *,
    situations_path: Path | None,
    situations_dir: Path | None,
    default_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    if situations_path is not None:
        paths = [situations_path]
        source = str(situations_path)
    elif situations_dir is not None:
        paths = sorted(situations_dir.glob("*.jsonl"))
        source = str(situations_dir)
    else:
        paths = sorted(default_dir.glob("*.jsonl"))
        source = str(default_dir)

    situations: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            situations.append(json.loads(line))
    return situations, source


def write_intermediate_manifest(
    *,
    intermediate_dir: Path,
    label_targets: Counter[str],
    label_schedule: list[str],
    profiles: list[dict[str, Any]],
    situations_by_signal: dict[str, list[dict[str, Any]]],
) -> None:
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    (intermediate_dir / "label_schedule.json").write_text(
        json.dumps(
            {
                "label_targets": dict(label_targets),
                "schedule_length": len(label_schedule),
                "schedule": label_schedule,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    profiles_index = []
    for profile in profiles:
        stable = [
            item.get("signal_name")
            for item in (profile.get("preference_profile") or {}).get(
                "stable_preferences", []
            )
            if isinstance(item, dict)
        ]
        profiles_index.append(
            {
                "human_id": profile.get("human_id"),
                "profile_index": profile.get("profile_index"),
                "stable_signals": stable,
                "matching_situations": sum(
                    len(situations_by_signal.get(signal, [])) for signal in stable
                ),
            }
        )
    (intermediate_dir / "profiles_index.json").write_text(
        json.dumps(profiles_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_training_plan(plan: dict[str, Any]) -> None:
    print("")
    print("=" * 72)
    print("Step 3/3 - assemble balanced training data")
    print("=" * 72)
    print(f"Profiles            : {plan['num_profiles']} -> {plan['profile_ids']}")
    print(f"Profiles dir        : {plan['profiles_dir']}")
    print(f"Situations source   : {plan['situations_source']}")
    print(f"Situations loaded   : {plan['num_situations']}")
    print(f"Anchored signals    : {len(plan['anchored_signals'])}")
    print(f"Target samples      : {plan['target_samples']}")
    print(f"Max Qwen calls      : {plan['planned_qwen_calls_max']}")
    print(f"Attempts per sample : {plan['max_attempts_per_sample']}")
    print(f"Accept mismatch     : {plan['accept_mismatch_if_useful']}")
    print(f"Routine consistency : {plan['routine_consistency']}")
    print("")
    print("Target label counts:")
    width = max((len(label) for label in plan["target_label_counts"]), default=0)
    for label, count in plan["target_label_counts"].items():
        print(f"  {label.ljust(width)} : {count}")
    print("")
    print(f"Output JSONL        : {plan['output']}")
    print(f"Summary JSON        : {plan['summary_output']}")
    print(f"Intermediates       : {plan['intermediate_dir']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
