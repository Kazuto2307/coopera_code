"""Final step: assemble preference-training data by FREE decision.

For each (synthetic human, situation) pair, Qwen decides naturally what the
person would want the home assistive robot to do - with NO forcing:

  * no label schedule / target label,
  * no retry-until-a-target-is-hit,
  * no preference anchoring on the situations.

Whatever label distribution emerges, emerges. The situation is taken as-is
(from build_situations_from_external.py -> translate_situations.py), and the
person's profile is the only thing that drives the choice.

New taxonomy: labels are {do_now, do_later, tell_the_user, no_action} and the
preference signals are the 13 hierarchical signals in preference_taxonomy.py
(their prefer/avoid semantics are injected into the prompt via
``describe_signal_semantics``).

Sample ids: ``synth:<human_id>:<situation_id>:<n>``.
Nothing loads Qwen under ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from human_sim_preference_data import (
    ProgressDisplay,
    build_sample_from_stages,
    print_counter,
    summarize,
)
from human_sim_preference_data_v2_balanced import register_routine, routine_signature
from preference_taxonomy import (
    PREFERENCE_SIGNALS,
    VALID_LABELS,
    describe_signal_semantics,
)
from qwen_labeler import QwenDecisionLabeler

SITU_DIR_DEFAULT = Path(__file__).resolve().parent / "generated_data" / "situations_external"
DEFAULT_SITUATIONS = SITU_DIR_DEFAULT / "external_situations_robot.jsonl"

LABEL_MEANINGS = (
    "- do_now: the robot performs the candidate action now.\n"
    "- do_later: the robot postpones the action to a more suitable moment.\n"
    "- tell_the_user: the robot tells or asks the user instead of doing it itself.\n"
    "- no_action: the robot stays passive and does nothing.\n"
)


def parse_args() -> argparse.Namespace:
    program_dir = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_profiles_dir = program_dir / "generated_data" / "profiles"
    default_output = program_dir / "generated_data" / f"preference_training_{stamp}.jsonl"
    default_summary = program_dir / "generated_data" / f"preference_training_{stamp}_summary.json"

    parser = argparse.ArgumentParser(
        description=(
            "Assemble preference-training JSONL by free (unforced) decision over "
            "dataset-grounded situations and synthetic-human profiles."
        )
    )
    parser.add_argument("--profiles-dir", type=Path, default=default_profiles_dir)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--situations", type=Path, default=None,
                     help="Path to a situations JSONL (default: translated robot situations).")
    src.add_argument("--situations-dir", type=Path, default=None,
                     help="Directory of situation JSONLs to combine.")
    parser.add_argument("--profile-indices", type=int, nargs="+", default=None,
                        help="Optional subset of profile_index values to use.")
    parser.add_argument("--target-samples", type=int, default=3000)
    parser.add_argument(
        "--routine-consistency",
        choices=["enforce", "off"],
        default="off",
        help=(
            "off (default): never reject. enforce: skip a sample if the same "
            "human+situation routine already got a different label (de-duplicates "
            "contradictions; it does NOT force any label)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for pairing.")
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
        "step": "generate_training_data (free decision)",
        "profiles_dir": str(profiles_dir),
        "num_profiles": len(profiles),
        "profile_ids": [str(p.get("human_id")) for p in profiles],
        "situations_source": situations_source,
        "num_situations": len(situations),
        "target_samples": args.target_samples,
        "routine_consistency": args.routine_consistency,
        "labels": sorted(VALID_LABELS),
        "num_signals": len(PREFERENCE_SIGNALS),
        "planned_qwen_calls": args.target_samples,
        "output": str(args.output),
        "summary_output": str(args.summary_output),
    }
    print_plan(plan)

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
    routine_registry: dict[str, dict[str, Any]] = {}
    consistency_counts: Counter[str] = Counter()
    memory_by_human: dict[str, list[dict[str, Any]]] = {}
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
            situation = rng.choice(situations)
            human_id = str(profile.get("human_id"))
            if profile.get("profile_index") not in seen_profiles:
                seen_profiles.add(profile.get("profile_index"))
                progress.profile_start(human_id=human_id, profile_index=profile.get("profile_index"))
                progress.profile_done()

            memory = memory_by_human.setdefault(human_id, [])
            progress.stage("free_decision", human_id=human_id)
            try:
                sample = build_one_sample(
                    generator=generator,
                    profile=profile,
                    situation=situation,
                    memory=memory,
                    existing_count=len(samples),
                    index=index,
                )
            except Exception as exc:
                errors.append({"human_id": human_id, "stage": "free_decision",
                               "error": f"{type(exc).__name__}: {exc}"})
                progress.error(human_id=human_id, stage="free_decision",
                               message=f"{type(exc).__name__}: {exc}")
                continue

            if args.routine_consistency == "enforce":
                signature = sample["source_metadata"]["routine_signature"]
                existing = routine_registry.get(signature)
                if existing and existing["label_action"] != sample["label_action"]:
                    consistency_counts["routine_label_conflicts_rejected"] += 1
                    continue

            samples.append(sample)
            output_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            output_handle.flush()
            register_routine(sample=sample, routine_registry=routine_registry,
                             consistency_counts=consistency_counts)
            memory.append(memory_entry(sample))
            progress.sample(label=sample["label_action"])
    finally:
        progress.close()
        generator.close()
        output_handle.close()

    summary = summarize(samples=samples, errors=errors, args=args, plan=plan)
    summary["label_distribution"] = dict(
        Counter(s["label_action"] for s in samples)
    )
    summary["source_dataset_distribution"] = dict(
        Counter(s["source_metadata"].get("source_dataset") for s in samples)
    )
    summary["routine_consistency"] = {
        "mode": args.routine_consistency,
        "unique_routines": len(routine_registry),
        **dict(consistency_counts),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary(summary)


def build_one_sample(
    *,
    generator: QwenDecisionLabeler,
    profile: dict[str, Any],
    situation: dict[str, Any],
    memory: list[dict[str, Any]],
    existing_count: int,
    index: int,
) -> dict[str, Any]:
    human_id = str(profile.get("human_id"))
    profile_summary = profile.get("profile_summary") or {}
    preference_profile = profile.get("preference_profile") or {}
    profile_context = {
        "human_id": human_id,
        "profile_index": profile.get("profile_index"),
        "mypersonality": {"big_five": profile.get("big_five")},
        "traits_summary": None,
    }
    situation_id = str(situation.get("situation_id"))

    decision_payload, decision_meta = reflect_free_decision(
        generator=generator,
        profile_summary=profile_summary,
        preference_profile=preference_profile,
        situation=situation,
        memory=memory,
    )
    sample = build_sample_from_stages(
        profile_context=profile_context,
        profile_summary=profile_summary,
        preference_profile=preference_profile,
        scenario=situation,
        scenario_metadata={
            "situation_id": situation_id,
            "source_dataset": situation.get("source_dataset"),
            "scenario_rationale": situation.get("scenario_rationale"),
        },
        decision_payload=decision_payload,
        decision_metadata=decision_meta,
        day="na",
        time_text="",
        scenario_idx=1,
        existing_count=existing_count,
    )
    sample["sample_id"] = f"synth:{human_id}:{situation_id}:{index}"
    meta = sample["source_metadata"]
    meta["source"] = "preference_training_free_decision"
    meta["human_id"] = human_id
    meta["situation_id"] = situation_id
    meta["source_dataset"] = situation.get("source_dataset")
    meta["original_action_text"] = (situation.get("source_metadata") or {}).get("original_action_text")
    meta["decision_rationale"] = decision_payload.get("decision_rationale")
    for key in ("action_input", "context_input", "structured_task_features"):
        sample["data_provenance"][key] = "external_dataset_situation_robot_action"
    meta["routine_signature"] = routine_signature(sample)
    return sample


def reflect_free_decision(
    *,
    generator: QwenDecisionLabeler,
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
    situation: dict[str, Any],
    memory: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, meta = generator.generate_json(
        system=(
            "You simulate one specific person deciding what a home assistive robot "
            "should do for them. Return only valid JSON."
        ),
        user=build_free_decision_prompt(
            profile_summary=profile_summary,
            preference_profile=preference_profile,
            situation=situation,
            memory=memory,
        ),
    )
    label = str(payload.get("label_action", "")).strip()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label_action={label!r}")
    return payload, {"stage": "free_decision", **meta}


def signals_block() -> str:
    return "\n".join(
        f"- {signal}: {describe_signal_semantics(signal)}"
        for signal in sorted(PREFERENCE_SIGNALS)
    )


def build_free_decision_prompt(
    *,
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
    situation: dict[str, Any],
    memory: list[dict[str, Any]],
) -> str:
    situation_view = {
        "candidate_robot_action": situation.get("action_input"),
        "context": situation.get("context_input"),
        "task_features": situation.get("structured_task_features"),
    }
    return (
        "Decide what THIS specific person would want the home assistive robot to do "
        "in the situation below.\n\n"
        "Choose exactly ONE label_action:\n"
        f"{LABEL_MEANINGS}\n"
        "Decide naturally, grounded in who this person is. Do NOT force any "
        "particular outcome and do NOT assume the robot must help: if this person "
        "would rather not be helped here, 'no_action' is the right answer.\n\n"
        "Then list 0-5 preference signals that drove the decision, each with "
        "polarity 'prefer' or 'avoid'. Use ONLY these signal names (copy exactly):\n"
        f"{signals_block()}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "label_action": "do_now|do_later|tell_the_user|no_action",\n'
        '  "preference_snapshot": [{"signal_name": "...", "polarity": "prefer|avoid"}],\n'
        '  "decision_rationale": "..."\n'
        "}\n\n"
        f"PERSON - profile summary:\n{json.dumps(profile_summary, ensure_ascii=False, indent=2)}\n\n"
        f"PERSON - stable preferences:\n{json.dumps(preference_profile, ensure_ascii=False, indent=2)}\n\n"
        f"SITUATION:\n{json.dumps(situation_view, ensure_ascii=False, indent=2)}\n\n"
        f"RECENT DECISIONS BY THIS PERSON:\n{json.dumps(memory[-6:], ensure_ascii=False, indent=2)}"
    )


def memory_entry(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_input": sample["action_input"],
        "label_action": sample["label_action"],
        "preference_snapshot": sample["preference_snapshot"],
    }


def load_profiles(profiles_dir: Path, profile_indices: list[int] | None) -> list[dict[str, Any]]:
    profiles = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(profiles_dir.glob("human_*.json"))
    ]
    if profile_indices is not None:
        wanted = set(profile_indices)
        profiles = [p for p in profiles if p.get("profile_index") in wanted]
    return profiles


def load_situations(
    *,
    situations_path: Path | None,
    situations_dir: Path | None,
    default_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    if situations_path is not None:
        paths = [situations_path]
        source = str(situations_path)
    elif situations_dir is not None:
        paths = sorted(situations_dir.glob("*.jsonl"))
        source = str(situations_dir)
    else:
        paths = [default_path]
        source = str(default_path)

    situations: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                situations.append(json.loads(line))
    return situations, source


def print_plan(plan: dict[str, Any]) -> None:
    print("")
    print("=" * 72)
    print("Generate training data - free (unforced) decision")
    print("=" * 72)
    print(f"Profiles            : {plan['num_profiles']} -> {plan['profile_ids']}")
    print(f"Situations source   : {plan['situations_source']}")
    print(f"Situations loaded   : {plan['num_situations']}")
    print(f"Target samples      : {plan['target_samples']}")
    print(f"Planned Qwen calls  : {plan['planned_qwen_calls']}")
    print(f"Routine consistency : {plan['routine_consistency']}")
    print(f"Labels              : {', '.join(plan['labels'])}")
    print(f"Signals             : {plan['num_signals']}")
    print(f"Output JSONL        : {plan['output']}")
    print(f"Summary JSON        : {plan['summary_output']}")
    print("=" * 72)


def print_summary(summary: dict[str, Any]) -> None:
    print("")
    print("=" * 72)
    print("Training data summary - free decision")
    print("=" * 72)
    print(f"Samples generated : {summary['num_samples']}")
    print(f"Errors            : {summary['num_errors']}")
    print(f"Output JSONL      : {summary['output']}")
    print("")
    print("Label distribution (natural, unforced):")
    print_counter(summary.get("label_distribution", {}))
    print("")
    print("By source dataset:")
    print_counter(summary.get("source_dataset_distribution", {}))
    print("")
    print("Top preference signals:")
    print_counter(summary.get("top_preference_signals", {}), limit=15)
    if summary.get("routine_consistency"):
        print("")
        print("Routine consistency:")
        print_counter(summary["routine_consistency"])
    if summary.get("errors_preview"):
        print("")
        print("Errors preview:")
        for item in summary["errors_preview"][:5]:
            print(f"- {item}")
    print("=" * 72)


if __name__ == "__main__":
    main()
