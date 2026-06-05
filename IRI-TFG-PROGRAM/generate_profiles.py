"""Step 1 of the split preference-data pipeline: generate synthetic humans.

This script reads the COOPERA ``mypersonality_final.csv`` (plus an optional
``traits_summary`` when COOPERA already produced one) and, for each selected
profile, generates a stable ``profile_summary`` and ``preference_profile`` with
local Qwen. It then asks Qwen for one extra human-readable ``description`` and
persists everything as a single JSON file per human under ``--output-dir``.

The heavy lifting reuses the v1 pipeline functions
(``get_or_generate_profile_summary`` / ``generate_preference_profile`` and their
prompt builders) so the two scripts stay in sync. Nothing here loads Qwen when
``--dry-run`` is set.

Downstream, ``generate_training_data.py`` consumes these profile JSON files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from coopera_profile_loader import (
    load_latest_traits_summary,
    load_mypersonality_profiles,
    resolve_mypersonality_path,
)

# get_or_generate_profile_summary / generate_preference_profile call
# build_profile_summary_prompt / build_preference_profile_prompt internally, so
# importing the two high-level stages transitively reuses the v1 prompt builders.
from human_sim_preference_data import (
    ProgressDisplay,
    generate_preference_profile,
    get_or_generate_profile_summary,
    select_profile_indices,
)
from qwen_labeler import QwenDecisionLabeler


DEFAULT_NUM_PROFILES = 10


def parse_args() -> argparse.Namespace:
    program_dir = Path(__file__).resolve().parent
    coopera_root = program_dir.parent
    default_output_dir = program_dir / "generated_data" / "profiles"

    parser = argparse.ArgumentParser(
        description=(
            "Step 1/3: generate synthetic human profiles (profile_summary + "
            "preference_profile + description) from COOPERA mypersonality data "
            "and persist one JSON per human."
        )
    )
    parser.add_argument("--coopera-root", type=Path, default=coopera_root)
    parser.add_argument("--mypersonality-path", type=Path, default=None)
    parser.add_argument(
        "--response-source",
        choices=["gpt_response", "llama_response"],
        default="gpt_response",
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--num-profiles",
        type=int,
        default=None,
        help=f"Generate the first N profiles (default: {DEFAULT_NUM_PROFILES}).",
    )
    selection.add_argument(
        "--profile-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit profile indices, e.g. --profile-indices 0 3 7.",
    )

    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate profiles even if their JSON file already exists.",
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--qwen-temperature", type=float, default=0.25)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1800)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generation plan without loading Qwen or writing files.",
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
    max_profiles = (
        args.num_profiles if args.num_profiles is not None else DEFAULT_NUM_PROFILES
    )
    selected_indices = select_profile_indices(
        total=len(profiles),
        explicit=args.profile_indices,
        max_profiles=max_profiles,
    )
    output_dir = args.output_dir.resolve()

    plan = {
        "step": "1/3 generate_profiles",
        "coopera_root": str(coopera_root),
        "mypersonality_path": str(mypersonality_path),
        "num_profiles_available": len(profiles),
        "profile_indices": selected_indices,
        "output_dir": str(output_dir),
        "overwrite": args.overwrite,
        "response_source": args.response_source,
    }
    print_profiles_plan(plan)

    if args.dry_run:
        return
    if not profiles:
        raise FileNotFoundError(f"No COOPERA profiles found at: {mypersonality_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generator = QwenDecisionLabeler(
        model_name=args.qwen_model,
        temperature=args.qwen_temperature,
        max_new_tokens=args.qwen_max_new_tokens,
    )

    saved: list[str] = []
    skipped: list[str] = []
    errors: list[dict[str, str]] = []
    progress = ProgressDisplay(
        enabled=not args.no_progress,
        total_profiles=len(selected_indices),
        total_samples=len(selected_indices),
    )
    progress.start()

    try:
        for profile_index in selected_indices:
            profile = profiles[profile_index]
            human_id = str(profile_index).zfill(5)
            target_path = output_dir / f"human_{human_id}.json"
            progress.profile_start(human_id=human_id, profile_index=profile_index)

            if target_path.exists() and not args.overwrite:
                progress.stage("skip_existing", human_id=human_id)
                skipped.append(human_id)
                progress.sample(label="skipped")
                progress.profile_done()
                continue

            try:
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

                progress.stage("profile_summary", human_id=human_id)
                profile_summary, _ = get_or_generate_profile_summary(
                    generator=generator,
                    profile_context=profile_context,
                )
                progress.stage("preference_profile", human_id=human_id)
                preference_profile, _ = generate_preference_profile(
                    generator=generator,
                    profile_summary=profile_summary,
                    profile_context=profile_context,
                )
                progress.stage("description", human_id=human_id)
                description = generate_profile_description(
                    generator=generator,
                    profile_summary=profile_summary,
                    preference_profile=preference_profile,
                )

                record = {
                    "human_id": human_id,
                    "profile_index": profile_index,
                    "big_five": profile.get("big_five"),
                    "profile_summary": profile_summary,
                    "preference_profile": preference_profile,
                    "description": description,
                    "generated_at": datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                }
                target_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                saved.append(human_id)
                progress.sample(label="saved")
            except Exception as exc:
                errors.append(
                    {
                        "human_id": human_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                progress.error(
                    human_id=human_id,
                    stage="generate_profile",
                    message=f"{type(exc).__name__}: {exc}",
                )
            progress.profile_done()
    finally:
        progress.close()
        generator.close()

    print_profiles_summary(
        output_dir=output_dir,
        saved=saved,
        skipped=skipped,
        errors=errors,
    )


def generate_profile_description(
    *,
    generator: QwenDecisionLabeler,
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
) -> str | None:
    """Ask Qwen for one readable sentence describing the synthetic human.

    This is best-effort: any failure returns None so the profile is still
    persisted without a description rather than aborting the run.
    """
    try:
        payload, _ = generator.generate_json(
            system=(
                "You write one-sentence descriptions of synthetic humans for an "
                "assistive-robot study. Return only valid JSON."
            ),
            user=build_profile_description_prompt(
                profile_summary=profile_summary,
                preference_profile=preference_profile,
            ),
        )
        description = payload.get("description")
        if isinstance(description, str):
            description = description.strip()
        return description or None
    except Exception:
        return None


def build_profile_description_prompt(
    *,
    profile_summary: dict[str, Any],
    preference_profile: dict[str, Any],
) -> str:
    return (
        "Write exactly ONE short, human-readable sentence (max ~30 words) that "
        "describes this synthetic human for an assistive-robot study.\n\n"
        "When supported by the evidence, mention how much robot initiative and "
        "autonomous execution they tolerate, how much control they want to keep, "
        "their sensitivity to interruption/context, and their attitude to safety "
        "and risk.\n\n"
        "Rules:\n"
        "- Base the sentence ONLY on the profile summary and stable preferences below.\n"
        "- Do not invent demographic facts (age, gender, diagnosis) that are not present.\n"
        "- One plain-language sentence, no lists, no preamble.\n\n"
        'Return JSON: {"description": "..."}\n\n'
        f"PROFILE SUMMARY:\n{json.dumps(profile_summary, ensure_ascii=False, indent=2)}\n\n"
        f"STABLE PREFERENCES:\n{json.dumps(preference_profile, ensure_ascii=False, indent=2)}"
    )


def print_profiles_plan(plan: dict[str, Any]) -> None:
    print("")
    print("=" * 72)
    print("Step 1/3 - generate synthetic human profiles")
    print("=" * 72)
    print(f"Profiles selected : {len(plan['profile_indices'])} / {plan['num_profiles_available']}")
    print(f"Profile indices   : {plan['profile_indices']}")
    print(f"Response source   : {plan['response_source']}")
    print(f"Overwrite         : {plan['overwrite']}")
    print("")
    print(f"Profiles CSV      : {plan['mypersonality_path']}")
    print(f"Output dir        : {plan['output_dir']}")
    print("=" * 72)


def print_profiles_summary(
    *,
    output_dir: Path,
    saved: list[str],
    skipped: list[str],
    errors: list[dict[str, str]],
) -> None:
    print("")
    print("=" * 72)
    print("Profile generation summary")
    print("=" * 72)
    print(f"Saved   : {len(saved)} -> {', '.join(saved) if saved else 'none'}")
    print(f"Skipped : {len(skipped)} -> {', '.join(skipped) if skipped else 'none'}")
    print(f"Errors  : {len(errors)}")
    print(f"Output  : {output_dir}")
    if errors:
        print("")
        print("Errors preview:")
        for item in errors[:5]:
            print(f"- {item['human_id']}: {item['error']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
