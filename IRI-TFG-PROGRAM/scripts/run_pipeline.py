"""Run the dataset-grounded preference-data pipeline sequentially.

  Step 1: generate_profiles.py             (Qwen/GPU)  synthetic humans
  Step 2a: build_situations_from_external  (CPU)       parse EPIC+Charades -> raw situations
  Step 2b: translate_situations.py         (Qwen/GPU)  -> robot-action situations
  Step 3: generate_training_data.py        (Qwen/GPU)  free (unforced) decision -> JSONL

Situations come straight from public datasets, with NO preference anchoring.
Each step can be skipped (--skip-profiles, --skip-build, --skip-translate,
--skip-training) so you can resume without redoing what already exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root = IRI-TFG-PROGRAM/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

GEN = PROJECT_ROOT / "data/generated"
DEFAULT_PROFILES_DIR = GEN / "profiles"
DEFAULT_SITU_RAW = GEN / "situations_external" / "external_situations.jsonl"
DEFAULT_SITU_ROBOT = GEN / "situations_external" / "external_situations_robot.jsonl"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = GEN / f"preference_training_{stamp}.jsonl"
    default_summary = GEN / f"preference_training_{stamp}_summary.json"

    parser = argparse.ArgumentParser(
        description="Run the dataset-grounded preference-data pipeline end to end."
    )

    # Step control
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="Skip dataset parsing (step 2a).")
    parser.add_argument("--skip-translate", action="store_true", help="Skip robot-action translation (step 2b).")
    parser.add_argument("--skip-training", action="store_true")

    # Step 1: profiles
    parser.add_argument("--coopera-root", type=Path, default=PROJECT_ROOT.parent)
    parser.add_argument("--mypersonality-path", type=Path, default=None)
    parser.add_argument("--response-source", choices=["gpt_response", "llama_response"], default="gpt_response")
    profile_sel = parser.add_mutually_exclusive_group()
    profile_sel.add_argument("--num-profiles", type=int, default=25)
    profile_sel.add_argument("--profile-indices", type=int, nargs="+", default=None)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument("--overwrite-profiles", action="store_true")

    # Step 2a: build situations from datasets
    parser.add_argument("--sources", nargs="+", choices=["epic", "charades"], default=["charades"])
    parser.add_argument("--max-per-source", type=int, default=None)
    parser.add_argument("--situations-raw", type=Path, default=DEFAULT_SITU_RAW)

    # Step 2b: translate to robot actions
    parser.add_argument("--situations-robot", type=Path, default=DEFAULT_SITU_ROBOT)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--reuse-map", action="store_true")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Quick test: translate only N RANDOM situations and train on those. "
            "The robot situations go to a *_sampleN.jsonl file so the full dataset "
            "is not overwritten. Use a small --target-samples too."
        ),
    )

    # Step 3: training data
    parser.add_argument("--target-samples", type=int, default=3000)
    parser.add_argument("--routine-consistency", choices=["enforce", "off"], default="off")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary-output", type=Path, default=default_summary)

    # Shared (same Qwen model everywhere; per-step temperatures keep their tuned defaults)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def banner(text: str) -> None:
    print("")
    print("=" * 72)
    print(text)
    print("=" * 72)


def run_step(label: str, module_path: Path, argv: list[str], dry_run: bool) -> None:
    import importlib.util

    banner(label)
    if dry_run:
        argv = argv + ["--dry-run"]
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [str(module_path)] + argv
    try:
        spec.loader.exec_module(module)
        module.main()
    finally:
        sys.argv = saved


def main() -> None:
    args = parse_args()
    shared = ["--qwen-model", args.qwen_model]
    if args.no_progress:
        shared += ["--no-progress"]

    # When sampling for a quick test, route the translated situations to a
    # sample-specific file so the full robot dataset is never overwritten.
    robot_path = args.situations_robot
    if args.sample is not None:
        robot_path = robot_path.with_name(
            f"{robot_path.stem}_sample{args.sample}{robot_path.suffix}"
        )

    # Step 1: profiles
    if not args.skip_profiles:
        argv = [
            "--coopera-root", str(args.coopera_root),
            "--response-source", args.response_source,
            "--output-dir", str(args.profiles_dir),
        ] + shared
        if args.mypersonality_path:
            argv += ["--mypersonality-path", str(args.mypersonality_path)]
        if args.profile_indices:
            argv += ["--profile-indices"] + [str(i) for i in args.profile_indices]
        else:
            argv += ["--num-profiles", str(args.num_profiles)]
        if args.overwrite_profiles:
            argv += ["--overwrite"]
        run_step("Step 1 - generate_profiles", SCRIPTS_DIR / "generate_profiles.py", argv, args.dry_run)
    else:
        banner("Step 1 - generate_profiles  [SKIPPED]")

    # Step 2a: build situations from datasets (CPU, no Qwen)
    if not args.skip_build:
        argv = [
            "--sources", *args.sources,
            "--output", str(args.situations_raw),
        ]
        if args.max_per_source is not None:
            argv += ["--max-per-source", str(args.max_per_source)]
        run_step("Step 2a - build_situations_from_external", SCRIPTS_DIR / "build_situations_from_external.py",
                 argv, args.dry_run)
    else:
        banner("Step 2a - build_situations_from_external  [SKIPPED]")

    # Step 2b: translate to robot actions
    if not args.skip_translate:
        argv = [
            "--input", str(args.situations_raw),
            "--output", str(robot_path),
            "--batch-size", str(args.batch_size),
        ] + shared
        if args.reuse_map:
            argv += ["--reuse-map"]
        if args.sample is not None:
            argv += ["--sample", str(args.sample), "--seed", str(args.seed)]
        run_step("Step 2b - translate_situations", SCRIPTS_DIR / "translate_situations.py", argv, args.dry_run)
    else:
        banner("Step 2b - translate_situations  [SKIPPED]")

    # Step 3: training data (free decision)
    if not args.skip_training:
        argv = [
            "--profiles-dir", str(args.profiles_dir),
            "--situations", str(robot_path),
            "--target-samples", str(args.target_samples),
            "--routine-consistency", args.routine_consistency,
            "--seed", str(args.seed),
            "--output", str(args.output),
            "--summary-output", str(args.summary_output),
        ] + shared
        if args.profile_indices:
            argv += ["--profile-indices"] + [str(i) for i in args.profile_indices]
        run_step("Step 3 - generate_training_data", SCRIPTS_DIR / "generate_training_data.py", argv, args.dry_run)
    else:
        banner("Step 3 - generate_training_data  [SKIPPED]")

    banner("Pipeline finished")


if __name__ == "__main__":
    main()
