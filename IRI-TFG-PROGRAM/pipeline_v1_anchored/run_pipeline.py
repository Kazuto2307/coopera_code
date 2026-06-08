"""Wrapper that runs the 3-step preference-data pipeline sequentially.

  Step 1: generate_profiles.py
  Step 2: generate_situations.py
  Step 3: generate_training_data.py

Each step can be skipped individually (--skip-profiles, --skip-situations,
--skip-training) so you can resume a partial run without regenerating what
already exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROGRAM_DIR = Path(__file__).resolve().parent

DEFAULT_PROFILES_DIR   = PROGRAM_DIR / "generated_data" / "profiles"
DEFAULT_SITUATIONS_DIR = PROGRAM_DIR / "generated_data" / "situations"

DEFAULT_TIMES = [
    "6 am", "7 am", "8 am", "9 am", "10 am", "11 am", "12 pm",
    "1 pm", "2 pm", "3 pm", "4 pm", "5 pm", "6 pm", "7 pm", "8 pm",
    "9 pm", "10 pm", "11 pm", "12 am", "1 am", "2 am", "3 am", "4 am", "5 am",
]


def parse_args() -> argparse.Namespace:
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = PROGRAM_DIR / "generated_data" / f"preference_training_{stamp}.jsonl"
    default_summary = PROGRAM_DIR / "generated_data" / f"preference_training_{stamp}_summary.json"

    parser = argparse.ArgumentParser(
        description=(
            "Run the full 3-step preference-data pipeline: "
            "profiles -> situations -> training JSONL."
        )
    )

    # ── Step control ──────────────────────────────────────────────────────────
    parser.add_argument("--skip-profiles",   action="store_true", help="Skip step 1.")
    parser.add_argument("--skip-situations", action="store_true", help="Skip step 2.")
    parser.add_argument("--skip-training",   action="store_true", help="Skip step 3.")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    parser.add_argument("--coopera-root", type=Path, default=PROGRAM_DIR.parent)
    parser.add_argument("--mypersonality-path", type=Path, default=None)
    parser.add_argument("--response-source", choices=["gpt_response", "llama_response"], default="gpt_response")
    profile_sel = parser.add_mutually_exclusive_group()
    profile_sel.add_argument("--num-profiles", type=int, default=25)
    profile_sel.add_argument("--profile-indices", type=int, nargs="+", default=None)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument("--overwrite-profiles", action="store_true")

    # ── Step 2 ────────────────────────────────────────────────────────────────
    sig_sel = parser.add_mutually_exclusive_group()
    sig_sel.add_argument("--signals", nargs="+", default=None)
    sig_sel.add_argument("--all-signals", action="store_true", default=True)
    parser.add_argument("--situations-per-signal", type=int, default=24)
    parser.add_argument("--times", nargs="*", default=None)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--situations-dir", type=Path, default=DEFAULT_SITUATIONS_DIR)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    parser.add_argument("--target-samples", type=int, default=3000)
    parser.add_argument(
        "--label-weights", default=None,
        help='JSON dict, e.g. \'{"do_now": 0.25, "do_later": 0.25, "remind": 0.25, "no_action": 0.25}\'',
    )
    parser.add_argument("--max-attempts-per-sample", type=int, default=3)
    parser.add_argument("--accept-mismatch-if-useful", action="store_true", default=True)
    parser.add_argument("--routine-consistency", choices=["enforce", "off"], default="enforce")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary-output", type=Path, default=default_summary)

    # ── Shared ────────────────────────────────────────────────────────────────
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--qwen-temperature", type=float, default=0.35)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1800)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def banner(text: str) -> None:
    print("")
    print("=" * 72)
    print(text)
    print("=" * 72)


def run_step(label: str, module_path: Path, argv: list[str], dry_run: bool) -> None:
    """Import and call main() of a sibling script with the given argv."""
    import importlib.util

    banner(label)
    if dry_run:
        argv = argv + ["--dry-run"]

    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    saved_argv = sys.argv
    sys.argv = [str(module_path)] + argv
    try:
        spec.loader.exec_module(module)
        module.main()
    finally:
        sys.argv = saved_argv


def main() -> None:
    args = parse_args()
    times = args.times or DEFAULT_TIMES
    qwen_shared = [
        "--qwen-model", args.qwen_model,
        "--qwen-temperature", str(args.qwen_temperature),
        "--qwen-max-new-tokens", str(args.qwen_max_new_tokens),
    ]
    if args.no_progress:
        qwen_shared += ["--no-progress"]

    # ── Step 1: profiles ──────────────────────────────────────────────────────
    if not args.skip_profiles:
        argv1 = [
            "--coopera-root", str(args.coopera_root),
            "--response-source", args.response_source,
            "--output-dir", str(args.profiles_dir),
        ] + qwen_shared
        if args.mypersonality_path:
            argv1 += ["--mypersonality-path", str(args.mypersonality_path)]
        if args.profile_indices:
            argv1 += ["--profile-indices"] + [str(i) for i in args.profile_indices]
        else:
            argv1 += ["--num-profiles", str(args.num_profiles)]
        if args.overwrite_profiles:
            argv1 += ["--overwrite"]
        run_step("Step 1/3 — generate_profiles", PROGRAM_DIR / "generate_profiles.py",
                 argv1, args.dry_run)
    else:
        banner("Step 1/3 — generate_profiles  [SKIPPED]")

    # ── Step 2: situations ────────────────────────────────────────────────────
    if not args.skip_situations:
        argv2 = [
            "--situations-per-signal", str(args.situations_per_signal),
            "--times", *times,
            "--days", str(args.days),
            "--output-dir", str(args.situations_dir),
        ] + qwen_shared
        if args.signals:
            argv2 += ["--signals"] + args.signals
        else:
            argv2 += ["--all-signals"]
        run_step("Step 2/3 — generate_situations", PROGRAM_DIR / "generate_situations.py",
                 argv2, args.dry_run)
    else:
        banner("Step 2/3 — generate_situations  [SKIPPED]")

    # ── Step 3: training data ─────────────────────────────────────────────────
    if not args.skip_training:
        argv3 = [
            "--profiles-dir", str(args.profiles_dir),
            "--situations-dir", str(args.situations_dir),
            "--target-samples", str(args.target_samples),
            "--max-attempts-per-sample", str(args.max_attempts_per_sample),
            "--routine-consistency", args.routine_consistency,
            "--output", str(args.output),
            "--summary-output", str(args.summary_output),
        ] + qwen_shared
        if args.label_weights:
            argv3 += ["--label-weights", args.label_weights]
        if args.accept_mismatch_if_useful:
            argv3 += ["--accept-mismatch-if-useful"]
        if args.profile_indices:
            argv3 += ["--profile-indices"] + [str(i) for i in args.profile_indices]
        run_step("Step 3/3 — generate_training_data", PROGRAM_DIR / "generate_training_data.py",
                 argv3, args.dry_run)
    else:
        banner("Step 3/3 — generate_training_data  [SKIPPED]")

    banner("Pipeline finished")


if __name__ == "__main__":
    main()
