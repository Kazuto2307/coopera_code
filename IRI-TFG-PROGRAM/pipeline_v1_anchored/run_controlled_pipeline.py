from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    program_dir = Path(__file__).resolve().parent
    coopera_root = program_dir.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = (
        program_dir
        / "generated_data"
        / f"training_samples_from_coopera_{stamp}.jsonl"
    )
    default_summary = (
        program_dir
        / "generated_data"
        / f"training_samples_from_coopera_{stamp}_summary.json"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Controlled IRI TFG pipeline: run COOPERA human_sim and then build "
            "preference-model JSONL from its generated plans."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--coopera-root", type=Path, default=coopera_root)
    parser.add_argument(
        "--human-sim-script",
        type=Path,
        default=coopera_root / "habitat-lab" / "coopera_main" / "human_sim" / "human_sim.py",
    )
    parser.add_argument(
        "--builder-script",
        type=Path,
        default=program_dir / "build_training_data_from_coopera.py",
    )
    parser.add_argument("--scene-indices", type=int, nargs="+", required=True)
    parser.add_argument("--profile-indices", type=int, nargs="+", required=True)
    parser.add_argument("--max-days", type=int, default=1)
    parser.add_argument("--collab-type", type=int, choices=[1, 2], default=1)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--use-gpt-human", choices=["True", "False"], default="True")
    parser.add_argument("--start-logic-human", choices=["True", "False"], default="True")
    parser.add_argument(
        "--response-source",
        choices=["gpt_response", "llama_response"],
        default="gpt_response",
    )
    parser.add_argument(
        "--generation-strategy",
        choices=["qwen_profile", "rules_debug"],
        default="qwen_profile",
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--qwen-temperature", type=float, default=0.2)
    parser.add_argument("--first-task-only", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--allow-missing-profile", action="store_true")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary-output", type=Path, default=default_summary)
    parser.add_argument("--skip-human-sim", action="store_true")
    parser.add_argument("--skip-dataset-build", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing Qwen/Habitat.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional path for a manifest JSON. Defaults next to the output JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coopera_root = args.coopera_root.resolve()
    manifest_path = args.manifest_output or args.output.with_suffix(".manifest.json")
    started = time.time()

    human_cmd = build_human_sim_command(args)
    dataset_cmd = build_dataset_command(args)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "coopera_root": str(coopera_root),
        "dry_run": bool(args.dry_run),
        "skip_human_sim": bool(args.skip_human_sim),
        "skip_dataset_build": bool(args.skip_dataset_build),
        "human_sim_command": human_cmd,
        "dataset_build_command": dataset_cmd,
        "output": str(args.output),
        "summary_output": str(args.summary_output),
        "status": "planned",
    }

    print_section("Controlled COOPERA -> Preference Dataset Pipeline")
    print_command("human_sim", human_cmd, skipped=args.skip_human_sim)
    print_command("dataset_build", dataset_cmd, skipped=args.skip_dataset_build)

    if args.dry_run:
        manifest["status"] = "dry_run"
        write_manifest(manifest_path, manifest)
        print(f"\nDry run only. Manifest written to: {manifest_path}")
        return

    try:
        if not args.skip_human_sim:
            run_command("human_sim", human_cmd, cwd=coopera_root)
        if not args.skip_dataset_build:
            run_command("dataset_build", dataset_cmd, cwd=coopera_root)
        manifest["status"] = "completed"
    except subprocess.CalledProcessError as exc:
        manifest["status"] = "failed"
        manifest["failed_stage_returncode"] = int(exc.returncode)
        write_manifest(manifest_path, manifest)
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.time() - started, 3)
        write_manifest(manifest_path, manifest)

    print_section("Done")
    print(f"JSONL: {args.output}")
    print(f"Summary: {args.summary_output}")
    print(f"Manifest: {manifest_path}")


def build_human_sim_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        str(args.human_sim_script),
        "--use-gpt-human",
        args.use_gpt_human,
        "--start-logic-human",
        args.start_logic_human,
        "--collab-type",
        str(args.collab_type),
        "--max-days",
        str(args.max_days),
        "--scene-indices",
        *[str(i) for i in args.scene_indices],
        "--profile-indices",
        *[str(i) for i in args.profile_indices],
        "--gpu-id",
        str(args.gpu_id),
    ]


def build_dataset_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        str(args.builder_script),
        "--results-dir",
        str(args.coopera_root / "results"),
        "--response-source",
        args.response_source,
        "--collab-type",
        str(args.collab_type),
        "--generation-strategy",
        args.generation_strategy,
        "--qwen-model",
        args.qwen_model,
        "--qwen-temperature",
        str(args.qwen_temperature),
        "--human-ids",
        *[str(i) for i in args.profile_indices],
        "--days",
        *[str(i) for i in range(args.max_days)],
        "--output",
        str(args.output),
        "--summary-output",
        str(args.summary_output),
    ]
    if args.first_task_only:
        cmd.append("--first-task-only")
    if args.allow_missing_profile:
        cmd.append("--allow-missing-profile")
    if args.max_files is not None:
        cmd.extend(["--max-files", str(args.max_files)])
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    return cmd


def run_command(name: str, command: list[str], *, cwd: Path) -> None:
    print_section(f"Running {name}")
    print(" ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def print_section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_command(name: str, command: list[str], *, skipped: bool) -> None:
    status = "SKIPPED" if skipped else "READY"
    print()
    print(f"[{status}] {name}")
    print(" ".join(command))


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

