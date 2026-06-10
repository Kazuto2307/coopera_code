"""Quick exploratory study of a situations dataset.

Answers "does it make sense to use this?" by reporting diversity, redundancy,
balance and decision-context richness of a situations JSONL (the output of
build_situations_from_external.py or translate_situations.py).

  python IRI-TFG-PROGRAM/study_situations.py \
    --input IRI-TFG-PROGRAM/generated_data/situations_external/external_situations.jsonl

Optional: --report out.json (machine-readable), --plots <dir> (PNG charts,
needs matplotlib). Core report is pure stdlib and runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from iri_tfg_program import PROJECT_ROOT
from typing import Any

PROGRAM_DIR = PROJECT_ROOT
DEFAULT_INPUT = PROGRAM_DIR / "data/generated" / "situations_external" / "external_situations.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study a situations JSONL dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=None, help="Write the full report as JSON.")
    parser.add_argument("--plots", type=Path, default=None, help="Directory for PNG charts (needs matplotlib).")
    parser.add_argument("--top", type=int, default=15, help="How many entries in top-N lists.")
    return parser.parse_args()


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Situations file not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 1) if total else 0.0


def study(situations: list[dict[str, Any]], *, top: int) -> dict[str, Any]:
    total = len(situations)
    by_source = Counter()
    by_activity = Counter()
    by_location = Counter()
    by_kind = Counter()
    by_urgency = Counter()
    by_sensitivity = Counter()
    by_time = Counter()
    user_states = Counter()
    objects = Counter()
    action_counts = Counter()

    action_word_lens: list[int] = []
    n_objects: list[int] = []
    no_objects = 0
    unknown_location = 0
    translated = 0
    domain_location = Counter()

    for s in situations:
        ai = s.get("action_input") or {}
        ci = s.get("context_input") or {}
        feat = s.get("structured_task_features") or {}
        meta = s.get("source_metadata") or {}

        by_source[s.get("source_dataset", "unknown")] += 1
        activity = ai.get("activity") or "unknown"
        location = ci.get("location_current") or "unknown"
        by_activity[activity] += 1
        by_location[location] += 1
        domain_location[(activity, location)] += 1
        by_kind[feat.get("kind", "unknown")] += 1
        by_urgency[feat.get("urgency", "unknown")] += 1
        by_sensitivity[feat.get("sensitivity", "unknown")] += 1
        by_time[ci.get("time_of_day", "unknown")] += 1

        for st in ci.get("user_state") or []:
            user_states[st] += 1
        objs = ci.get("objects_nearby") or []
        n_objects.append(len(objs))
        if not objs:
            no_objects += 1
        for o in objs:
            objects[o] += 1
        if location == "unknown":
            unknown_location += 1
        if meta.get("original_action_text"):
            translated += 1

        action = str(ai.get("action_text", "")).strip()
        if action:
            action_counts[action] += 1
            action_word_lens.append(len(action.split()))

    unique_actions = len(action_counts)
    return {
        "total": total,
        "is_translated_to_robot_actions": translated > 0,
        "translated_fraction_pct": pct(translated, total),
        "by_source": dict(by_source),
        "by_activity": dict(by_activity.most_common()),
        "by_location_top": dict(by_location.most_common(top)),
        "num_unique_locations": len(by_location),
        "unknown_location_pct": pct(unknown_location, total),
        "by_kind": dict(by_kind.most_common()),
        "by_urgency": dict(by_urgency),
        "by_sensitivity": dict(by_sensitivity),
        "by_time_of_day": dict(by_time),
        "user_state_top": dict(user_states.most_common(top)),
        "objects": {
            "situations_with_no_objects_pct": pct(no_objects, total),
            "avg_objects": round(statistics.mean(n_objects), 2) if n_objects else 0,
            "median_objects": statistics.median(n_objects) if n_objects else 0,
            "top_objects": dict(objects.most_common(top)),
        },
        "action_text": {
            "unique": unique_actions,
            "redundancy_ratio": round(unique_actions / total, 3) if total else 0,
            "avg_words": round(statistics.mean(action_word_lens), 2) if action_word_lens else 0,
            "most_frequent": dict(action_counts.most_common(top)),
        },
        "decision_context": {
            "has_known_location_pct": pct(total - unknown_location, total),
            "has_objects_pct": pct(total - no_objects, total),
        },
        "top_domain_location_combos": {
            f"{a} @ {loc}": n for (a, loc), n in domain_location.most_common(top)
        },
    }


def diagnose(report: dict[str, Any]) -> list[tuple[str, str]]:
    """Heuristic flags to judge whether the dataset is usable."""
    total = report["total"]
    out: list[tuple[str, str]] = []

    src = report["by_source"]
    if src:
        top_src, top_n = max(src.items(), key=lambda kv: kv[1])
        if pct(top_n, total) > 80:
            out.append(("WARN", f"Source imbalance: '{top_src}' is {pct(top_n, total)}% of the data."))
        else:
            out.append(("OK", f"Sources reasonably mixed ({len(src)} sources)."))

    loc = report["by_location_top"]
    if loc:
        top_loc, top_n = next(iter(loc.items()))
        if pct(top_n, total) > 60:
            out.append(("WARN", f"Location skew: '{top_loc}' is {pct(top_n, total)}% of situations."))
        else:
            out.append(("OK", f"Locations spread across {report['num_unique_locations']} rooms."))

    rr = report["action_text"]["redundancy_ratio"]
    if rr < 0.2:
        out.append(("WARN", f"High redundancy: only {rr:.0%} of action_texts are unique."))
    elif rr < 0.5:
        out.append(("INFO", f"Moderate redundancy: {rr:.0%} unique action_texts ({report['action_text']['unique']})."))
    else:
        out.append(("OK", f"Good action diversity: {rr:.0%} unique ({report['action_text']['unique']})."))

    no_obj = report["objects"]["situations_with_no_objects_pct"]
    if no_obj > 50:
        out.append(("WARN", f"Sparse context: {no_obj}% of situations have no objects_nearby."))
    else:
        out.append(("OK", f"{report['decision_context']['has_objects_pct']}% of situations name nearby objects."))

    dom = report["by_activity"]
    if dom:
        top_dom, top_n = max(dom.items(), key=lambda kv: kv[1])
        if pct(top_n, total) > 60:
            out.append(("WARN", f"Domain skew: '{top_dom}' is {pct(top_n, total)}% of situations."))
        else:
            out.append(("OK", f"Assistance domains spread across {len(dom)} categories."))

    if report["by_time_of_day"].get("unknown", 0) == total:
        out.append(("INFO", "No time-of-day signal (all 'unknown') — these datasets carry no clock."))

    if not report["is_translated_to_robot_actions"]:
        out.append(("INFO", "action_text is still the HUMAN action — run translate_situations.py for robot actions."))

    return out


def print_counter(title: str, data: dict[str, Any], total: int, limit: int | None = None) -> None:
    print(f"\n{title}:")
    items = list(data.items())
    if limit:
        items = items[:limit]
    if not items:
        print("  (none)")
        return
    width = max(len(str(k)) for k, _ in items)
    for key, n in items:
        bar = "#" * int(40 * n / max(data.values())) if data else ""
        print(f"  {str(key).ljust(width)} : {n:>7}  {pct(n, total):>5}%  {bar}")


def print_report(report: dict[str, Any], *, top: int) -> None:
    total = report["total"]
    print("\n" + "=" * 72)
    print("SITUATIONS DATASET STUDY")
    print("=" * 72)
    print(f"Total situations            : {total}")
    print(f"Translated to robot actions : {report['is_translated_to_robot_actions']} "
          f"({report['translated_fraction_pct']}%)")
    print(f"Unique action_texts         : {report['action_text']['unique']} "
          f"(redundancy ratio {report['action_text']['redundancy_ratio']}, "
          f"avg {report['action_text']['avg_words']} words)")
    print(f"Unique locations            : {report['num_unique_locations']} "
          f"(unknown {report['unknown_location_pct']}%)")

    print_counter("By source dataset", report["by_source"], total)
    print_counter("By assistance domain", report["by_activity"], total)
    print_counter(f"Top {top} locations", report["by_location_top"], total)
    print_counter("By task kind", report["by_kind"], total)
    print_counter("By urgency", report["by_urgency"], total)
    print_counter("By sensitivity", report["by_sensitivity"], total)
    print_counter("Top user states", report["user_state_top"], total)
    print_counter(f"Top {top} objects nearby", report["objects"]["top_objects"], total)
    print_counter(f"Top {top} most frequent action_texts", report["action_text"]["most_frequent"], total)
    print_counter(f"Top {top} domain @ location combos", report["top_domain_location_combos"], total)

    print(f"\nDecision context richness:")
    print(f"  situations with a known location : {report['decision_context']['has_known_location_pct']}%")
    print(f"  situations naming nearby objects : {report['decision_context']['has_objects_pct']}%")
    print(f"  situations with no objects        : {report['objects']['situations_with_no_objects_pct']}%")

    print("\n" + "=" * 72)
    print("VERDICT (heuristics)")
    print("=" * 72)
    for level, msg in diagnose(report):
        print(f"  [{level}] {msg}")
    print("=" * 72)


def make_plots(report: dict[str, Any], plots_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"\n[plots skipped] matplotlib unavailable: {exc}")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    def barh(data: dict[str, Any], title: str, fname: str, limit: int = 15) -> None:
        items = list(data.items())[:limit][::-1]
        if not items:
            return
        labels = [str(k) for k, _ in items]
        values = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(items))))
        ax.barh(labels, values, color="#4C78A8")
        ax.set_title(title)
        ax.set_xlabel("situations")
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=120)
        plt.close(fig)

    barh(report["by_activity"], "Assistance domain", "domain.png")
    barh(report["by_location_top"], "Top locations", "locations.png")
    barh(report["by_urgency"], "Urgency", "urgency.png")
    barh(report["by_sensitivity"], "Sensitivity", "sensitivity.png")
    barh(report["objects"]["top_objects"], "Top objects nearby", "objects.png")
    barh(report["action_text"]["most_frequent"], "Most frequent action_texts", "top_actions.png")
    print(f"\nPlots written to {plots_dir}")


def main() -> None:
    args = parse_args()
    situations = load(args.input)
    report = study(situations, top=args.top)
    report["input"] = str(args.input)
    print_report(report, top=args.top)
    if args.plots:
        make_plots(report, args.plots)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.report}")


if __name__ == "__main__":
    main()
