from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class CooperaTask:
    index: int
    thought: str
    act: list[Any]
    collab_type: int


@dataclass(frozen=True)
class CooperaPlan:
    source_path: Path
    human_id: str
    scene_id: str
    day: str
    hour_folder: str
    time_text: str
    intention: str
    tasks: list[CooperaTask]
    raw_json: dict[str, Any]


def iter_predicates_reflection_files(
    *,
    results_dir: Path,
    response_source: str,
    collab_type: int,
    human_ids: set[str] | None = None,
    scene_ids: set[str] | None = None,
    days: set[str] | None = None,
    max_files: int | None = None,
) -> Iterator[Path]:
    """Yield COOPERA predicates_reflection_2 files in stable order."""
    base = (
        results_dir
        / "human"
        / response_source
        / f"collaboration_{collab_type}"
        / "predicates_reflection_2"
    )
    if not base.exists():
        return

    yielded = 0
    for human_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if human_ids is not None and human_dir.name not in human_ids:
            continue
        for scene_dir in sorted(p for p in human_dir.iterdir() if p.is_dir()):
            if scene_ids is not None and scene_dir.name not in scene_ids:
                continue
            for day_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
                if days is not None and day_dir.name not in days:
                    continue
                for hour_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
                    json_path = hour_dir / "predicates_reflection_2.json"
                    if not json_path.exists():
                        continue
                    yield json_path
                    yielded += 1
                    if max_files is not None and yielded >= max_files:
                        return


def parse_predicates_reflection_file(path: Path, *, collab_type: int) -> CooperaPlan:
    raw = json.loads(path.read_text(encoding="utf-8"))
    res = str(raw.get("res", ""))
    time_text, intention, tasks = parse_coopera_response_text(
        res,
        collab_type=collab_type,
    )
    day_dir = path.parents[1]
    scene_dir = path.parents[2]
    human_dir = path.parents[3]
    hour_dir = path.parent
    return CooperaPlan(
        source_path=path,
        human_id=human_dir.name,
        scene_id=scene_dir.name,
        day=day_dir.name,
        hour_folder=hour_dir.name,
        time_text=time_text,
        intention=intention,
        tasks=tasks,
        raw_json=raw,
    )


def parse_coopera_response_text(res: str, *, collab_type: int) -> tuple[str, str, list[CooperaTask]]:
    """Parse the final LLM response stored in COOPERA's JSON field `res`.

    This mirrors the semantics used by Habitat's `extract_code`, but returns
    all tasks instead of only the first one.
    """
    lines = [line.rstrip() for line in res.splitlines()]
    non_empty = [line.strip() for line in lines if line.strip()]
    if len(non_empty) < 2:
        raise ValueError("COOPERA response must contain Time and Intention lines.")

    time_text = _strip_prefix(non_empty[0], "Time:")
    intention = _strip_prefix(non_empty[1], "Intention:")

    in_revised_tasks = False
    tasks: list[CooperaTask] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Revised Tasks:"):
            in_revised_tasks = True
            continue
        if not in_revised_tasks or "Thought:" not in line or "Act:" not in line:
            continue

        thought = _extract_between(line, "Thought:", "Act:").strip()
        act = parse_act(line, collab_type=collab_type)
        tasks.append(
            CooperaTask(
                index=len(tasks) + 1,
                thought=thought,
                act=act,
                collab_type=collab_type,
            )
        )

    return time_text, intention, tasks


def parse_act(line: str, *, collab_type: int) -> list[Any]:
    act_match = re.search(r"Act:\s*\[(.*)\]\s*$", line)
    if not act_match:
        raise ValueError(f"Could not find Act: [...] in line: {line}")
    content = act_match.group(1).strip()
    if collab_type == 1:
        return _parse_collab1_act(content)
    if collab_type == 2:
        return _parse_collab2_act(content)
    raise ValueError(f"Unsupported collab_type={collab_type}")


def _parse_collab1_act(content: str) -> list[Any]:
    keyed = [
        _extract_int_key(content, "static_obj_id"),
        _extract_str_key(content, "static_obj_name"),
        _extract_int_key(content, "dynamic_obj_id"),
        _extract_str_key(content, "dynamic_obj_name"),
    ]
    if all(value is not None for value in keyed):
        return keyed
    return _literal_list_fallback(content, expected_len=4)


def _parse_collab2_act(content: str) -> list[Any]:
    keyed = [
        _extract_int_key(content, "type"),
        _extract_int_key(content, "inter_obj_id"),
        _extract_str_key(content, "inter_obj_name"),
        _extract_str_key(content, "inhand_obj_name"),
        _extract_str_key(content, "motion"),
    ]
    if all(value is not None for value in keyed):
        return keyed
    return _literal_list_fallback(content, expected_len=5)


def _literal_list_fallback(content: str, *, expected_len: int) -> list[Any]:
    parsed = ast.literal_eval(f"[{content}]")
    if not isinstance(parsed, list) or len(parsed) != expected_len:
        raise ValueError(f"Expected {expected_len} Act fields, got: {parsed}")
    return [_clean_value(value) for value in parsed]


def _extract_int_key(content: str, key: str) -> int | None:
    match = re.search(rf"{re.escape(key)}\s*:\s*(-?\d+)", content)
    if not match:
        return None
    return int(match.group(1))


def _extract_str_key(content: str, key: str) -> str | None:
    patterns = [
        rf"{re.escape(key)}\s*:\s*'([^']*)'",
        rf'{re.escape(key)}\s*:\s*"([^"]*)"',
        rf"{re.escape(key)}\s*:\s*([^,]+)(?:,|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return _clean_text(match.group(1))
    return None


def _extract_between(text: str, start: str, end: str) -> str:
    start_idx = text.find(start)
    if start_idx == -1:
        return ""
    start_idx += len(start)
    end_idx = text.find(end, start_idx)
    if end_idx == -1:
        return text[start_idx:]
    return text[start_idx:end_idx]


def _strip_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text.strip()


def _clean_value(value: Any) -> Any:
    if isinstance(value, str):
        return _clean_text(value)
    return value


def _clean_text(text: Any) -> str:
    return str(text).strip().strip(",").strip().strip("'").strip('"')

