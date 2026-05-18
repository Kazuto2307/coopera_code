from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_human_profile_context(
    *,
    coopera_root: Path,
    results_dir: Path,
    response_source: str,
    human_id: str,
) -> dict[str, Any]:
    """Load the COOPERA human profile context used to simulate a person.

    This avoids importing `human_utils.py`, because that module pulls Habitat and
    GPU-heavy dependencies. The ordering mirrors `read_human_data_mypersonality`.
    """
    profile: dict[str, Any] = {
        "human_id": human_id,
        "traits_summary": load_latest_traits_summary(
            results_dir=results_dir,
            response_source=response_source,
            human_id=human_id,
        ),
        "mypersonality": None,
    }

    mypersonality_path = (
        coopera_root
        / "data"
        / "humanoids"
        / "humanoid_data"
        / "mypersonality_final.csv"
    )
    rows = load_mypersonality_profiles(mypersonality_path)
    human_index = int(human_id) if str(human_id).isdigit() else None
    if human_index is not None and 0 <= human_index < len(rows):
        profile["mypersonality"] = rows[human_index]

    return profile


def load_latest_traits_summary(
    *,
    results_dir: Path,
    response_source: str,
    human_id: str,
) -> dict[str, Any] | None:
    base = results_dir / "human" / response_source / "traits_summary" / str(human_id).zfill(5)
    if not base.exists():
        return None

    candidates = [
        path / "traits_summary.json"
        for path in base.iterdir()
        if path.is_dir() and (path / "traits_summary.json").exists()
    ]
    if not candidates:
        return None

    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    payload = json.loads(latest.read_text(encoding="utf-8"))
    return {
        "path": str(latest),
        "res": payload.get("res", ""),
        "system": payload.get("system"),
        "user": payload.get("user"),
        "assistant": payload.get("assistant"),
    }


def load_mypersonality_profiles(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []

    profile_text_by_authid: dict[str, str] = {}
    big_five_by_authid: dict[str, dict[str, float]] = {}
    row_count_by_authid: dict[str, int] = {}

    with csv_path.open("r", encoding="latin1", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            authid = str(row.get("#AUTHID", "")).strip()
            if not authid:
                continue
            status = str(row.get("STATUS", "")).strip()
            if authid in profile_text_by_authid:
                profile_text_by_authid[authid] += " " + status
                row_count_by_authid[authid] += 1
            else:
                profile_text_by_authid[authid] = status
                big_five_by_authid[authid] = {
                    "openness": _float_or_none(row.get("sOPN")),
                    "conscientiousness": _float_or_none(row.get("sCON")),
                    "extroversion": _float_or_none(row.get("sEXT")),
                    "agreeableness": _float_or_none(row.get("sAGR")),
                    "neuroticism": _float_or_none(row.get("sNEU")),
                }
                row_count_by_authid[authid] = 1

    sorted_authids = sorted(row_count_by_authid, key=row_count_by_authid.get, reverse=True)
    return [
        {
            "authid": authid,
            "profile_text": profile_text_by_authid[authid],
            "big_five": big_five_by_authid[authid],
            "num_status_rows": row_count_by_authid[authid],
        }
        for authid in sorted_authids
    ]


def profile_context_is_available(profile_context: dict[str, Any]) -> bool:
    traits = profile_context.get("traits_summary")
    mypersonality = profile_context.get("mypersonality")
    return bool(
        (isinstance(traits, dict) and str(traits.get("res", "")).strip())
        or (
            isinstance(mypersonality, dict)
            and (
                str(mypersonality.get("profile_text", "")).strip()
                or isinstance(mypersonality.get("big_five"), dict)
            )
        )
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None

