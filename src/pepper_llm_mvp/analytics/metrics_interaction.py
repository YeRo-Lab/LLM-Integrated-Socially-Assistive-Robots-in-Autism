from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def _load_records(sessions_dir: str | Path) -> list[dict]:
    records = []
    for path in Path(sessions_dir).glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def compute_interaction_table(sessions_dir: str | Path) -> pd.DataFrame:
    records = _load_records(sessions_dir)
    if not records:
        return pd.DataFrame(columns=["Metric", "Value"])

    turns_per_session = []
    durations_min = []
    completions = []
    participations = []
    breakdowns = []

    for r in records:
        summary = r.get("session_summary", {})
        turns = r.get("turns", [])
        turns_per_session.append(summary.get("turn_count", len(turns)))
        durations_min.append(summary.get("session_duration_s", 0.0) / 60.0)
        completions.append(1.0 if summary.get("task_completed", False) else 0.0)
        participations.append(float(summary.get("participation_rate", 0.0)))
        breakdowns.append(1.0 if summary.get("interaction_breakdown", False) else 0.0)

    return pd.DataFrame([
        {"Metric": "Avg Turns per Session", "Value": sum(turns_per_session) / len(turns_per_session)},
        {"Metric": "Avg Session Duration (min)", "Value": sum(durations_min) / len(durations_min)},
        {"Metric": "Task Completion Rate (%)", "Value": 100.0 * sum(completions) / len(completions)},
        {"Metric": "Response Participation Rate (%)", "Value": 100.0 * sum(participations) / len(participations)},
        {"Metric": "Interaction Breakdown Rate (%)", "Value": 100.0 * sum(breakdowns) / len(breakdowns)},
    ])