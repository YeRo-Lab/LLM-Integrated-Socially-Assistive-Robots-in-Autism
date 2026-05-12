from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def compute_architecture_table(sessions_dir: str | Path) -> pd.DataFrame:
    sessions = []
    for path in Path(sessions_dir).glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            sessions.append(json.load(f))
    if not sessions:
        return pd.DataFrame(columns=["Metric", "Score"])

    total_turns = sum(len(s.get("turns", [])) for s in sessions)
    pipeline_success = 0
    constraint_compliance = 0
    policy_adherence = 0
    error_propagation = 0

    for s in sessions:
        for t in s.get("turns", []):
            if t["pipeline_success"]:
                pipeline_success += 1
            if t["constraint_valid"]:
                constraint_compliance += 1
            po = t["policy_output"]
            expected_policy = t.get("expected_policy_decision")
            actual_policy = po.get("policy_decision")
            allowed_policy_decisions = {expected_policy}
            if t.get("fallback_used"):
                allowed_policy_decisions.add("fallback_policy")
            if expected_policy == "story_continue":
                allowed_policy_decisions.add("story_conclude")
            adherent = (
                po.get("feedback_type") == t.get("expected_feedback_type")
                and actual_policy in allowed_policy_decisions
            )
            if adherent:
                policy_adherence += 1
            if t.get("error_stage"):
                error_propagation += 1

    return pd.DataFrame([
        {"Metric": "Pipeline success rate", "Score": pipeline_success / total_turns if total_turns else 0.0},
        {"Metric": "Constraint compliance", "Score": constraint_compliance / total_turns if total_turns else 0.0},
        {"Metric": "Policy adherence", "Score": policy_adherence / total_turns if total_turns else 0.0},
        {"Metric": "Error propagation", "Score": error_propagation / total_turns if total_turns else 0.0},
    ])