from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def compute_ablation(sessions_dir: str | Path) -> pd.DataFrame:
    sessions = []
    for path in Path(sessions_dir).glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            sessions.append(json.load(f))

    rows = []
    for condition in ["scripted", "unconstrained_llm", "constrained_llm"]:
        cond_sessions = [s for s in sessions if s.get("condition") == condition]
        if not cond_sessions:
            rows.append({
                "Configuration": condition,
                "Coherence": 0.0,
                "Latency (s)": 0.0,
                "Turns/Session": 0.0,
                "Completion Rate (%)": 0.0,
            })
            continue

        coherence_scores = []
        latencies = []
        turns = []
        completions = []

        for s in cond_sessions:
            turns.append(len(s.get("turns", [])))
            completions.append(1.0 if s.get("session_summary", {}).get("task_completed", False) else 0.0)
            for t in s.get("turns", []):
                ts = t["timestamps"]
                latencies.append((ts["asr_end"] - ts["asr_start"]) + (ts["llm_end"] - ts["llm_start"]) + (ts["execution_end"] - ts["execution_start"]))
                coherence = 1.0
                if condition != "scripted":
                    if not t["constraint_valid"]:
                        coherence -= 0.25
                    if t["policy_output"]["feedback_type"] != t["expected_feedback_type"]:
                        coherence -= 0.25
                    if t.get("error_stage"):
                        coherence -= 0.25
                    if t["breakdown"]:
                        coherence -= 0.15
                    if t.get("fallback_used"):
                        coherence -= 0.05
                else:
                    if t.get("error_stage"):
                        coherence -= 0.25
                    if t["breakdown"]:
                        coherence -= 0.15
                coherence_scores.append(min(1.0, max(0.0, coherence)))

        rows.append({
            "Configuration": condition,
            "Coherence": sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0,
            "Latency (s)": sum(latencies) / len(latencies) if latencies else 0.0,
            "Turns/Session": sum(turns) / len(turns) if turns else 0.0,
            "Completion Rate (%)": 100.0 * sum(completions) / len(completions) if completions else 0.0,
        })
    return pd.DataFrame(rows)