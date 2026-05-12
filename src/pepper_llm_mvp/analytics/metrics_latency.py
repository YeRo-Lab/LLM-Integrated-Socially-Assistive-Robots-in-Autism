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


def compute_latency_table(sessions_dir: str | Path) -> pd.DataFrame:
    rows = []
    for session in _load_records(sessions_dir):
        for turn in session.get("turns", []):
            ts = turn["timestamps"]
            asr = ts["asr_end"] - ts["asr_start"]
            llm = ts["llm_end"] - ts["llm_start"]
            exe = ts["execution_end"] - ts["execution_start"]
            total = (ts["asr_end"] - ts["asr_start"]) + (ts["llm_end"] - ts["llm_start"]) + (ts["execution_end"] - ts["execution_start"])
            rows.append({
                "session_id": session["session_id"],
                "turn_index": turn["turn_index"],
                "asr_s": asr,
                "llm_s": llm,
                "execution_s": exe,
                "total_s": total,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Metric", "Mean", "Std Dev", "Max"])
    summary = pd.DataFrame([
        {"Metric": "End-to-End Latency (s)", "Mean": df["total_s"].mean(), "Std Dev": df["total_s"].std(ddof=0), "Max": df["total_s"].max()},
        {"Metric": "Speech-to-Text (ASR) (s)", "Mean": df["asr_s"].mean(), "Std Dev": df["asr_s"].std(ddof=0), "Max": df["asr_s"].max()},
        {"Metric": "LLM Response Time (s)", "Mean": df["llm_s"].mean(), "Std Dev": df["llm_s"].std(ddof=0), "Max": df["llm_s"].max()},
        {"Metric": "Robot Execution Delay (s)", "Mean": df["execution_s"].mean(), "Std Dev": df["execution_s"].std(ddof=0), "Max": df["execution_s"].max()},
    ])
    return summary
