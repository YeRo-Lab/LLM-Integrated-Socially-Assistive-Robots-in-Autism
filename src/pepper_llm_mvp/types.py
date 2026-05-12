from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime, UTC


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_session_id(prefix: str = "sess") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class TaskConfig:
    task_name: str
    task_type: str
    intro_text: str
    stimulus_type: str
    stimulus_path: str
    prompt_text: str
    expected_answers: List[str]
    hint_text: str
    reengage_text: str
    success_text: str
    max_wait_seconds: int = 6
    max_retries: int = 2
    story: Optional[Dict[str, Any]] = None

@dataclass
class PerceptionResult:
    transcript_raw: str
    transcript_normalized: str
    response_latency_s: float
    attention_state: str
    asr_confidence: float
    scenario_label: str = "unknown"


@dataclass
class PolicyDecision:
    classification: str
    feedback_type: str
    utterance: str
    next_action: str
    policy_decision: str


@dataclass
class TurnTimestamps:
    prompt_start: float
    asr_start: float
    asr_end: float
    llm_start: float
    llm_end: float
    execution_start: float
    execution_end: float


@dataclass
class TurnRecord:
    turn_index: int
    timestamps: TurnTimestamps
    prompt_text: str
    stimulus: str
    transcript_raw: str
    transcript_normalized: str
    response_latency_s: float
    attention_state: str
    asr_confidence: float
    policy_output: Dict[str, Any]
    expected_feedback_type: str
    expected_policy_decision: str
    scenario_label: str
    constraint_valid: bool
    fallback_used: bool
    pipeline_success: bool
    breakdown: bool
    error_stage: Optional[str] = None


@dataclass
class SessionSummary:
    task_completed: bool
    turn_count: int
    avg_latency_s: float
    session_duration_s: float
    participation_rate: float
    interaction_breakdown: bool


@dataclass
class SessionRecord:
    session_id: str
    condition: str
    task_name: str
    start_time: str
    end_time: str = ""
    turns: List[TurnRecord] = field(default_factory=list)
    session_summary: Optional[SessionSummary] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)