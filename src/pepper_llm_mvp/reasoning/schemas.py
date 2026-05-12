from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PolicyInput:
    task_name: str
    prompt_text: str
    transcript: str
    normalized_response: str
    expected_answers: List[str]
    response_latency_s: float
    attention_state: str
    success_text: str
    hint_text: str
    reengage_text: str
    
    task_type: str = "recognition"
    turn_index: int = 1
    story_context: str = ""
    story_goal: str = ""
    story_phase: str = ""


@dataclass
class LLMOutput:
    classification: str
    feedback_type: str
    utterance: str
    next_action: str
    policy_decision: str
    generation_source: str = "template"
    template_reference: str = ""
    safety_notes: str = ""
    next_prompt: str = ""
    story_status: str = "in_progress"