from __future__ import annotations

from pepper_llm_mvp.reasoning.schemas import LLMOutput


class ActionMapper:
    def map_feedback_to_gesture(self, output: LLMOutput) -> str:
        if output.feedback_type == "praise":
            return "nod"
        if output.feedback_type == "hint":
            return "point_tablet"
        if output.feedback_type == "reengage":
            return "encourage"
        return "neutral"
