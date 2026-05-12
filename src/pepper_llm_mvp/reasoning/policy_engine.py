from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional

from pepper_llm_mvp.reasoning.schemas import PolicyInput, LLMOutput


class PolicyEngine:
    def __init__(
        self,
        condition: str = "constrained_llm",
        simulation_cfg: Dict[str, Any] | None = None,
        llm_cfg: Dict[str, Any] | None = None,
        seed: int = 7,
    ):
        if condition not in {"scripted", "unconstrained_llm", "constrained_llm"}:
            raise ValueError(f"Unsupported condition: {condition}")

        self.condition = condition
        self.cfg = simulation_cfg or {}
        self.llm_cfg = llm_cfg or {}
        self.rng = random.Random(seed)
        self.real_llm_client: Optional[Any] = None

        use_real_llm = bool(self.llm_cfg.get("use_real_llm", False))
        provider = self.llm_cfg.get("provider", "mock")

        if use_real_llm and provider == "openai" and condition in {"constrained_llm", "unconstrained_llm"}:
            from pepper_llm_mvp.reasoning.llm_client import OpenAILLMClient

            self.real_llm_client = OpenAILLMClient(
                model=self.llm_cfg.get("model", "gpt-5.5"),
                max_output_tokens=int(self.llm_cfg.get("max_output_tokens", 350)),
                timeout_s=int(self.llm_cfg.get("timeout_s", 20)),
                generation_cfg=self.llm_cfg.get("generation", {}),
                mode=condition,
            )

    def _sleep_for_model_latency(self) -> None:
        latency_cfg = self.cfg.get("latency", {})
        mean = float(latency_cfg.get("llm_mean_s", {}).get(self.condition, 0.2))
        std = float(latency_cfg.get("llm_std_s", {}).get(self.condition, 0.05))
        simulated = max(0.03, self.rng.gauss(mean, std))
        time.sleep(min(simulated, 3.5))

    def _policy_error_rate(self) -> float:
        failure_cfg = self.cfg.get("failure", {})
        key = {
            "scripted": "scripted_policy_error_rate",
            "unconstrained_llm": "unconstrained_policy_error_rate",
            "constrained_llm": "constrained_policy_error_rate",
        }[self.condition]
        return float(failure_cfg.get(key, 0.0))

    def decide(self, inp: PolicyInput) -> tuple[LLMOutput, float, float]:
        if self.real_llm_client is not None:
            return self.real_llm_client.decide(inp)

        if inp.task_type == "storytelling":
            start = time.perf_counter()
            self._sleep_for_model_latency()

            ending_cues = ["the end", "done", "finished", "that's all", "end story"]
            user_ended_story = any(cue in inp.normalized_response for cue in ending_cues)

            if user_ended_story:
                out = LLMOutput(
                    classification="correct",
                    feedback_type="praise",
                    utterance="That is a nice ending. Pip had a kind and friendly day.",
                    next_action="continue",
                    policy_decision="story_conclude",
                    generation_source="template",
                    template_reference="story_closing_template",
                    safety_notes="Safe story wrap-up after user ended the story.",
                    next_prompt="",
                    story_status="complete",
                )

            elif not inp.transcript.strip() or inp.attention_state == "disengaged":
                next_prompt = "What could happen next?"
                out = LLMOutput(
                    classification="disengaged",
                    feedback_type="reengage",
                    utterance=f"{inp.reengage_text} {next_prompt}",
                    next_action="retry",
                    policy_decision="engagement_recovery",
                    generation_source="template",
                    template_reference="reengage_text",
                    safety_notes="Storytelling fallback reengagement.",
                    next_prompt=next_prompt,
                    story_status="in_progress",
                )

            else:
                if inp.story_phase == "story_contribution":
                    next_prompt = "How do you think Pip feels?"
                elif inp.story_phase == "emotion_identification":
                    next_prompt = "What should Pip do next?"
                else:
                    next_prompt = "What happens next?"

                out = LLMOutput(
                    classification="correct",
                    feedback_type="praise",
                    utterance=f"Nice idea. Let's add that to the story: {inp.transcript}. {next_prompt}",
                    next_action="retry",
                    policy_decision="story_continue",
                    generation_source="template",
                    template_reference="story_mock_template",
                    safety_notes="Safe storytelling mock continuation.",
                    next_prompt=next_prompt,
                    story_status="in_progress",
                )

            end = time.perf_counter()
            return out, start, end

        start = time.perf_counter()
        self._sleep_for_model_latency()

        if not inp.transcript.strip() or inp.attention_state == "disengaged":
            out = LLMOutput(
                classification="disengaged",
                feedback_type="reengage",
                utterance=inp.reengage_text,
                next_action="retry",
                policy_decision="engagement_recovery",
            )
        elif inp.normalized_response in [x.lower() for x in inp.expected_answers]:
            utterance = inp.success_text
            if self.condition == "unconstrained_llm":
                utterance += (
                    " Tell me more about every possible feeling and explain your "
                    "full reasoning in detail."
                )
            out = LLMOutput(
                classification="correct",
                feedback_type="praise",
                utterance=utterance,
                next_action="continue",
                policy_decision="reinforce_and_continue",
                generation_source="template",
                template_reference="success_text",
                safety_notes="Template fallback response.",
            )
        else:
            utterance = inp.hint_text
            if self.condition == "unconstrained_llm":
                utterance = (
                    "That may not match exactly, but let us discuss a broad "
                    "interpretation of the scene and several alternate labels."
                )
            out = LLMOutput(
                classification="incorrect",
                feedback_type="hint",
                utterance=utterance,
                next_action="retry",
                policy_decision="hint_then_retry",
                generation_source="template",
                template_reference="hint_text",
                safety_notes="Template fallback response.",
            )

        if self.rng.random() < self._policy_error_rate():
            if out.feedback_type == "praise":
                out.feedback_type = "hint"
                out.next_action = "retry"
                out.policy_decision = "misclassified_correct_as_incorrect"
            elif out.feedback_type == "hint":
                out.feedback_type = "praise"
                out.next_action = "continue"
                out.policy_decision = "misclassified_incorrect_as_correct"
            else:
                out.feedback_type = "hint"
                out.next_action = "retry"
                out.policy_decision = "missed_disengagement"

        end = time.perf_counter()
        return out, start, end