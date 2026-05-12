from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pepper_llm_mvp.types import TaskConfig


@dataclass
class TurnScenario:
    label: str
    transcript: str
    attention_state: str
    response_latency_s: float
    asr_confidence: float
    expected_feedback_type: str
    expected_policy_decision: str
    end_session_after_turn: bool = False
    completed_after_turn: bool = False
    force_asr_dropout: bool = False


class ScenarioGenerator:
    def __init__(self, simulation_cfg: Dict[str, Any], task: TaskConfig, seed: int = 7, condition: str = "constrained_llm"):
        self.cfg = simulation_cfg
        self.task = task
        self.condition = condition
        self.rng = random.Random(seed)
        self.plan: List[TurnScenario] = self._sample_plan()

    def _latency(self, max_wait_s: int) -> float:
        behavior_cfg = self.cfg.get("user_response_latency", {})
        mean = float(behavior_cfg.get("mean_s", 4.2))
        std = float(behavior_cfg.get("std_s", 1.1))
        value = max(1.5, self.rng.gauss(mean, std))
        return round(min(value, max_wait_s), 2)

    def _correct_answer(self) -> str:
        return self.task.expected_answers[0]

    def _incorrect_answer(self) -> str:
        wrong_pool = {
            "emotion_recognition": ["sad", "angry", "surprised", "tired"],
            "turn_taking": ["I do not know", "maybe", "nothing", "robot"],
        }
        options = wrong_pool.get(self.task.task_name, ["I don't know", "maybe"])
        return self.rng.choice(options)

    def _ambiguous_answer(self) -> str:
        options = {
            "emotion_recognition": ["maybe good", "not sure", "kind of happy"],
            "turn_taking": ["an animal", "something cute", "I forgot"],
        }
        return self.rng.choice(options.get(self.task.task_name, ["not sure"]))

    def _sample_plan(self) -> List[TurnScenario]:
        if self.task.task_type == "storytelling":
            return self._sample_storytelling_plan()
        probs = self.cfg["user_behavior"]
        roll = self.rng.random()
        cumulative = 0.0
        scenario_name = "correct_first_try"
        for name, prob in probs.items():
            cumulative += float(prob)
            if roll <= cumulative:
                scenario_name = name
                break

        if scenario_name == "correct_first_try":
            return [
                TurnScenario(
                    label=scenario_name,
                    transcript=self._correct_answer(),
                    attention_state="engaged",
                    response_latency_s=self._latency(self.task.max_wait_seconds),
                    asr_confidence=0.97,
                    expected_feedback_type="praise",
                    expected_policy_decision="reinforce_and_continue",
                    completed_after_turn=True,
                )
            ]

        if scenario_name == "incorrect_then_recover":
            return [
                TurnScenario(
                    label=scenario_name,
                    transcript=self._incorrect_answer(),
                    attention_state="engaged",
                    response_latency_s=self._latency(self.task.max_wait_seconds),
                    asr_confidence=0.92,
                    expected_feedback_type="hint",
                    expected_policy_decision="hint_then_retry",
                ),
                TurnScenario(
                    label=scenario_name,
                    transcript=self._correct_answer(),
                    attention_state="engaged",
                    response_latency_s=self._latency(self.task.max_wait_seconds),
                    asr_confidence=0.96,
                    expected_feedback_type="praise",
                    expected_policy_decision="reinforce_and_continue",
                    completed_after_turn=True,
                ),
            ]

        if scenario_name == "incorrect_persistent":
            return [
                TurnScenario(
                    label=scenario_name,
                    transcript=self._incorrect_answer(),
                    attention_state="engaged",
                    response_latency_s=self._latency(self.task.max_wait_seconds),
                    asr_confidence=0.90,
                    expected_feedback_type="hint",
                    expected_policy_decision="hint_then_retry",
                ),
                TurnScenario(
                    label=scenario_name,
                    transcript=self._incorrect_answer(),
                    attention_state="engaged",
                    response_latency_s=self._latency(self.task.max_wait_seconds),
                    asr_confidence=0.88,
                    expected_feedback_type="hint",
                    expected_policy_decision="hint_then_retry",
                    end_session_after_turn=True,
                ),
            ]

        if scenario_name == "no_response_recover":
            return [
                TurnScenario(
                    label=scenario_name,
                    transcript="",
                    attention_state="disengaged",
                    response_latency_s=round(float(self.task.max_wait_seconds), 2),
                    asr_confidence=0.0,
                    expected_feedback_type="reengage",
                    expected_policy_decision="engagement_recovery",
                ),
                TurnScenario(
                    label=scenario_name,
                    transcript=self._correct_answer(),
                    attention_state="engaged",
                    response_latency_s=self._latency(self.task.max_wait_seconds),
                    asr_confidence=0.94,
                    expected_feedback_type="praise",
                    expected_policy_decision="reinforce_and_continue",
                    completed_after_turn=True,
                ),
            ]

        if scenario_name == "no_response_dropout":
            return [
                TurnScenario(
                    label=scenario_name,
                    transcript="",
                    attention_state="disengaged",
                    response_latency_s=round(float(self.task.max_wait_seconds), 2),
                    asr_confidence=0.0,
                    expected_feedback_type="reengage",
                    expected_policy_decision="engagement_recovery",
                    force_asr_dropout=True,
                ),
                TurnScenario(
                    label=scenario_name,
                    transcript="",
                    attention_state="disengaged",
                    response_latency_s=round(float(self.task.max_wait_seconds), 2),
                    asr_confidence=0.0,
                    expected_feedback_type="reengage",
                    expected_policy_decision="engagement_recovery",
                    end_session_after_turn=True,
                    force_asr_dropout=True,
                ),
            ]

        return [
            TurnScenario(
                label="ambiguous_then_recover",
                transcript=self._ambiguous_answer(),
                attention_state="engaged",
                response_latency_s=self._latency(self.task.max_wait_seconds),
                asr_confidence=0.68,
                expected_feedback_type="hint",
                expected_policy_decision="hint_then_retry",
            ),
            TurnScenario(
                label="ambiguous_then_recover",
                transcript=self._correct_answer(),
                attention_state="engaged",
                response_latency_s=self._latency(self.task.max_wait_seconds),
                asr_confidence=0.96,
                expected_feedback_type="praise",
                expected_policy_decision="reinforce_and_continue",
                completed_after_turn=True,
            ),
        ]
        
    def _sample_storytelling_plan(self) -> List[TurnScenario]:
        dropout_rate = {
            "scripted": 0.00,
            "constrained_llm": 0.02,
            "unconstrained_llm": 0.06,
        }.get(self.condition, 0.04)

        if self.rng.random() < dropout_rate:
            return [
                TurnScenario(
                    label="story_dropout",
                    transcript="",
                    attention_state="disengaged",
                    response_latency_s=float(self.task.max_wait_seconds),
                    asr_confidence=0.0,
                    expected_feedback_type="reengage",
                    expected_policy_decision="engagement_recovery",
                    force_asr_dropout=True,
                ),
                TurnScenario(
                    label="story_dropout",
                    transcript="",
                    attention_state="disengaged",
                    response_latency_s=float(self.task.max_wait_seconds),
                    asr_confidence=0.0,
                    expected_feedback_type="reengage",
                    expected_policy_decision="engagement_recovery",
                    force_asr_dropout=True,
                    end_session_after_turn=True,
                ),
            ]
        
        possible_contributions = [
            "Pip sees a friendly dog",
            "Pip finds a red ball",
            "Pip meets a new friend",
            "Pip sees a bird in a tree",
        ]

        possible_emotions = [
            "happy",
            "excited",
            "nervous",
            "surprised",
        ]

        possible_actions = [
            "Pip should say hello",
            "Pip can share the ball",
            "Pip should help",
            "Pip can play gently",
        ]

        return [
            TurnScenario(
                label="story_contribution",
                transcript=self.rng.choice(possible_contributions),
                attention_state="engaged",
                response_latency_s=self._latency(self.task.max_wait_seconds),
                asr_confidence=0.94,
                expected_feedback_type="praise",
                expected_policy_decision="story_continue",
            ),
            TurnScenario(
                label="emotion_identification",
                transcript=self.rng.choice(possible_emotions),
                attention_state="engaged",
                response_latency_s=self._latency(self.task.max_wait_seconds),
                asr_confidence=0.92,
                expected_feedback_type="praise",
                expected_policy_decision="story_continue",
            ),
            TurnScenario(
                label="next_action_choice",
                transcript=self.rng.choice(possible_actions),
                attention_state="engaged",
                response_latency_s=self._latency(self.task.max_wait_seconds),
                asr_confidence=0.93,
                expected_feedback_type="praise",
                expected_policy_decision="story_continue",
            ),
            TurnScenario(
                label="story_detail",
                transcript="Pip asks the friend to play",
                attention_state="engaged",
                response_latency_s=self._latency(self.task.max_wait_seconds),
                asr_confidence=0.92,
                expected_feedback_type="praise",
                expected_policy_decision="story_continue",
            ),
            TurnScenario(
                label="story_resolution",
                transcript="They share and feel happy",
                attention_state="engaged",
                response_latency_s=self._latency(self.task.max_wait_seconds),
                asr_confidence=0.91,
                expected_feedback_type="praise",
                expected_policy_decision="story_continue",
            ),
            TurnScenario(
                label="story_wrapup",
                transcript="The end",
                attention_state="engaged",
                response_latency_s=self._latency(self.task.max_wait_seconds),
                asr_confidence=0.95,
                expected_feedback_type="praise",
                expected_policy_decision="story_conclude",
                completed_after_turn=True,
            ),
        ]

    def get_turn(self, turn_index: int, scripted_response: Optional[str] = None) -> TurnScenario:
        if scripted_response is not None:
            normalized = scripted_response.strip().lower()
            is_correct = normalized in [x.lower() for x in self.task.expected_answers]
            return TurnScenario(
                label="scripted_response",
                transcript=scripted_response,
                attention_state="engaged" if normalized else "disengaged",
                response_latency_s=self._latency(self.task.max_wait_seconds),
                asr_confidence=0.95 if normalized else 0.0,
                expected_feedback_type="praise" if is_correct else ("reengage" if not normalized else "hint"),
                expected_policy_decision=(
                    "reinforce_and_continue" if is_correct else ("engagement_recovery" if not normalized else "hint_then_retry")
                ),
                completed_after_turn=is_correct,
                end_session_after_turn=(not is_correct and turn_index >= self.task.max_retries),
            )
        idx = min(turn_index - 1, len(self.plan) - 1)
        return self.plan[idx]