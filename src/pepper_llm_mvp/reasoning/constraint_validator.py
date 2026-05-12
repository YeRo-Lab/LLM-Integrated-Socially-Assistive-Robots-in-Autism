from __future__ import annotations

import random
import re
from typing import Any, Dict

from pepper_llm_mvp.reasoning.schemas import LLMOutput


class ConstraintValidator:
    def __init__(
        self,
        forbidden_terms: list[str],
        allowed_feedback_types: list[str],
        max_sentences: int = 2,
        condition: str = "constrained_llm",
        simulation_cfg: Dict[str, Any] | None = None,
        seed: int = 7,
        max_words: int = 18,
    ):
        self.forbidden_terms = [x.lower() for x in forbidden_terms]
        self.allowed_feedback_types = set(allowed_feedback_types)
        self.max_sentences = max_sentences
        self.max_words = max_words
        self.condition = condition
        self.cfg = simulation_cfg or {}
        self.rng = random.Random(seed)
        self.last_failure_reason = ""

    def _violation_rate(self) -> float:
        failure_cfg = self.cfg.get("failure", {})
        if self.condition == "unconstrained_llm":
            return float(failure_cfg.get("unconstrained_constraint_violation_rate", 0.0))
        if self.condition == "constrained_llm":
            return float(failure_cfg.get("constrained_constraint_violation_rate", 0.0))
        return 0.0

    def _sentence_count(self, text: str) -> int:
        pieces = [p for p in re.split(r"[.!?]+", text.strip()) if p.strip()]
        return max(1, len(pieces))

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\b\w+\b", text))

    def validate(self, output: LLMOutput) -> bool:
        self.last_failure_reason = ""
        text = output.utterance.strip()
        lower = text.lower()

        if output.feedback_type not in self.allowed_feedback_types:
            self.last_failure_reason = "invalid_feedback_type"
            return False

        if not text:
            self.last_failure_reason = "empty_utterance"
            return False

        if any(term in lower for term in self.forbidden_terms):
            self.last_failure_reason = "forbidden_term"
            return False

        if self._sentence_count(text) > self.max_sentences:
            self.last_failure_reason = "too_many_sentences"
            return False

        if self._word_count(text) > self.max_words:
            self.last_failure_reason = "too_many_words"
            return False

        if output.generation_source not in {"generated", "template"}:
            self.last_failure_reason = "invalid_generation_source"
            return False

        if self.rng.random() < self._violation_rate():
            self.last_failure_reason = "simulated_constraint_violation"
            return False

        return True