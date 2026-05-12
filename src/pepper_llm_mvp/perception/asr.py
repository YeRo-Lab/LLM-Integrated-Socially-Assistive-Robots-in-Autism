from __future__ import annotations

import random
import time
from typing import Optional

from pepper_llm_mvp.simulation.scenario_generator import TurnScenario
from pepper_llm_mvp.types import PerceptionResult, TaskConfig


class MockASR:
    def __init__(self, scripted_response: Optional[str] = None, seed: int = 7):
        self.scripted_response = scripted_response
        self.rng = random.Random(seed)

    def listen_and_transcribe(
        self,
        task: TaskConfig,
        scenario: TurnScenario,
        asr_dropout_rate: float = 0.0,
    ) -> tuple[PerceptionResult, float, float]:
        asr_start = time.perf_counter()
        simulated_asr_compute = max(0.15, 0.25 + 0.08 * scenario.response_latency_s + self.rng.uniform(0.0, 0.15))
        time.sleep(min(simulated_asr_compute, 1.0))

        raw = scenario.transcript
        if scenario.force_asr_dropout or (raw and self.rng.random() < asr_dropout_rate):
            raw = ""

        norm = raw.strip().lower()
        attention = scenario.attention_state if raw or scenario.attention_state == "disengaged" else "engaged"
        confidence = scenario.asr_confidence if raw else 0.0
        asr_end = time.perf_counter()

        return PerceptionResult(
            transcript_raw=raw,
            transcript_normalized=norm,
            response_latency_s=scenario.response_latency_s,
            attention_state=attention,
            asr_confidence=confidence,
            scenario_label=scenario.label,
        ), asr_start, asr_end