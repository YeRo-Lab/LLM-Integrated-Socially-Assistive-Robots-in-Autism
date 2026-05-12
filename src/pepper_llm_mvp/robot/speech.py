from __future__ import annotations

import random
import time
from typing import Any, Dict

from pepper_llm_mvp.robot.pepper_client import PepperClient


class SpeechController:
    def __init__(
        self,
        robot: PepperClient,
        rate: int = 90,
        volume: float = 0.7,
        simulation_cfg: Dict[str, Any] | None = None,
        seed: int = 7,
    ):
        self.robot = robot
        self.rate = rate
        self.volume = volume
        self.cfg = simulation_cfg or {}
        self.rng = random.Random(seed)

    def speak(self, text: str) -> tuple[float, float]:
        start = time.perf_counter()
        self.robot.say(text)
        latency_cfg = self.cfg.get("latency", {})
        mean = float(latency_cfg.get("execution_mean_s", 0.45))
        std = float(latency_cfg.get("execution_std_s", 0.12))
        simulated = max(0.12, self.rng.gauss(mean + 0.003 * len(text), std))
        time.sleep(min(simulated, 1.6))
        end = time.perf_counter()
        return start, end