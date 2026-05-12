from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class MockRobotAction:
    action_type: str
    payload: str


class PepperClient:
    """Local mock of Pepper/libqi client.

    Replace methods here with real NAOqi/libqi calls later.
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.connected = False
        self.actions: List[MockRobotAction] = []

    def connect(self, ip: str = "127.0.0.1", port: int = 9559) -> None:
        self.connected = True
        self.actions.append(MockRobotAction("connect", f"{ip}:{port}"))

    def set_posture(self, posture: str) -> None:
        self.actions.append(MockRobotAction("posture", posture))

    def display(self, stimulus_path: str) -> None:
        self.actions.append(MockRobotAction("display", stimulus_path))

    def gesture(self, gesture_name: str) -> None:
        self.actions.append(MockRobotAction("gesture", gesture_name))

    def say(self, text: str) -> None:
        self.actions.append(MockRobotAction("say", text))

    def reset_actions(self) -> None:
        self.actions.clear()
