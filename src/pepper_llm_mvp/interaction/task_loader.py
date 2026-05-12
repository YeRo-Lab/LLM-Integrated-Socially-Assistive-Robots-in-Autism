from __future__ import annotations

from pathlib import Path
import yaml
from pepper_llm_mvp.types import TaskConfig


class TaskLoader:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load_task(self, task_name: str) -> TaskConfig:
        path = self.root / "config" / "tasks" / f"{task_name}.yaml"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return TaskConfig(**data)
