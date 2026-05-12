from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigLoader:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load_yaml(self, relative_path: str) -> Dict[str, Any]:
        path = self.root / relative_path
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_app(self) -> Dict[str, Any]:
        return self.load_yaml("config/app.yaml")

    def load_robot(self) -> Dict[str, Any]:
        return self.load_yaml("config/robot.yaml")

    def load_llm(self) -> Dict[str, Any]:
        return self.load_yaml("config/llm.yaml")
