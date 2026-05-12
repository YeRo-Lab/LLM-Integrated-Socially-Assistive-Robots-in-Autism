from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Any
from pepper_llm_mvp.types import SessionRecord


class SessionLogger:
    def __init__(self, sessions_dir: str | Path):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def write_session(self, session: SessionRecord) -> Path:
        output_path = self.sessions_dir / f"{session.session_id}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2)
        return output_path

    @staticmethod
    def read_session(path: str | Path) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
