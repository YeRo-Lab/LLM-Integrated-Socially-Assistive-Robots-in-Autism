from __future__ import annotations

from pathlib import Path

from pepper_llm_mvp.config.loader import ConfigLoader
from pepper_llm_mvp.reasoning.policy_engine import PolicyEngine
from pepper_llm_mvp.reasoning.schemas import PolicyInput


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = ConfigLoader(project_root)
    app_cfg = cfg.load_app()
    llm_cfg = cfg.load_llm()["llm"]

    engine = PolicyEngine(
        condition="constrained_llm",
        simulation_cfg=app_cfg.get("simulation", {}),
        llm_cfg=llm_cfg,
        seed=123,
    )

    inp = PolicyInput(
        task_name="emotion_recognition",
        prompt_text="How is this person feeling?",
        transcript="happy",
        normalized_response="happy",
        expected_answers=["happy", "joyful"],
        response_latency_s=1.4,
        attention_state="engaged",
        success_text="Nice job. They do look happy.",
        hint_text="Look at their mouth. Are they smiling or frowning?",
        reengage_text="That's okay, let's try this one together.",
    )

    out, start, end = engine.decide(inp)
    print(out)
    print(f"LLM latency: {end - start:.3f}s")


if __name__ == "__main__":
    main()