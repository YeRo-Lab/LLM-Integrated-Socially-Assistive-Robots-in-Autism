from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from pepper_llm_mvp.reasoning.schemas import LLMOutput, PolicyInput


POLICY_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["correct", "incorrect", "unclear", "disengaged"],
        },
        "feedback_type": {
            "type": "string",
            "enum": ["praise", "hint", "reengage"],
        },
        "utterance": {
            "type": "string",
            "description": "The final child-facing robot response. Must be safe, short, simple, supportive, and task-relevant.",
        },
        "next_action": {
            "type": "string",
            "enum": ["continue", "retry"],
        },
        "policy_decision": {
            "type": "string",
            "enum": [
                "reinforce_and_continue",
                "hint_then_retry",
                "engagement_recovery",
                "clarify_then_retry",
                "story_continue",
                "story_conclude"
            ],
        },
        "generation_source": {
            "type": "string",
            "enum": ["generated", "template"],
        },
        "template_reference": {
            "type": "string",
            "description": "Name of fallback template if used, otherwise empty string.",
        },
        "safety_notes": {
            "type": "string",
            "description": "Brief internal note explaining why the response is safe and task-relevant. Five words or fewer.",
        },
        "next_prompt": {
            "type": "string",
            "description": "The exact question that should be asked at the start of the next turn. Empty if the story is complete."
        },
        "story_status": {
            "type": "string",
            "enum": ["in_progress", "complete"]
        },
    },
    "required": [
        "classification",
        "feedback_type",
        "utterance",
        "next_action",
        "policy_decision",
        "generation_source",
        "template_reference",
        "safety_notes",
        "next_prompt",
        "story_status",
    ],
}


class OpenAILLMClient:
    def __init__(
        self,
        model: str,
        max_output_tokens: int = 350,
        timeout_s: int = 20,
        generation_cfg: Dict[str, Any] | None = None,
        mode: str = "constrained_llm",
    ):
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or your environment."
            )

        self.client = OpenAI(timeout=timeout_s)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.generation_cfg = generation_cfg or {}
        self.mode = mode

    def build_input(self, inp: PolicyInput) -> List[Dict[str, str]]:
        max_words = int(self.generation_cfg.get("max_words", 18))
        max_sentences = int(self.generation_cfg.get("max_sentences", 2))
        vocabulary_level = self.generation_cfg.get(
            "vocabulary_level", "simple_child_friendly"
        )

        if self.mode == "unconstrained_llm":
            system_msg = """
You are a creative storytelling module for a social robot.

Your job:
1. Interpret the user's response.
2. Continue the story in an expressive and imaginative way.
3. Ask a follow-up question when the story should continue.
4. If the user clearly ends the story, wrap it up.

You are not required to keep responses short, predictable, or template-like.
You may use richer language, longer story continuations, and more creative elaboration.

For storytelling tasks:
- Use the user's latest contribution and story_context.
- Continue the story creatively.
- You may introduce new details if they seem interesting.
- If the user says "the end", "done", "finished", "that's all", or "end story", set story_status="complete", policy_decision="story_conclude", next_action="continue", and next_prompt="".
- If the story continues, ask one follow-up question and put that same question in next_prompt.

Return only the required JSON object.
""".strip()

        else:
            system_msg = f"""
You are the constrained reasoning and response-generation module for a socially assistive robot.

Your job has two parts:
1. Classify the user's response within the current task.
2. Generate the final child-facing robot utterance.

You must obey these constraints:
- Do not diagnose, score, evaluate clinical status, or mention autism severity.
- Do not say "wrong", "incorrect", "bad", or "failed".
- Keep the utterance supportive, calm, and predictable.
- Keep the utterance task-relevant.
- Use {vocabulary_level} language.
- Use no more than {max_sentences} sentence(s).
- Use no more than {max_words} words.
- If the user is correct, briefly praise and optionally mention the observed cue.
- If the user is not correct, give a gentle hint without negative wording.
- If the user gives no response or seems disengaged, gently reengage.
- Do not introduce a new topic.

For storytelling tasks:
- Use the user's latest contribution and the story_context to continue the story.
- Preserve turn taking. Do not complete the whole story unless the user indicates the story is over.
- The utterance must follow this pattern: brief praise + one short story sentence + one short question.
- Do not exceed the configured word limit. Prefer short phrases over detailed narration.
- For turn 1 of storytelling, if the user gives a valid object or event, use feedback_type="praise" and policy_decision="story_continue".
- End every in-progress storytelling utterance with exactly one question.
- The final question in the utterance must exactly match next_prompt.
- The system will use next_prompt as the robot's prompt at the start of the next turn.
- If the user says "the end", "done", "finished", "that's all", or "end story", wrap up the story warmly.
- When wrapping up, set story_status="complete", policy_decision="story_conclude", next_action="continue", and next_prompt="".
- If story_status="complete", do not ask another question.

Return only the required JSON object.
""".strip()

        user_payload = {
            "task_name": inp.task_name,
            "task_type": inp.task_type,
            "turn_index": inp.turn_index,
            "story_context": inp.story_context,
            "story_goal": inp.story_goal,
            "story_phase": inp.story_phase,
            "robot_prompt": inp.prompt_text,
            "user_transcript": inp.transcript,
            "normalized_response": inp.normalized_response,
            "expected_answers": inp.expected_answers,
            "response_latency_s": inp.response_latency_s,
            "attention_state": inp.attention_state,
            "fallback_templates": {
                "success_text": inp.success_text,
                "hint_text": inp.hint_text,
                "reengage_text": inp.reengage_text,
            },
            "story_completion_cues": ["the end", "done", "finished", "that's all", "end story"],
            "next_prompt_instruction": (
                "For storytelling tasks, end the utterance with exactly one question. "
                "Copy that exact question into next_prompt. "
                "If the story is complete, leave next_prompt empty."
            ),
            "decision_rules": [
                "For non-storytelling tasks only: if normalized_response matches expected_answers, use classification='correct', feedback_type='praise', next_action='continue', policy_decision='reinforce_and_continue'.",
                "For non-storytelling tasks only: if transcript is present but does not match expected_answers, use feedback_type='hint', next_action='retry', policy_decision='hint_then_retry'.",
                "For any task: if transcript is empty or attention_state='disengaged', use classification='disengaged', feedback_type='reengage', next_action='retry', policy_decision='engagement_recovery'.",
                "For storytelling tasks: do not grade the user against expected_answers. Treat any non-empty relevant contribution as acceptable.",
                "For storytelling tasks: use feedback_type='praise', next_action='retry', policy_decision='story_continue', and story_status='in_progress' unless the user clearly ends the story.",
                "For storytelling tasks: if the user says the end, done, finished, that's all, or end story, use feedback_type='praise', next_action='continue', policy_decision='story_conclude', and story_status='complete'."
            ],
            "generation_instruction": (
                "Generate a fresh utterance that follows the constraints. "
                "Do not simply copy a fallback template unless generation would be unsafe."
            ),
        }

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_payload)},
        ]

    def decide(self, inp: PolicyInput) -> tuple[LLMOutput, float, float]:
        start = time.perf_counter()

        params = {
            "model": self.model,
            "input": self.build_input(inp),
            "max_output_tokens": self.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "robot_policy_output",
                    "strict": True,
                    "schema": POLICY_OUTPUT_SCHEMA,
                }
            },
        }

        response = self.client.responses.create(**params)

        end = time.perf_counter()

        raw_text = response.output_text or ""

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            debug_path = Path("debug_last_openai_response.txt")
            debug_path.write_text(raw_text, encoding="utf-8")

            data = {
                "classification": "uncertain",
                "feedback_type": "reengage",
                "next_action": "retry",
                "utterance": "Let's try that again together.",
                "policy_decision": "fallback_policy",
                "story_status": "in_progress",
                "rationale": "Invalid or incomplete JSON returned by LLM.",
                "generation_source": "fallback",
            }

        return LLMOutput(
            classification=data["classification"],
            feedback_type=data["feedback_type"],
            utterance=data["utterance"],
            next_action=data["next_action"],
            policy_decision=data["policy_decision"],
            generation_source=data["generation_source"],
            template_reference=data["template_reference"],
            safety_notes=data["safety_notes"],
            next_prompt=data["next_prompt"],
            story_status=data["story_status"],
        ), start, end