from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from pepper_llm_mvp.analytics.logger import SessionLogger
from pepper_llm_mvp.config.loader import ConfigLoader
from pepper_llm_mvp.interaction.task_loader import TaskLoader
from pepper_llm_mvp.perception.asr import MockASR
from pepper_llm_mvp.reasoning.action_mapper import ActionMapper
from pepper_llm_mvp.reasoning.constraint_validator import ConstraintValidator
from pepper_llm_mvp.reasoning.policy_engine import PolicyEngine
from pepper_llm_mvp.reasoning.schemas import PolicyInput
from pepper_llm_mvp.robot.pepper_client import PepperClient
from pepper_llm_mvp.robot.speech import SpeechController
from pepper_llm_mvp.simulation.scenario_generator import ScenarioGenerator
from pepper_llm_mvp.types import (
    SessionRecord,
    SessionSummary,
    TurnRecord,
    TurnTimestamps,
    new_session_id,
    utc_now_iso,
)


class SessionManager:
    def __init__(
        self,
        project_root: str | Path,
        condition: str = "constrained_llm",
        scripted_response: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.project_root = Path(project_root)
        self.config_loader = ConfigLoader(self.project_root)
        self.app_cfg = self.config_loader.load_app()
        self.llm_cfg = self.config_loader.load_llm()["llm"]
        self.robot_cfg = self.config_loader.load_robot()["robot"]
        self.seed = int(seed if seed is not None else self.app_cfg.get("random_seed", 7))
        self.sim_cfg = self.app_cfg.get("simulation", {})

        self.task_loader = TaskLoader(self.project_root)
        self.logger = SessionLogger(self.project_root / self.app_cfg["sessions_dir"])
        self.robot = PepperClient(use_mock=self.robot_cfg.get("use_mock", True))
        self.robot.connect(self.robot_cfg.get("ip", "127.0.0.1"), int(self.robot_cfg.get("port", 9559)))
        self.speech = SpeechController(
            self.robot,
            rate=int(self.robot_cfg.get("speech_rate", 90)),
            volume=float(self.robot_cfg.get("volume", 0.7)),
            simulation_cfg=self.sim_cfg,
            seed=self.seed + 1,
        )
        self.asr = MockASR(scripted_response=scripted_response, seed=self.seed + 2)
        self.policy_engine = PolicyEngine(
            condition=condition,
            simulation_cfg=self.sim_cfg,
            llm_cfg=self.llm_cfg,
            seed=self.seed + 3,
        )        
        generation_cfg = self.llm_cfg.get("generation", {})
        self.validator = ConstraintValidator(
            forbidden_terms=self.llm_cfg["forbidden_terms"],
            allowed_feedback_types=self.llm_cfg["allowed_feedback_types"],
            max_sentences=int(generation_cfg.get("max_sentences", self.llm_cfg.get("max_sentences", 2))),
            max_words=int(generation_cfg.get("max_words", 18)),
            condition=condition,
            simulation_cfg=self.sim_cfg,
            seed=self.seed + 4,
        )
        self.mapper = ActionMapper()
        self.condition = condition
        self.scripted_response = scripted_response

    def _estimate_real_world_turn_duration(self, turn: TurnRecord) -> float:
        timing_cfg = self.sim_cfg.get("real_world_timing", {})

        prompt_speech = float(timing_cfg.get("prompt_speech_base_s", 3.5))
        stimulus_display = float(timing_cfg.get("stimulus_display_s", 2.0))
        gesture_overhead = float(timing_cfg.get("gesture_overhead_s", 1.2))
        tablet_update = float(timing_cfg.get("tablet_update_s", 0.8))
        inter_turn_pause = float(timing_cfg.get("inter_turn_pause_s", 1.5))

        robot_response = turn.policy_output.get("utterance", "")
        response_words = len(robot_response.split())

        robot_response_s = max(
            float(timing_cfg.get("robot_response_min_s", 3.0)),
            response_words * float(timing_cfg.get("robot_response_seconds_per_word", 0.35)),
        )
        robot_response_s = min(
            robot_response_s,
            float(timing_cfg.get("robot_response_max_s", 8.0)),
        )

        system_processing_s = (
            (turn.timestamps.asr_end - turn.timestamps.asr_start)
            + (turn.timestamps.llm_end - turn.timestamps.llm_start)
            + (turn.timestamps.execution_end - turn.timestamps.execution_start)
        )

        return (
            prompt_speech
            + stimulus_display
            + turn.response_latency_s
            + system_processing_s
            + robot_response_s
            + gesture_overhead
            + tablet_update
            + inter_turn_pause
        )

    def run_task(self, task_name: str) -> Path:
        task = self.task_loader.load_task(task_name)
        scenario_gen = ScenarioGenerator(self.sim_cfg, task=task, seed=self.seed + 5, condition=self.condition)
        
        session = SessionRecord(
            session_id=new_session_id(),
            condition=self.condition,
            task_name=task.task_name,
            start_time=utc_now_iso(),
        )

        session_wall_start = time.perf_counter()
        self.robot.set_posture(self.robot_cfg.get("default_posture", "StandInit"))
        self.speech.speak(task.intro_text)
        if task.stimulus_type != "none":
            self.robot.display(task.stimulus_path)
            
        story_context = ""
        story_phase = "story_contribution"
        max_story_turns = 1

        if task.task_type == "storytelling" and task.story:
            story_context = task.story.get("opening_context", "")
            max_story_turns = int(task.story.get("max_story_turns", 4))

        completed = False
        breakdown = False
        responded_turns = 0
        simulated_total_latency = 0.0
        if task.task_type == "storytelling":
            if self.condition == "scripted":
                max_turns = 4
            elif self.condition == "unconstrained_llm":
                max_turns = 6
            else:
                max_turns = 7
        else:
            max_turns = max(1, int(task.max_retries) + 1)
        asr_dropout_rate = float(self.sim_cfg.get("failure", {}).get("asr_dropout_rate", 0.0))

        next_story_prompt = task.prompt_text
        for turn_index in range(1, max_turns + 1):
            scenario = scenario_gen.get_turn(turn_index, scripted_response=self.scripted_response)

            prompt_start = time.perf_counter()
            if task.task_type == "storytelling":
                prompt_text = next_story_prompt
            else:
                if turn_index == 1:
                    prompt_text = task.prompt_text
                else:
                    prompt_text = task.prompt_text if session.turns[-1].policy_output["feedback_type"] == "reengage" else task.hint_text

            perception, asr_start, asr_end = self.asr.listen_and_transcribe(
                task,
                scenario=scenario,
                asr_dropout_rate=asr_dropout_rate,
            )
            if perception.transcript_raw.strip():
                responded_turns += 1

            inp = PolicyInput(
                task_name=task.task_name,
                prompt_text=prompt_text,
                transcript=perception.transcript_raw,
                normalized_response=perception.transcript_normalized,
                expected_answers=task.expected_answers,
                response_latency_s=perception.response_latency_s,
                attention_state=perception.attention_state,
                success_text=task.success_text,
                hint_text=task.hint_text,
                reengage_text=task.reengage_text,
                task_type=task.task_type,
                turn_index=turn_index,
                story_context=story_context,
                story_goal=task.story.get("learning_goal", "") if task.story else "",
                story_phase=story_phase,
            )
            llm_output, llm_start, llm_end = self.policy_engine.decide(inp)
            raw_constraint_valid = self.validator.validate(llm_output)
            constraint_failure_reason = getattr(self.validator, "last_failure_reason", "")
            fallback_used = False
            if not raw_constraint_valid and self.condition != "unconstrained_llm":
                fallback_used = True
                llm_output.generation_source = "template"
                llm_output.policy_decision = "fallback_policy"
                llm_output.safety_notes = (
                    llm_output.safety_notes + f" | validation_failure={constraint_failure_reason}"
                    if constraint_failure_reason
                    else llm_output.safety_notes
                )                
                llm_output.story_status = "in_progress"

                if task.task_type == "storytelling":
                    if perception.transcript_raw.strip():
                        if story_phase == "story_contribution":
                            next_prompt = task.story.get("emotion_prompt", "How do you think Pip feels?")
                            llm_output.feedback_type = "praise"
                            llm_output.utterance = f"Nice idea. Pip sees that in the story. {next_prompt}"
                            llm_output.next_action = "retry"
                            llm_output.template_reference = "story_emotion_fallback"
                            llm_output.next_prompt = next_prompt

                        elif story_phase == "emotion_identification":
                            next_prompt = task.story.get("action_prompt", "What should Pip do next?")
                            llm_output.feedback_type = "praise"
                            llm_output.utterance = f"Good feeling idea. Pip feels that way. {next_prompt}"
                            llm_output.next_action = "retry"
                            llm_output.template_reference = "story_action_fallback"
                            llm_output.next_prompt = next_prompt

                        elif "the end" in perception.transcript_normalized or "done" in perception.transcript_normalized or "finished" in perception.transcript_normalized:
                            llm_output.feedback_type = "praise"
                            llm_output.utterance = "Great storytelling. Pip had a kind and friendly day."
                            llm_output.next_action = "continue"
                            llm_output.policy_decision = "story_conclude"
                            llm_output.template_reference = "story_closing_fallback"
                            llm_output.next_prompt = ""
                            llm_output.story_status = "complete"

                        else:
                            next_prompt = "What happens next?"
                            llm_output.feedback_type = "praise"
                            llm_output.utterance = f"Nice choice. Pip adds that to the story. {next_prompt}"
                            llm_output.next_action = "retry"
                            llm_output.template_reference = "story_progress_fallback"
                            llm_output.next_prompt = next_prompt

                    else:
                        next_prompt = "What could happen next?"
                        llm_output.feedback_type = "reengage"
                        llm_output.utterance = f"{task.reengage_text} {next_prompt}"
                        llm_output.next_action = "retry"
                        llm_output.template_reference = "story_reengage_fallback"
                        llm_output.next_prompt = next_prompt

                else:
                    if perception.transcript_raw.strip():
                        llm_output.feedback_type = "hint"
                        llm_output.utterance = task.hint_text
                        llm_output.next_action = "retry"
                        llm_output.template_reference = "hint_text"
                        llm_output.next_prompt = ""
                    else:
                        llm_output.feedback_type = "reengage"
                        llm_output.utterance = task.reengage_text
                        llm_output.next_action = "retry"
                        llm_output.template_reference = "reengage_text"
                        llm_output.next_prompt = ""
            if not raw_constraint_valid and self.condition == "unconstrained_llm":
                fallback_used = False

            constraint_valid = self.validator.validate(llm_output)

            execution_start, execution_end = self.speech.speak(llm_output.utterance)
            self.robot.gesture(self.mapper.map_feedback_to_gesture(llm_output))

            pipeline_success = not scenario.force_asr_dropout
            error_stage = "asr" if scenario.force_asr_dropout else None
            turn_breakdown = (not perception.transcript_raw.strip()) and turn_index >= max_turns

            turn = TurnRecord(
                turn_index=turn_index,
                timestamps=TurnTimestamps(
                    prompt_start=prompt_start,
                    asr_start=asr_start,
                    asr_end=asr_end,
                    llm_start=llm_start,
                    llm_end=llm_end,
                    execution_start=execution_start,
                    execution_end=execution_end,
                ),
                prompt_text=prompt_text,
                stimulus=task.stimulus_path,
                transcript_raw=perception.transcript_raw,
                transcript_normalized=perception.transcript_normalized,
                response_latency_s=perception.response_latency_s,
                attention_state=perception.attention_state,
                asr_confidence=perception.asr_confidence,
                policy_output=llm_output.__dict__,
                expected_feedback_type=scenario.expected_feedback_type,
                expected_policy_decision=scenario.expected_policy_decision,
                scenario_label=scenario.label,
                constraint_valid=constraint_valid,
                fallback_used=fallback_used,
                pipeline_success=pipeline_success,
                breakdown=turn_breakdown,
                error_stage=error_stage,
            )
            session.turns.append(turn)
            
            simulated_total_latency += (asr_end - asr_start) + (llm_end - llm_start) + (execution_end - execution_start)
            
            if task.task_type == "storytelling":
                story_context += f"\nRobot prompt {turn_index}: {prompt_text}"

                if perception.transcript_raw.strip():
                    story_context += f"\nUser contribution {turn_index}: {perception.transcript_raw.strip()}"

                story_context += f"\nRobot response {turn_index}: {llm_output.utterance.strip()}"

                if llm_output.story_status == "complete" or llm_output.policy_decision == "story_conclude":
                    completed = True
                    break

                if llm_output.next_prompt.strip():
                    next_story_prompt = llm_output.next_prompt.strip()
                else:
                    next_story_prompt = task.story.get("action_prompt", "What should happen next?") if task.story else "What should happen next?"

                if turn_index == 1:
                    story_phase = "emotion_identification"
                elif turn_index == 2:
                    story_phase = "next_action_choice"
                else:
                    story_phase = "story_progression"

            if task.task_type == "storytelling":
                if scenario.end_session_after_turn:
                    breakdown = True
                    completed = False
                    break
                if turn_index >= max_turns:
                    if self.condition == "scripted":
                        completed = True
                        breakdown = False
                    else:
                        completed = llm_output.policy_decision == "story_conclude" or llm_output.story_status == "complete"
                        breakdown = not completed
                    break
            else:
                if llm_output.next_action == "continue" or scenario.completed_after_turn:
                    completed = True
                    break
            if scenario.end_session_after_turn or turn_index >= max_turns:
                breakdown = True
                break

        if not completed and (breakdown or session.turns[-1].policy_output["next_action"] != "continue"):
            breakdown = True

        wall_duration_s = time.perf_counter() - session_wall_start
        session.session_summary = SessionSummary(
            task_completed=completed,
            turn_count=len(session.turns),
            avg_latency_s=(simulated_total_latency / len(session.turns)) if session.turns else 0.0,
            session_duration_s=sum(
                self._estimate_real_world_turn_duration(t)
                for t in session.turns
            ),
            participation_rate=(responded_turns / len(session.turns)) if session.turns else 0.0,
            interaction_breakdown=breakdown,
        )
        session.end_time = utc_now_iso()
        return self.logger.write_session(session)