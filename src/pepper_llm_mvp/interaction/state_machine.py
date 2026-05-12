from __future__ import annotations

from enum import Enum


class InteractionState(str, Enum):
    INTRO = "INTRO"
    STIMULUS = "STIMULUS"
    WAIT_FOR_RESPONSE = "WAIT_FOR_RESPONSE"
    EVALUATE = "EVALUATE"
    FEEDBACK = "FEEDBACK"
    TRANSITION = "TRANSITION"
    DONE = "DONE"


STATE_FLOW = {
    InteractionState.INTRO: InteractionState.STIMULUS,
    InteractionState.STIMULUS: InteractionState.WAIT_FOR_RESPONSE,
    InteractionState.WAIT_FOR_RESPONSE: InteractionState.EVALUATE,
    InteractionState.EVALUATE: InteractionState.FEEDBACK,
    InteractionState.FEEDBACK: InteractionState.TRANSITION,
    InteractionState.TRANSITION: InteractionState.DONE,
}


def next_state(state: InteractionState) -> InteractionState:
    return STATE_FLOW.get(state, InteractionState.DONE)
