import pandas as pd
import logic
from custom_state import State

def planner_agent(state: State, reasoning_stream=None, plan_stream=None):
    try:
        logic.get_data_insight(state)
        logic.create_initial_plan(
            state=state,
            reasoning_stream=reasoning_stream,
            plan_stream=plan_stream
        )
    except Exception as e:
        state.errors.append(str(e))
        state.stage = "failed"


def diagnostician_agent(state: State) -> State:
    
    return state
