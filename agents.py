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
        raise e


def preprocessing_agent(state: State) -> State:
    try:
        logic.create_preprocess_spec(state)
        logic.execute_preprocess_spec(state)
    except Exception as e:
        state.errors.append(str(e))
        state.stage = "failed"
        
        
def training_agent(state: State) -> State:
    try:
        logic.initiate_training_process(state)
    except Exception as e:
        state.errors.append(str(e))
        state.stage = "failed"
        