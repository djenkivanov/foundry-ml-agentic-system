from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import logic

Stage = Literal["plan", "train", "diagnose", "evaluate", "complete", "failed"]
Task = Literal["regression", "classification"]

@dataclass
class State:
    prompt: str
    train_ds: pd.DataFrame
    test_ds: pd.DataFrame
    target: str
    Task: Task
    
    stage: Stage = "plan"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)


def planner_agent(state: State) -> State:
    plan = logic.create_initial_plan(
        prompt=state.prompt,
        train_ds_diagnostics=state.train_ds,
        test_ds_diagnostics=state.test_ds
    )
    state.plan = plan
    state.stage = "train"
    return state


def diagnostician_agent(state: State) -> State:
    
    return state
