from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
import pandas as pd

Stage = Literal["plan", "preprocess", "train", "evaluate", "complete", "failed"]
Task = Literal["regression", "classification"]

@dataclass
class State:
    prompt: str
    train_ds: pd.DataFrame
    test_ds: pd.DataFrame
    
    target: str = ""
    task: Task = ""
    insights: Dict[str, Any] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=list)
    
    stage: Stage = "plan"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)