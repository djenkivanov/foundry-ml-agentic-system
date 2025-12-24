from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, TypedDict, Union
import pandas as pd
from preprocess_spec import PreprocessSpec

Stage = Literal["plan", "preprocess", "train", "evaluate", "complete", "failed"]
Task = Literal["regression", "classification"]

@dataclass
class State:
    prompt: str
    train_ds: pd.DataFrame
    test_ds: pd.DataFrame
    
    x_train: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    x_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.Series] = None
    
    target: str = ""
    task: Task = ""
    insights: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    preprocess_spec: PreprocessSpec = field(default_factory=dict)
    training_plan: Dict[str, Any] = field(default_factory=dict)
    
    
    stage: Stage = "plan"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)