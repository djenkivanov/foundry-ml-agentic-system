from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, TypedDict, Union
import pandas as pd
from preprocess_spec import PreprocessSpec

Stage = Literal["plan", "preprocess", "train", "success", "failed"]
Task = Literal["regression", "classification"]

@dataclass
class State:
    prompt: str
    raw_train_ds: pd.DataFrame
    fe_train_ds: pd.DataFrame = None
    train_ds_path: str = ""
    
    x_train: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    
    target: str = ""
    task: Task = ""
    insights: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    preprocess_spec: PreprocessSpec = field(default_factory=dict)
    training_plan: Dict[str, Any] = field(default_factory=dict)
    model: Any = None
    best_model_scores: Dict[str, Any] = field(default_factory=dict)
    all_model_scores: Dict[str, Any] = field(default_factory=dict)
    
    stage: Stage = "plan"
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)