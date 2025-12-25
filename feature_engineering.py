from typing import List, Optional
import re
import pandas as pd
from sklearn import preprocessing
import logic


def init_feature_engineering(state):
    refine_and_execute_operations(state)


def refine_and_execute_operations(state):
    list_of_ops = state.preprocess_spec.get("feature_engineering", [])
    feature_engineered_df_train = execute_operations(state.fe_train_ds, list_of_ops)
    state.fe_train_ds = feature_engineered_df_train


def execute_operations(df, operations: List[dict]) -> dict:
    for operation in operations:
        for _, params in operation.items():
            df = execute_fe_code(df, **params)
            
    return df


def execute_fe_code(df, new_column, expression):
    df[new_column] = eval(expression)
    return df
