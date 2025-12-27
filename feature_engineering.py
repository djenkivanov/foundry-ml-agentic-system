from typing import List, Optional
import re
import pandas as pd
from sklearn import preprocessing
import logic


def init_feature_engineering(state):
    refine_and_execute_operations(state)


def refine_and_execute_operations(state):
    list_of_ops = state.preprocess_spec.get("feature_engineering", [])
    feature_engineered_df_train, executed_operations = execute_operations(state, list_of_ops)
    state.fe_train_ds = feature_engineered_df_train
    state.trace.append({"feature_engineering_operations": executed_operations})

def execute_operations(state, operations: List[dict]) -> dict:
    executed_operations = {}
    df = state.fe_train_ds.copy()
    for operation in operations:
        for _, params in operation.items():
            df = execute_fe_code(df, **params, executed_operations=executed_operations)
            
    return df, executed_operations


def execute_fe_code(df, new_column, expression, executed_operations):
    df[new_column] = eval(expression)
    executed_operations[f'New column "{new_column}"'] = expression
    return df
