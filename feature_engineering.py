from typing import List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def init_feature_engineering(state):
    refine_and_execute_operations(state)
    ops = state.preprocess_spec.get("feature_engineering", [])
    drop_cols = state.preprocess_spec.get("drop_columns", [])
    return FeatureEngineeringTransformer(ops, drop_cols)


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


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, operations: List[dict], drop_columns: List[str] = None):
        self.operations = operations or []
        self.drop_columns = drop_columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        executed_operations = {}
        for operation in self.operations:
            for _, params in operation.items():
                df = execute_fe_code(df, **params, executed_operations=executed_operations)

        cols_to_drop = [col for col in self.drop_columns if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df
