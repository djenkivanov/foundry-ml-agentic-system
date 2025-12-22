from typing import List, Optional
import re
import pandas as pd
from sklearn import preprocessing


def execute_operations(df, operations: List[dict]) -> dict:
    for operation in operations:
        for op_name, params in operation.items():
            if op_name in available_operations:
                df = available_operations[op_name](df, **params)
            else:
                raise ValueError(f"Unsupported operation: {op_name}")
    return df


def str_extractor(s: str, patterns: List[str]) -> Optional[str]:
    pattern = re.compile("|".join(patterns))
    match = pattern.search(s)
    return match.group(0) if match else None


def derive(df, new_column, expression):
    df[new_column] = df.eval(expression)
    return df


def extract(df, source_column, new_column, match):
    df[new_column] = df[source_column].apply(lambda x: str_extractor(str(x), match))
    return df


def encode(df, source_column, method):
    if method == "onehot":
        encoder = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cols = encoder.fit_transform(df[[source_column]])
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out([source_column]))
        df = pd.concat([df.drop(columns=[source_column]), encoded_df], axis=1)
    elif method == "ordinal":
        encoder = preprocessing.OrdinalEncoder()
        df[source_column] = encoder.fit_transform(df[[source_column]])
    return df


available_operations = {
    "derive": derive,
    "extract": extract,
    "encode": encode,
}

def get_valid_feature_engineering_methods():
    return """
    {
        "derive": "Derive a new feature using an expression involving existing columns.",
        "extract": "Extract parts of a feature based on matching patterns.",
        "encode": "Encode a categorical feature using a specified encoding method.",
    }
    """

def get_feature_engineering_spec():
    return """
    feature_engineering": [
        {
            "derive": {
                "new_column": "new_feature_name",
                "expression": "existing_column1 + existing_column2"
            }
        },
        {
            "extract": {
                "source_column": "date_column",
                "new_column": "year", /* None if overwriting and simply wish to delete matched parts */
                "match": ["2020", "2021", "2022", "2023", "2024"]
            }
        },  
        {
            "encode": {
                "source_column": "year",
                "method": "onehot"
            }
        }      
    ]
    """