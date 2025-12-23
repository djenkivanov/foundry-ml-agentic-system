from typing import List, Dict, Any, Optional, Literal, TypedDict, Union






NumericColumns = Union[Literal["auto"], List[str]]

# SimpleImputer with 'strategy' parameter
NumericImputer = Literal[
    "mean",
    "median",
    "most_frequent",
    "constant",
    None
]

# StandardScaler, MinMaxScaler, RobustScaler
NumericScaler = Literal[
    "standard",
    "minmax",
    "robust",
    None
]

CategoricalColumns = Union[Literal["auto"], List[str]]

# SimpleImputer with 'strategy' parameter
CategoricalImputer = Literal[
    "most_frequent",
    "constant",
    None
]

# OneHotEncoder, OrdinalEncoder
CategoricalEncoder = Literal[
    "onehot",
    "ordinal",
    None
]

FeatureEngineering = Union[List[Dict[str, Any]], Dict[str, Any]]

class NumericSpec(TypedDict, total=False):
    columns: NumericColumns
    imputer: NumericImputer
    scaler: NumericScaler
    
class CategoricalSpec(TypedDict, total=False):
    columns: CategoricalColumns
    imputer: CategoricalImputer
    encoder: CategoricalEncoder

class FeatureEngineeringSpec(TypedDict, total=False):
    feature_engineering: FeatureEngineering

class PreprocessSpec(TypedDict, total=False):
    numeric: NumericSpec
    categorical: CategoricalSpec
    feature_engineering: FeatureEngineeringSpec

def get_preprocess_spec():
    return """
    {
        "drop_columns": ["list", "of", "columns", "to", "drop"],
        "numeric": {
            "columns": "auto" | ["list", "of", "numeric", "columns"],
            "imputer": "mean" | "median" | "most_frequent" | "constant" | null,
            "scaler": "standard" | "minmax" | "robust" | null
        },
        "categorical": {
            "columns": "auto" | ["list", "of", "categorical", "columns"],
            "imputer": "most_frequent" | "constant" | null,
            "encoder": "onehot" | "ordinal" | null
        },
        "feature_engineering": {
            detailed instructions for feature engineering steps can be added here
        }
    }
"""