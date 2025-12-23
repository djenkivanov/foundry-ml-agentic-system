from typing import List, Optional
import re
import pandas as pd
from sklearn import preprocessing
import logic


def init_feature_engineering(state):
    refine_and_execute_operations(state)


def refine_and_execute_operations(state):
    fe_spec = logic.get_refined_feature_engineering_spec(state)
    state.new_fe_spec = fe_spec
    list_of_ops = fe_spec.get("feature_engineering", [])
    feature_engineered_df_train = execute_operations(state.train_ds, list_of_ops)
    feature_engineered_df_test = execute_operations(state.test_ds, list_of_ops)
    state.train_ds = feature_engineered_df_train
    state.test_ds = feature_engineered_df_test
    state.new_fe_spec = fe_spec
    state.preprocess_spec["feature_engineering"] = fe_spec


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
    
if __name__ == "__main__":
    train_ds = pd.read_csv("titanic train.csv")
    test_ds = pd.read_csv("titanic test.csv")
    target = "Survived"
    task = "classification"
    
    preprocess_spec = {
    "drop_columns": [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ],
    "numeric": {
        "columns": [
        "Age",
        "Fare",
        "SibSp",
        "Parch",
        "FamilySize",
        "IsAlone"
        ],
        "imputer": "median",
        "scaler": "standard"
    },
    "categorical": {
        "columns": [
        "Sex",
        "Embarked",
        "Title",
        "Deck"
        ],
        "imputer": "most_frequent",
        "encoder": "onehot"
    },
    "feature_engineering": {
        "Title": "Parse Name to extract honorific (Mr, Mrs, Miss, Master, etc.); map all other rare titles to 'Other'",
        "Deck": "Fill missing Cabin values as 'U'; take first character of Cabin string",
        "FamilySize": "Compute as SibSp + Parch + 1",
        "IsAlone": "Set to 1 if FamilySize equals 1, else 0"
    }
    }
    
    plan = {
    "plan": {
        "task": "classification",
        "target": "Survived",
        "preprocess": {
        "drop_columns": [
            "PassengerId",
            "Name",
            "Ticket"
        ],
        "missing_values": {
            "Age": "Impute with median age stratified by Pclass and Sex; fallback to overall median",
            "Embarked": "Impute with mode ('S')",
            "Fare": "Impute with median fare of corresponding Pclass; fallback to overall median",
            "Cabin": "Extract deck letter; fill missing as 'U' (unknown)"
        },
        "feature_engineering": {
            "Title": "Extract title from Name and group rare titles into 'Other'",
            "Deck": "First character of Cabin",
            "FamilySize": "Compute SibSp + Parch + 1",
            "IsAlone": "1 if FamilySize == 1 else 0"
        },
        "categorical_encoding": {
            "Sex": "Binary encode (female=0, male=1)",
            "Embarked": "One-hot encode",
            "Title": "One-hot encode",
            "Deck": "One-hot encode",
            "IsAlone": "Already binary"
        },
        "scaling": {
            "numerical_features": [
            "Age",
            "Fare",
            "FamilySize"
            ],
            "method": "StandardScaler"
        }
        },
        "model_selection": {
        "candidates": [
            "LogisticRegression (baseline, interpretable)",
            "RandomForestClassifier (handles nonlinearity, missing values)",
            "XGBClassifier or LightGBM (gradient boosting, robust performance)"
        ],
        "selection_criteria": "Cross-validated F1-score and ROC AUC"
        },
        "training": {
        "train_validation_split": "StratifiedKFold with 5 folds on training data",
        "hyperparameter_tuning": {
            "method": "RandomizedSearchCV or GridSearchCV",
            "parameters": {
            "LogisticRegression": {
                "C": [
                0.01,
                0.1,
                1,
                10
                ],
                "penalty": [
                "l2"
                ]
            },
            "RandomForestClassifier": {
                "n_estimators": [
                100,
                200
                ],
                "max_depth": [
                4,
                6,
                8
                ],
                "min_samples_split": [
                2,
                5
                ]
            },
            "XGBClassifier": {
                "n_estimators": [
                100,
                200
                ],
                "learning_rate": [
                0.01,
                0.1
                ],
                "max_depth": [
                3,
                5,
                7
                ]
            }
            },
            "scoring": "f1",
            "cv": 5,
            "n_iter": 20
        },
        "pipeline": "Combine preprocessing and model in sklearn Pipeline for reproducibility"
        },
        "evaluation": {
        "metrics": [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ],
        "cross_validation": "Report mean and standard deviation across folds",
        "final_model_selection": "Choose model with best balance of F1 and ROC AUC",
        "test_set_evaluation": "Retrain selected model on full training data, evaluate on hold-out test set"
        },
        "prediction": {
        "procedure": "Apply preprocessing pipeline to test set, predict Survived probabilities and classes, prepare submission"
        }
    }
    }
    
    state = logic.State(
        train_ds=train_ds,
        test_ds=test_ds,
        prompt="",
        target=target,
        task=task,
        preprocess_spec=preprocess_spec,
        plan=plan
    )
    
    init_feature_engineering(state)