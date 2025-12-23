from typing import List, Optional
import re
import pandas as pd
from sklearn import preprocessing
import logic


def init_feature_engineering(state):
    refine_and_execute_operations(state)


def refine_and_execute_operations(state):
    # fe_spec = logic.get_refined_feature_engineering_spec(state)
    # state.new_fe_spec = fe_spec
    list_of_ops = fe_spec.get("feature_engineering", [])
    feature_engineered_df_train = execute_operations(state.train_ds, list_of_ops)
    feature_engineered_df_test = execute_operations(state.test_ds, list_of_ops)
    state.train_ds = feature_engineered_df_train
    state.test_ds = feature_engineered_df_test
    state.new_fe_spec = fe_spec
    state.preprocess_spec["feature_engineering"] = fe_spec


def execute_operations(df, operations: List[dict]) -> dict:
    for operation in operations:
        for _, params in operation.items():
            df = execute_fe_code(df, **params)
            
    return df


def str_extractor(s: str, patterns: List[str]) -> Optional[str]:
    pattern = re.compile("|".join(patterns))
    match = pattern.search(s)
    return match.group(0) if match else None


def execute_fe_code(df, new_column, expression):
    df[new_column] = eval(expression)
    return df

# TODO: need to handle case where new_column is None (i.e., overwrite)
# def extract(df, source_column, new_column, match):
#     df[new_column] = df[source_column].apply(lambda x: str_extractor(str(x), match))
#     return df


# def encode(df, source_column, method):
#     if method == "onehot":
#         encoder = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#         encoded_cols = encoder.fit_transform(df[[source_column]])
#         encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out([source_column]))
#         df = pd.concat([df.drop(columns=[source_column]), encoded_df], axis=1)
#     elif method == "ordinal":
#         encoder = preprocessing.OrdinalEncoder()
#         df[source_column] = encoder.fit_transform(df[[source_column]])
#     return df


# available_operations = {
#     "derive": derive,
#     "extract": extract,
#     "encode": encode,
# }

# def get_valid_feature_engineering_methods():
#     return """
#     {
#         "derive": "Derive a new feature using an expression involving existing columns.",
#         "extract": "Extract parts of a feature based on matching patterns.",
#         "encode": "Encode a categorical feature using a specified encoding method.",
#     }
#     """

# def get_feature_engineering_spec():
#     return """
#     feature_engineering": [
#         {
#             "operation": {
#                 "new_column": "new_feature_name",
#                 /* only write the expression needed to compute the new column in valid Python syntax */
#                 "expression": "df['column1'] + df['column2']" 
#             }
#         },
#         {
#             "operation": {
#                 "new_column": "another_feature_name",
#                 "expression": "df['column1'].apply(lambda x: str_extractor(str(x), ['pattern1', 'pattern2']))"
#             }
#         },
#     ]
#     """
    
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
        "Fare"
        ],
        "imputer": "median",
        "scaler": "standard"
    },
    "categorical": {
        "columns": "auto",
        "imputer": "most_frequent",
        "encoder": "onehot"
    },
    "feature_engineering": {
        "Title": "Extract title substring from Name (text between comma and period), then group rare titles under 'Other'",
        "FamilySize": "Compute SibSp + Parch + 1",
        "IsAlone": "Set to 1 if FamilySize == 1, otherwise 0",
        "TicketPrefix": "Extract leading non-numeric characters from Ticket; if none, assign 'None'",
        "CabinDeck": "Extract first letter of Cabin; replace missing values with 'Unknown'",
        "SexBinary": "Map Sex: male \u2192 0, female \u2192 1",
        "PclassOrdinal": "Treat Pclass as ordinal feature with its integer values (1, 2, 3)"
    }
    }
    
    plan = {
    "plan": {
        "task": "classification",
        "target": "Survived",
        "preprocess": {
        "missing_values": {
            "Age": "impute with median age",
            "Fare": "impute with median fare",
            "Embarked": "impute with mode",
            "Cabin": "extract deck letter; fill missing as 'Unknown'"
        },
        "categorical_encoding": {
            "Sex": "map to binary (male=0, female=1)",
            "Embarked": "one-hot encode",
            "Pclass": "treat as ordinal or one-hot encode",
            "Deck": "one-hot encode extracted deck"
        },
        "feature_engineering": [
            "Title: extract from Name (Mr, Mrs, Miss, etc.)",
            "FamilySize: SibSp + Parch + 1",
            "IsAlone: FamilySize == 1",
            "TicketPrefix: extract non-numeric prefix from Ticket",
            "CabinDeck: first letter of Cabin"
        ],
        "feature_scaling": {
            "continuous": "standardize Age and Fare for models requiring scaling"
        },
        "feature_selection": [
            "drop raw Name, Ticket, Cabin, PassengerId",
            "drop features with low variance or high collinearity after analysis"
        ]
        },
        "model_selection": "Compare baseline Logistic Regression, Random Forest and Gradient Boosting (e.g. XGBoost). Tree-based models handle heterogenous data and missing values well; logistic regression provides interpretability.",
        "training": {
        "pipeline": "assemble imputation, encoding, feature engineering, scaling and model into scikit-learn Pipeline or equivalent",
        "cross_validation": "Stratified K-Fold (k=5) on training set to preserve class balance",
        "hyperparameter_tuning": "use RandomizedSearchCV or Bayesian optimization over key parameters (n_estimators, max_depth, learning_rate, C for logistic)",
        "class_imbalance": "if needed, apply class weights or resampling to handle any imbalance"
        },
        "evaluation": {
        "metrics": [
            "Accuracy",
            "ROC AUC",
            "Precision",
            "Recall",
            "F1-score"
        ],
        "procedure": "evaluate via cross-validation metrics; final check on hold-out test set",
        "model_comparison": "select model with best ROC AUC and balanced precision/recall",
        "calibration": "optionally calibrate probabilities (e.g. isotonic or Platt scaling) for downstream decision thresholds"
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