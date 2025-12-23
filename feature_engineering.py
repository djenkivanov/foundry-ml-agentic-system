from typing import List, Optional
import re
import pandas as pd
from sklearn import preprocessing
import logic


def init_feature_engineering(state):
    refine_and_execute_operations(state)


def refine_and_execute_operations(state):
    list_of_ops = state.preprocess_spec.get("feature_engineering", [])
    feature_engineered_df_train = execute_operations(state.train_ds, list_of_ops)
    feature_engineered_df_test = execute_operations(state.test_ds, list_of_ops)
    state.train_ds = feature_engineered_df_train
    state.test_ds = feature_engineered_df_test


def execute_operations(df, operations: List[dict]) -> dict:
    for operation in operations:
        for _, params in operation.items():
            df = execute_fe_code(df, **params)
            
    return df


def execute_fe_code(df, new_column, expression):
    df[new_column] = eval(expression)
    return df

    
if __name__ == "__main__":
    train_ds = pd.read_csv("titanic train.csv")
    test_ds = pd.read_csv("titanic test.csv")
    target = "Survived"
    task = "classification"
    
    preprocess_spec = {
  "drop_columns": [
    "Name",
    "Ticket",
    "Cabin"
  ],
  "numeric": {
    "columns": [
      "Age",
      "Fare",
      "FamilySize"
    ],
    "imputer": "median",
    "scaler": "standard"
  },
  "categorical": {
    "columns": [
      "Pclass",
      "Embarked",
      "Title",
      "Deck"
    ],
    "imputer": "most_frequent",
    "encoder": "onehot"
  },
  "feature_engineering": [
    {
      "operation": {
        "new_column": "Deck",
        "expression": "df['Cabin'].str[0].fillna('Unknown')"
      }
    },
    {
      "operation": {
        "new_column": "Title",
        "expression": "df['Name'].str.extract(' ([A-Za-z]+)\\\\.', expand=False)"
      }
    },
    {
      "operation": {
        "new_column": "FamilySize",
        "expression": "df['SibSp'] + df['Parch'] + 1"
      }
    },
    {
      "operation": {
        "new_column": "IsAlone",
        "expression": "(df['FamilySize'] == 1).astype(int)"
      }
    },
    {
      "operation": {
        "new_column": "Sex",
        "expression": "df['Sex'].map({'male': 0, 'female': 1})"
      }
    }
  ]
}
    
    plan = {
  "plan": {
    "task": "classification",
    "target": "Survived",
    "preprocess": {
      "missing_values": "Impute Age with median (overall or group-wise by Title), Fare with median, Embarked with mode. Extract Deck from Cabin (first letter) and fill missing Deck as 'Unknown'.",
      "feature_engineering": "Extract Title from Name, compute FamilySize = SibSp + Parch + 1, create IsAlone flag (FamilySize == 1). Drop Name, Ticket and raw Cabin after Deck extraction.",
      "categorical_encoding": "Map Sex to binary, one-hot encode Pclass, Embarked, Title and Deck.",
      "feature_scaling": "Standardize numeric features (Age, Fare, FamilySize) using StandardScaler."
    },
    "model_selection": "Train a baseline Logistic Regression for reference, then tree-based models: RandomForestClassifier and GradientBoostingClassifier (or XGBoost) to capture nonlinearities and interactions. These handle mixed features well and are robust to outliers.",
    "training": {
      "pipeline": "Build an sklearn Pipeline chaining preprocessing and model.",
      "cross_validation": "Use StratifiedKFold with 5 splits to preserve class balance.",
      "hyperparameter_tuning": "Use RandomizedSearchCV over key model hyperparameters (e.g., n_estimators, max_depth, learning_rate for GBM; n_estimators, max_features, max_depth for RF) with 5-fold CV and scoring='roc_auc'.",
      "class_imbalance": "Leverage class_weight='balanced' in models or use SMOTE as needed."
    },
    "evaluation": {
      "metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
      ],
      "validation_strategy": "Assess performance on CV folds and check variance. Select best model based on ROC AUC and balanced F1.",
      "final_evaluation": "Retrain best model on full training set, predict on test set, and output Survived probabilities and classes."
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
    
    logic.execute_preprocess_spec(state)