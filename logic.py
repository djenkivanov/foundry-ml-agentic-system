from openai import OpenAI
import prompts
import os
from dotenv import load_dotenv
import pandas as pd
import agents
import json
import re
import streamlit as st
from custom_state import State, Task
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
import feature_engineering
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
model = "o4-mini"

def get_data_insight(state: State) -> State:
    df_train_insights = return_insight_summary(state.train_ds)
    df_test_insights = return_insight_summary(state.test_ds)
    
    insights = {
        "training_dataset": df_train_insights,
        "test_dataset": df_test_insights
    }
    
    state.insights = insights
        

def return_insight_summary(df):
    shape = df.shape

    # cols_missing_values_sum = df.isna().sum()
    # cols_missing_values = [k for k, v in cols_missing_values_sum.items() if v > 0]

    cols_missing_values = df.isna().sum()

    dtypes = df.dtypes
    # description = df.describe()
    unique_counts = df.nunique()

    insights = {
        "Shape": shape,
        "Columns with missing values": cols_missing_values,
        "Data Types": dtypes,
        # "Description": description,
        "Unique Counts": unique_counts
    }
    
    return insights


def create_initial_plan(state, reasoning_stream=None, plan_stream=None):
    planner_prompt = build_planner_prompt(state)
    state.prompt = "None" if not state.prompt else state.prompt
    final_response = None
    
    with client.responses.stream(
        model=model,
        reasoning={"summary": "detailed"},
        input=[
            {
                "role": "system",
                "content": prompts.PLANNER_AG
            },
            {
                "role": "user",
                "content": f"{planner_prompt}\nUser Prompt: {state.prompt}"
            }
        ],
    ) as stream:
        for event in stream:
            if event.type == "response.reasoning_summary_text.delta":
                st.session_state.reasoning_text += event.delta
                st.session_state.reasoning_text = re.sub(r"(?<!\n)\*\*(?=\S)", "\n\n**", st.session_state.reasoning_text)
                reasoning_stream.markdown(f"## Planner Agent Reasoning:\n\n{st.session_state.reasoning_text}")

            elif event.type == "response.output_text.delta":
                st.session_state.plan_text += event.delta
                plan_stream.markdown(f"## Planner Agent Plan\n\n```json\n{st.session_state.plan_text}\n```")
    
        final_response = stream.get_final_response()
    
    state.plan = json.loads(final_response.output[1].content[0].text)
    state.stage = "preprocess"
    state.task = state.plan.get("plan", [{}]).get("task", "")
    state.target = state.plan.get("plan", [{}]).get("target", "")


def build_planner_prompt(state):
    valid_tasks = ", ".join(Task.__args__)
    pretty_train_insights = "\n".join([f"{k}:\n{v}\n\n" for k, v in state.insights["training_dataset"].items()])
    pretty_test_insights = "\n".join([f"{k}:\n{v}\n\n" for k, v in state.insights["test_dataset"].items()])
    prompt = f"""
    {prompts.PLANNER_AG}
    
    For the task, choose one of the following valid task types: {valid_tasks}.
    
    Training dataset insights:
    {pretty_train_insights}
    
    Test dataset insights:
    {pretty_test_insights}
    """
    return prompt


def create_preprocess_spec(state: State) -> str:
    pretty_train_insights = "\n".join([f"{k}:\n{v}\n\n" for k, v in state.insights["training_dataset"].items()])
    preprocess_prompt = f"""
    Here is the training dataset insights:
    {pretty_train_insights}
    
    Here is the preprocessing plan:
    {json.dumps(state.plan, indent=2)}
    """
    final_response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompts.PREPROCESSING_AG
            },
            {
                "role": "user",
                "content": preprocess_prompt
            }
        ],
    )
    
    preprocess_spec = final_response.choices[0].message.content
    state.preprocess_spec = json.loads(preprocess_spec)


def execute_preprocess_spec(state: State) -> State:
    feature_engineering.init_feature_engineering(state)
    
    ct, df_train, df_test = get_ct(state)

    df_processed_train = ct.fit_transform(df_train)
    
    state.x_train = pd.DataFrame(df_processed_train, columns=ct.get_feature_names_out())
    state.y_train = state.train_ds[state.target]
    
    df_processed_test = ct.transform(df_test)
    
    state.x_test = pd.DataFrame(df_processed_test, columns=ct.get_feature_names_out())
    
    state.stage = "train"
    

def get_ct(state):
    drop_columns = state.preprocess_spec.get("drop_columns", [])
    numeric = state.preprocess_spec.get("numeric", {})
    categorical = state.preprocess_spec.get("categorical", {})
    df_train = state.train_ds.copy()
    df_train = df_train.drop(columns=[state.target])
    
    df_test = state.test_ds.copy()
    
    if state.target in drop_columns:
        drop_columns.remove(state.target)
    
    if drop_columns:
        df_train = df_train.drop(columns=drop_columns)
        df_test = df_test.drop(columns=drop_columns)
    if numeric.get("columns"):
        cols_num = numeric["columns"] if numeric.get("columns") != "auto" else df_train.select_dtypes(include=["number"]).columns.tolist()
               
    if numeric.get("imputer"):
        imputer_strategy = numeric["imputer"]
        
    if numeric.get("scaler"):
        scaler = scalers.get(numeric["scaler"])
       
    if categorical.get("columns"):
        cols_cat = categorical["columns"] if categorical.get("columns") != "auto" else df_train.select_dtypes(include=["object", "category"]).columns.tolist()

    if categorical.get("imputer"):
        imputer_strategy = categorical["imputer"]
        
    if categorical.get("encoder"):
        encoder = encoders.get(categorical["encoder"])
        
    remove_unknown_columns([cols_num, cols_cat], state)
        
    ct = ColumnTransformer(transformers=[
        ('n1', SimpleImputer(strategy=imputer_strategy), cols_num),
        ('n2', scaler(), cols_num),
        ('c1', SimpleImputer(strategy=imputer_strategy), cols_cat),
        ('c2', encoder(), cols_cat)
    ], remainder='drop')

    return ct, df_train, df_test


def remove_unknown_columns(cols, state):
    for col_list in cols:
        for col in col_list[:]:
            if col not in state.train_ds.columns:
                col_list.remove(col)


def refine_training_plan(state: State):
    training_prompt = f"""
    Here is the initial plan:
    {json.dumps(state.plan, indent=2)}
    
    Here is the preprocessing specification:
    {json.dumps(state.preprocess_spec, indent=2)}
    
    The target variable is: {state.target}
    The task type is: {state.task}
    
    Based on this information, create a detailed training plan as a valid JSON object.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompts.TRAINING_AG
            },
            {
                "role": "user",
                "content": training_prompt
            }
        ],
    )
    
    training_plan = response.choices[0].message.content
    state.training_plan = json.loads(training_plan)


def convert_training_plan_to_code(state: State):
    training_plain = state.training_plan.get("training", {})
    split = training_plain.get("split", {})
    cv = training_plain.get("cv", {})
    
    random_state = split.get("random_state", 42)
    x_train, x_val, y_train, y_val = train_test_split(
      state.x_train,
      state.y_train,
      test_size=split.get("val_size", 0.2),
      stratify=state.y_train if str(split.get("stratified", False)).lower() == "true" else None,
      random_state=random_state
    )
    
    cross_val_params = {
        "scoring": cv.get("scoring", "accuracy"),
        "cv": cv.get("n_splits", 5),
    }
    
    models_with_params = [(model_dict.get("name"), model_dict.get("params_grid", {})) for model_dict in training_plain.get("models", [])]
    
    best_model = None
    best_val_score = -float("inf")
    scores = {}
    
    for model_name, params_grid in models_with_params:
        model_init = models.get(model_name)
        if not model_init:
            continue
        
        model_instance = model_init(random_state=random_state)
        
        grid_search = GridSearchCV(
            estimator=model_instance,
            param_grid=params_grid,
            scoring=cross_val_params["scoring"],
            cv=cross_val_params["cv"],
            n_jobs=-1
        )
        
        grid_search.fit(x_train, y_train)
        
        val_score = grid_search.best_estimator_.score(x_val, y_val)
        train_score = grid_search.best_score_
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = grid_search.best_estimator_
            scores = {
                "model_name": model_name,
                "train_score": train_score,
                "val_score": val_score,
            }
    
    state.model = best_model
    state.model_scores = scores
    
    
models = {
    "LogisticRegression": lambda **kwargs: LogisticRegression(**kwargs),
    "RandomForestClassifier": lambda **kwargs: RandomForestClassifier(**kwargs),
    "LinearRegression": lambda **kwargs: LinearRegression(**kwargs),
    "RandomForestRegressor": lambda **kwargs: RandomForestRegressor(**kwargs)
}

scalers = {
    "standard": lambda: preprocessing.StandardScaler(),
    "minmax": lambda: preprocessing.MinMaxScaler(),
    "robust": lambda: preprocessing.RobustScaler()
}

encoders = {
    "onehot": lambda: preprocessing.OneHotEncoder(handle_unknown="ignore"),
    "ordinal": lambda: preprocessing.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
}


if __name__ == "__main__":
    train_ds = pd.read_csv("titanic train.csv")
    test_ds = pd.read_csv("titanic test.csv")
    target = "Survived"
    task = "classification"
    
    training_plan = json.loads("""{
  "training": {
    "split": {
      "stratified": "true",
      "val_size": 0.2,
      "random_state": 42
    },
    "cv": {
      "n_splits": 5,
      "scoring": "roc_auc"
    },
    "models": [
      {
        "name": "LogisticRegression",
        "params_grid": {
          "C": [
            0.01,
            0.1,
            1,
            10
          ],
          "penalty": [
            "l2"
          ],
          "solver": [
            "liblinear"
          ]
        }
      },
      {
        "name": "RandomForestClassifier",
        "params_grid": {
          "n_estimators": [
            100,
            200,
            500
          ],
          "max_depth": [
            null,
            5,
            10,
            20
          ],
          "min_samples_split": [
            2,
            5,
            10
          ]
        }
      },
      {
        "name": "XGBClassifier",
        "params_grid": {
          "n_estimators": [
            100,
            200,
            500
          ],
          "max_depth": [
            3,
            5,
            7
          ],
          "learning_rate": [
            0.01,
            0.1,
            0.2
          ],
          "subsample": [
            0.6,
            0.8,
            1.0
          ],
          "colsample_bytree": [
            0.6,
            0.8,
            1.0
          ]
        }
      }
    ]
  }
}
""")
    
    preprocess_spec = {
  "drop_columns": [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
  ],
  "numeric": {
    "columns": [
      "Pclass",
      "Age",
      "SibSp",
      "Parch",
      "Fare",
      "FamilySize",
      "IsAlone"
    ],
    "imputer": "median",
    "scaler": "robust"
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
  "feature_engineering": [
    {
      "operation": {
        "new_column": "Title",
        "expression": "df['Name'].str.extract(', (.*?)\\.')"
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
        "new_column": "Deck",
        "expression": "df['Cabin'].fillna('U').str[0]"
      }
    },
    {
      "operation": {
        "new_column": "AgeBin",
        "expression": "pd.qcut(df['Age'], 4, labels=False)"
      }
    },
    {
      "operation": {
        "new_column": "FareBin",
        "expression": "pd.qcut(df['Fare'], 4, labels=False)"
      }
    }
  ]
}
    
    plan = {
  "plan": {
    "task": "classification",
    "target": "Survived",
    "preprocess": {
      "missing_values": "Age: impute by median age grouped by Title (extracted from Name). Fare (test only one missing): impute with median. Embarked: fill with mode. Cabin: extract Deck letter (first char); fill missing Cabin as 'U' for Unknown.",
      "feature_engineering": "Extract Title from Name and group rare titles into 'Other'. Create FamilySize = SibSp + Parch + 1. Create IsAlone flag = 1 if FamilySize == 1 else 0. Extract Deck from Cabin. Optionally bin Fare and Age into quartiles.",
      "categorical_encoding": "Sex: map to {male:0, female:1}. One-hot encode Title, Embarked, Deck. Pclass kept as ordinal numeric.",
      "feature_selection": "Drop Name, Ticket, Cabin (after Deck extraction), PassengerId. Retain Pclass, Sex, Age, SibSp, Parch, Fare, Embarked dummies, Title dummies, Deck dummies, FamilySize, IsAlone.",
      "scaling": "Apply RobustScaler or StandardScaler to numeric features (Age, Fare, FamilySize) within a pipeline to mitigate outliers."
    },
    "model_selection": "Train a baseline Logistic Regression for interpretability. Then train tree-based models: RandomForestClassifier and XGBoostClassifier for non-linear interactions and robust handling of missing/imputed data. Consider light ensembling of top two models.",
    "training": {
      "train_val_split": "Perform stratified k-fold cross-validation (k=5) on the training set to maintain class balance.",
      "hyperparameter_tuning": "Use RandomizedSearchCV or Bayesian Optimization within the cross-validation folds to tune key hyperparameters: for RF (n_estimators, max_depth, min_samples_split), for XGBoost (n_estimators, max_depth, learning_rate, subsample, colsample_bytree).",
      "pipeline": "Build sklearn Pipeline combining preprocessing steps and model to prevent data leakage.",
      "feature_importance": "After training tree models, extract feature importances to confirm or refine feature set."
    },
    "evaluation": {
      "metrics": "Primary: ROC AUC. Secondary: accuracy, precision, recall, F1-score. Use confusion matrix to inspect error types.",
      "cross_validation": "Report mean and standard deviation of CV metrics. Use stratified 5-fold CV.",
      "final_assessment": "Plot ROC curves for each model; choose model with best trade-off of AUC and F1.",
      "threshold_tuning": "Optionally adjust decision threshold to optimize F1 or recall as per business needs."
    },
    "prediction": "Retrain the selected model on the full training set with best hyperparameters. Apply the same preprocessing pipeline to the test set and generate final Survived predictions."
  }
}
    
    state = State(
        train_ds=train_ds,
        test_ds=test_ds,
        prompt="",
        target=target,
        task=task,
        preprocess_spec=preprocess_spec,
        plan=plan,
        training_plan=training_plan
    )
    
    execute_preprocess_spec(state)
    convert_training_plan_to_code(state)


