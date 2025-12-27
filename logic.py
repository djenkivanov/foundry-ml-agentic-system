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
from sklearn.pipeline import Pipeline
import feature_engineering
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
import joblib
import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
model = "o4-mini"

def get_data_insight(state: State) -> State:
    df_train_insights = return_insight_summary(state.raw_train_ds)
    
    insights = {
        "training_dataset": df_train_insights,
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
    
    state.reasoning = st.session_state.reasoning_text
    state.plan = json.loads(final_response.output[1].content[0].text)
    state.stage = "preprocess"
    state.task = state.plan.get("plan", [{}]).get("task", "")
    state.target = state.plan.get("plan", [{}]).get("target", "")


def build_planner_prompt(state):
    valid_tasks = ", ".join(Task.__args__)
    pretty_train_insights = "\n".join([f"{k}:\n{v}\n\n" for k, v in state.insights["training_dataset"].items()])
    prompt = f"""
    {prompts.PLANNER_AG}
    
    For the task, choose one of the following valid task types: {valid_tasks}.
    
    Training dataset insights:
    {pretty_train_insights}
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


def execute_preprocess_spec(state: State):
    feature_engineering.init_feature_engineering(state)
    
    ct, df_train = get_ct(state)

    df_processed_train = ct.fit_transform(df_train)
    
    # in case of sparse matrix, convert to dense array
    if hasattr(df_processed_train, 'toarray'):
        df_processed_train = df_processed_train.toarray()
    
    state.x_train = pd.DataFrame(df_processed_train, columns=ct.get_feature_names_out())
    state.y_train = state.raw_train_ds[state.target]
    
    state.stage = "train"
    

def get_ct(state):
    drop_columns = state.preprocess_spec.get("drop_columns", [])
    numeric = state.preprocess_spec.get("numeric", {})
    categorical = state.preprocess_spec.get("categorical", {})
    df_train = state.fe_train_ds.copy()
    df_train = df_train.drop(columns=[state.target])
        
    if state.target in drop_columns:
        drop_columns.remove(state.target)
    
    if drop_columns:
        df_train = df_train.drop(columns=drop_columns)
    
    cols_num = numeric.get("columns", []) if numeric.get("columns") != "auto" else df_train.select_dtypes(include=["number"]).columns.tolist()
    num_imputer_strategy = numeric.get("imputer", "mean")
    num_scaler = scalers.get(numeric.get("scaler", "standard"))
    
    cols_cat = categorical.get("columns", []) if categorical.get("columns") != "auto" else df_train.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_imputer_strategy = categorical.get("imputer", "most_frequent")
    cat_encoder = encoders.get(categorical.get("encoder", "onehot"))
    
    remove_unknown_columns([cols_num, cols_cat], state)
    
    num_transformer = Pipeline([
      ('imputer', SimpleImputer(strategy=num_imputer_strategy)),
      ('scaler', num_scaler())
    ])
    cat_transformer = Pipeline([
      ('imputer', SimpleImputer(strategy=cat_imputer_strategy)),
      ('encoder', cat_encoder())
    ])
    
    ct = ColumnTransformer(transformers=[
        ('num', num_transformer, cols_num),
        ('cat', cat_transformer, cols_cat)
    ], remainder='drop')

    return ct, df_train


def remove_unknown_columns(cols, state):
    for col_list in cols:
        for col in col_list[:]:
            if col not in state.raw_train_ds.columns:
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
    
    def perform_gridsearch(model_name, params_grid):
        nonlocal best_model, best_val_score, best_score, all_scores
        model_init = models.get(model_name)
        if not model_init:
            return
                
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
        
        all_scores[model_name] = {
            "train_score": train_score,
            "val_score": val_score,
        }
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = grid_search.best_estimator_
            best_score = {
                "model_name": model_name,
                "train_score": train_score,
                "val_score": val_score,
            }
    
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
    best_score = {}
    all_scores = {}
    
    with st.status("Training models...", expanded=True) as status:
        for idx, (model_name, params_grid) in enumerate(models_with_params, start=1):
            st.write(f"Fitting {model_name} ({idx}/{len(models_with_params)})...")
            perform_gridsearch(model_name, params_grid)
            model_scores = all_scores.get(model_name, {})
            st.write(f"Finished {model_name} | Train Score: {model_scores.get('train_score', 0):.5f} | Val Score: {model_scores.get('val_score', 0):.5f}")
        if best_score:
            status.update(label=f"Training complete! Best: {best_score['model_name']} (val={best_val_score:.5f})")
    
    state.stage = "success"
    state.model = best_model
    state.best_model_scores = best_score
    state.all_model_scores = all_scores


def package_model(state: State):
    model_filename = f"foundryML_trained_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

    ct, _ = get_ct(state)
    full_pipeline = Pipeline([
        ('feature_engineering', feature_engineering.init_feature_engineering(state)),
        ('preprocessor', ct),
        ('model', state.model)
    ])
    
    full_pipeline.fit(state.raw_train_ds.drop(columns=[state.target]), state.raw_train_ds[state.target])
    joblib.dump(full_pipeline, model_filename)
    state.model_package_path = model_filename


models = {
    "LogisticRegression": lambda **kwargs: LogisticRegression(**kwargs),
    "RandomForestClassifier": lambda **kwargs: RandomForestClassifier(**kwargs),
    "LinearRegression": lambda **kwargs: LinearRegression(**kwargs),
    "RandomForestRegressor": lambda **kwargs: RandomForestRegressor(**kwargs),
    "XGBClassifier": lambda **kwargs: xgb.XGBClassifier(**kwargs),
    "XGBRegressor": lambda **kwargs: xgb.XGBRegressor(**kwargs)
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
