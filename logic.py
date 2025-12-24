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
from sklearn.model_selection import train_test_split, cross_validate


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
    ], remainder='passthrough')

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


def initiate_training_process(state: State):
    refine_training_plan(state)
    convert_training_plan_to_code(state)


def convert_training_plan_to_code(state: State):
    training_plain = state.training_plan.get("training", {})
    split = training_plain.get("split", {})
    cv = training_plain.get("cv", {})
    
    x_train, x_val, y_train, y_val = train_test_split(
        state.x_train,
        state.y_train,
        test_size=split.get("val_size", 0.2),
        stratify=state.y_train if split.get("stratified", False) else None,
        random_state=split.get("random_state", 42)
    )
    
    cross_val_params = {
        "scoring": cv.get("scoring", "accuracy"),
        "cv": cv.get("n_splits", 5),
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
    df_train = pd.read_csv("titanic train.csv")
    df_test = pd.read_csv("titanic test.csv")
    
    spec = {
        "drop_columns": ["Name", "Ticket", "Cabin"],
        "numeric": {
            "columns": ["Age", "Fare"],
            "imputer": "median",
            "scaler": "robust"
        },
        "categorical": {
            "columns": ["Sex", "Embarked", "Pclass", "Deck", "Title", "AgeBin", "FareBin"],
            "imputer": "most_frequent",
            "encoder": "onehot"
        }
    }
    
    state = State(
        prompt="",
        train_ds=df_train,
        test_ds=df_test,
        preprocess_spec=spec,
        target="Survived",
    )
    
    execute_preprocess_spec(state)


