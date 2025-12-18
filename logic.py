from openai import OpenAI
import prompts
import os
from dotenv import load_dotenv
import pandas as pd
import agents
import json
import re
import streamlit as st

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def get_data_insight(df_train, df_test):
    df_train_insights = return_insight_summary(df_train)
    df_test_insights = return_insight_summary(df_test)
    
    insights = {
        "training_dataset": df_train_insights,
        "test_dataset": df_test_insights
    }
    
    return insights
    

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


def create_initial_plan(user_prompt, insights, reasoning_stream=None, plan_stream=None):
    planner_prompt = build_planner_prompt(insights["training_dataset"], insights["test_dataset"])
    user_prompt = "None" if not user_prompt else user_prompt
    final_response = None
    
    with client.responses.stream(
        model="o4-mini",
        reasoning={"summary": "detailed"},
        input=[
            {
                "role": "system",
                "content": prompts.PLANNER_AG
            },
            {
                "role": "user",
                "content": f"{planner_prompt}\nUser Prompt: {user_prompt}"
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
    
    return final_response.output[1].content[0].text


def build_planner_prompt(train_ds_insights, test_ds_insights):
    valid_tasks = ", ".join(agents.Task.__args__)
    pretty_train_insights = "\n".join([f"{k}:\n{v}\n\n" for k, v in train_ds_insights.items()])
    pretty_test_insights = "\n".join([f"{k}:\n{v}\n\n" for k, v in test_ds_insights.items()])
    prompt = f"""
    {prompts.PLANNER_AG}
    
    For the task, choose one of the following valid task types: {valid_tasks}.
    
    Training dataset insights:
    {pretty_train_insights}
    
    Test dataset insights:
    {pretty_test_insights}
    """
    return prompt


if __name__ == "__main__":
    # df = pd.read_csv("DPtrain.csv")
    # df = pd.read_csv("trainWithNull.csv")
    # diagnostics = get_diagnostics(df)
    # diag_str = "\n".join([f"{k}:\n{v}" for k, v in diagnostics.items()])
    # print(diag_str)
    
    df = pd.read_csv("trainWithNull.csv")
    # print(df['Cabin'].unique())
    # print(df['Embarked'].unique())
    # print(df['Embarked'].isna().sum())
    
    insights = get_data_insight(df)
    reasoning, plan = create_initial_plan(
        user_prompt="Predict survival on the Titanic dataset.",
        train_ds_insights=insights,
        test_ds_insights=insights
    )
    json = "{\n  \"plan\": [\n    {\n      \"task\": \"classification\"\n    },\n    {\n      \"target\": \"Survived\"\n    },\n    {\n      \"preprocess\": \"Handle missing values and drop irrelevant columns. Impute Age by median (grouped by Title), fill Embarked with mode, extract Deck from Cabin (first letter) and fill unknown. Drop PassengerId, Ticket, Name (after feature extraction), Cabin. Log\u2010transform Fare to reduce skew.\"\n    },\n    {\n      \"feature_engineering\": \"Extract Title from Name; create FamilySize = SibSp + Parch + 1 and IsAlone flag; encode Pclass as ordinal; one\u2010hot encode Sex, Embarked, Title, Deck; scale numeric features (Age, Fare, FamilySize) with StandardScaler.\"\n    },\n    {\n      \"model_selection\": \"Benchmark Logistic Regression (baseline), Random Forest and XGBoost (for non-linear interactions). These handle mixed feature types and provide feature importance.\"\n    },\n    {\n      \"training\": \"Use stratified 5\u2010fold cross\u2010validation. Perform hyperparameter tuning via RandomizedSearchCV: regularization C for LR; n_estimators, max_depth, min_samples_split for RF; learning_rate, n_estimators, max_depth for XGB. Use early stopping on validation folds for XGB.\"\n    },\n    {\n      \"evaluation\": \"Assess performance on held-out folds and test set using accuracy, precision, recall, F1\u2010score and ROC AUC. Plot confusion matrix and ROC curve. Check calibration (calibration curve/Brier score).\"\n    },\n    {\n      \"ensemble\": \"If performance gains needed, ensemble top models via soft voting or stacking and re-evaluate metrics.\"\n    }\n  ]\n}"
    print(json.dumps(plan, indent=2))
    print(reasoning)
