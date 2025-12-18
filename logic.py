from openai import OpenAI
import prompts
import os
from dotenv import load_dotenv
import pandas as pd
import agents
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def get_data_insight(df):
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
    

def create_initial_plan(user_prompt, train_ds_insights, test_ds_insights):
    planner_prompt = build_planner_prompt(train_ds_insights, test_ds_insights)

    plan = client.responses.create(
        model="o4-mini",
        reasoning={"summary": "auto"},
        input=[
            {
                "role": "system",
                "content": prompts.PLANNER_AG
            },
            {
                "role": "user",
                "content": f"{planner_prompt}\nUser Prompt: {user_prompt}"
            }
        ]
    )
    
    return (plan.output[0], plan.output[1].content[0].text)


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
    print(json.loads(plan))