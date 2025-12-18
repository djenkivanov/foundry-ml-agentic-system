from openai import OpenAI
import prompts
import os
from dotenv import load_dotenv
import pandas as pd
import agents

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def get_diagnostics(df):
    shape = df.shape
    
    cols_missing_values_sum = df.isna().sum()
    cols_missing_values = [k for k, v in cols_missing_values_sum.items() if v > 0]
    
    dtypes = df.dtypes
    description = df.describe()
    unique_counts = df.nunique()
    
    diagnostics = {
        "Shape": shape,
        "Columns with missing values": cols_missing_values,
        "Data Types": dtypes,
        "Description": description,
        "Unique Counts": unique_counts
    }
    
    return diagnostics
    

def create_initial_plan(user_prompt, train_ds_diagnostics, test_ds_diagnostics):
    planner_prompt = build_planner_prompt(train_ds_diagnostics, test_ds_diagnostics)
    plan = client.responses.create(
        model="o4-mini",
        prompt=planner_prompt,
        input=user_prompt,
    )
    return plan


def build_planner_prompt(train_ds_diagnostics, test_ds_diagnostics):
    valid_tasks = ", ".join(agents.Task.__args__)
    prompt = f"""
    {prompts.PLANNER_AG}
    
    For the task, choose one of the following valid task types: {valid_tasks}.
    
    Training dataset diagnostics:
    {train_ds_diagnostics}
    
    Test dataset diagnostics:
    {test_ds_diagnostics}
    
    Target column: [Specify the target column here]
    """
    return prompt


if __name__ == "__main__":
    # df = pd.read_csv("DPtrain.csv")
    # df = pd.read_csv("trainWithNull.csv")
    # diagnostics = get_diagnostics(df)
    # diag_str = "\n".join([f"{k}:\n{v}" for k, v in diagnostics.items()])
    # print(diag_str)
    
    df = pd.read_csv("trainWithNull.csv")
    # print(df['Embarked'].value_counts())
    # print(df['Embarked'].unique())
    # print(df['Embarked'].isna().sum())
    diagnostics = get_diagnostics(df)
    plan = create_initial_plan(
        user_prompt="Predict survival on the Titanic dataset.",
        train_ds_diagnostics=diagnostics,
        test_ds_diagnostics=diagnostics
    )