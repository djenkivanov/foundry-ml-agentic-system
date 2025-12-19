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

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

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
        model="o4-mini",
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
    
    state.plan = final_response.output[1].content[0].text
    state.stage = "preprocess"
    state.task = json.loads(state.plan).get("plan", [{}]).get("task", "")
    state.target = json.loads(state.plan).get("plan", [{}]).get("target", "")


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


if __name__ == "__main__":
    df = pd.read_csv("trainWithNull.csv")
    state = State(
        prompt="",
        train_ds=df,
        test_ds=df,
    )
    
    # infer planner agent -> get data insight and create plan
    agents.planner_agent(state)
