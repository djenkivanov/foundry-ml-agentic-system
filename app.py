import streamlit as st
import pandas as pd
import logic
import agents
from custom_state import State

st.title("Foundry ML")

training_ds = st.file_uploader("Upload training dataset", type=["csv"])
test_ds = st.file_uploader("Upload test dataset", type=["csv"])
prompt = st.text_area("Enter the prompt for the desired ML model")

if "train_df" not in st.session_state:
    st.session_state.train_df = None
if "test_df" not in st.session_state:
    st.session_state.test_df = None
if "prompt" not in st.session_state:
    st.session_state.prompt = None
if "insights" not in st.session_state:
    st.session_state.insights = None
if "plan" not in st.session_state:
    st.session_state.plan = None
if "reasoning_text" not in st.session_state:
    st.session_state.reasoning_text = ""
if "plan_text" not in st.session_state:
    st.session_state.plan_text = ""

if st.button("Start foundry process"):
    if training_ds is None or test_ds is None:
        st.error("Please upload both training and test datasets, and fill in the prompt.")
    else:
        st.session_state.train_df = pd.read_csv(training_ds)
        st.session_state.test_df = pd.read_csv(test_ds)
        st.session_state.prompt = prompt

if st.session_state.train_df is not None and st.session_state.test_df is not None and st.session_state.prompt is not None:
    st.subheader("Preview")
    st.dataframe(st.session_state.train_df.head())

    if st.button("Get data insights and create plan"):
        reasoning_stream = st.empty()
        plan_stream = st.empty()
        status_box = st.empty()
        
        # init state that will be used across agents
        state = State(
            prompt=st.session_state.prompt,
            train_ds=st.session_state.train_df,
            test_ds=st.session_state.test_df,
        )
        
        # infer planner agent -> get data insight and create plan
        agents.planner_agent(state, reasoning_stream, plan_stream)
        if state.stage == "failed":
            status_box.error(f"Error during planning: {state.errors}")
        if state.target and state.task:
            status_box.success(f"Planning completed successfully! Target: {state.target}, Task: {state.task}")
        
