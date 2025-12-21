import json
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
if "plan_done" not in st.session_state:
    st.session_state.plan_done = False
if "preprocess_done" not in st.session_state:
    st.session_state.preprocess_done = False
if "preprocessing_started" not in st.session_state:
    st.session_state.preprocessing_started = False
if "streaming_in_progress" not in st.session_state:
    st.session_state.streaming_in_progress = False

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
        st.session_state.streaming_in_progress = True
        st.rerun()
    
    if st.session_state.streaming_in_progress and not st.session_state.plan_done:
        reasoning_stream = st.empty()
        plan_stream = st.empty()
        status_box = st.empty()
        
        # init state that will be used across agents
        st.session_state.state = State(
            prompt=st.session_state.prompt,
            train_ds=st.session_state.train_df,
            test_ds=st.session_state.test_df,
        )
        
        # infer planner agent -> get data insight and create plan
        agents.planner_agent(st.session_state.state, reasoning_stream, plan_stream)

        if st.session_state.state.stage == "failed":
            status_box.error(f"Error during planning: {st.session_state.state.errors}")
        else:
            st.session_state.plan_done = True
        
        st.session_state.streaming_in_progress = False
        reasoning_stream.empty()
        plan_stream.empty()

if st.session_state.plan_done:
    if st.session_state.reasoning_text:
        st.markdown(
            "## Planner Agent Reasoning\n\n"
            f"{st.session_state.reasoning_text}"
        )

    if st.session_state.plan_text:
        st.markdown(
            "## Planner Agent Plan\n\n"
            f"```json\n{st.session_state.plan_text}\n```"
        )
    
    st.success("Planning completed successfully!")
        
    if st.button("Proceed to Preprocessing Agent"):
        st.session_state.preprocessing_started = True


if st.session_state.preprocessing_started:
    st.subheader("Preprocessing Agent")
    agents.preprocessing_agent(st.session_state.state)
    st.markdown(f"```json\n{json.dumps(st.session_state.state.preprocess_spec)}\n```")
    if st.session_state.state.stage == "failed":
        st.error(f"Error during preprocessing: {st.session_state.state.errors}")
    else:
        st.success("Preprocessing completed successfully!")
    st.dataframe(st.session_state.state.x_train.head())
    st.dataframe(st.session_state.state.y_train.head())
        
        


    
