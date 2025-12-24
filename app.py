import json
import streamlit as st
import pandas as pd
import logic
import agents
from custom_state import State

st.title("Foundry ML")

training_ds = st.file_uploader("Upload TRAINING dataset", type=["csv"])
test_ds = st.file_uploader("Upload TEST dataset", type=["csv"])
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
if "plan_successful" not in st.session_state:
    st.session_state.plan_successful = False
if "training_started" not in st.session_state:
    st.session_state.training_started = False

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
            
        with st.spinner("Planning in progress...", show_time=True):
            agents.planner_agent(st.session_state.state, reasoning_stream, plan_stream)

            if st.session_state.state.stage == "failed":
                status_box.error(f"Error during planning: {st.session_state.state.errors}")
            else:
                st.session_state.plan_done = True
                st.session_state.plan_successful = True
            
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
            f"```json\n{json.dumps(json.loads(st.session_state.plan_text), indent=2)}\n```"
        )
    if st.session_state.plan_successful:
        st.success("Planning completed successfully!")
        
    if st.button("Proceed to Preprocessing Agent"):
        st.session_state.preprocessing_started = True


if st.session_state.preprocessing_started:
    st.subheader("Preprocessing Agent")
    with st.spinner("Preprocessing in progress...", show_time=True):
        agents.preprocessing_agent(st.session_state.state)
    st.markdown(f"```json\n{json.dumps(st.session_state.state.preprocess_spec, indent=2)}\n```")
    st.dataframe(st.session_state.state.train_ds.head())
    st.dataframe(st.session_state.state.test_ds.head())
    st.dataframe(st.session_state.state.x_train.head())
    st.dataframe(st.session_state.state.x_test.head())
    if st.session_state.state.stage == "failed":
        st.error(f"Error during preprocessing: {st.session_state.state.errors}")
    else:
        st.success("Preprocessing completed successfully!")
        
    if st.button("Proceed to Training Agent"):
        st.session_state.training_started = True
        
if st.session_state.training_started:
    st.subheader("Training Agent")
    with st.spinner("Training in progress...", show_time=True):
        agents.training_agent(st.session_state.state)
    st.subheader("Training Plan")
    st.markdown(f"```json\n{json.dumps(st.session_state.state.training_plan, indent=2)}\n```")
    
    st.subheader("Best Model and Scores")
    st.write(st.session_state.state.model)
    st.markdown(f"```json\n{json.dumps(st.session_state.state.model_scores, indent=2)}\n```")
    
    if st.session_state.state.stage == "failed":
        st.error(f"Error during training: {st.session_state.state.errors}")
    else:
        st.success("Training plan created successfully!")
    
        
        


    
