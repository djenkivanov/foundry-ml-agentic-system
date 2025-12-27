import json
import streamlit as st
import pandas as pd
import logic
import agents
from custom_state import State
import db

st.title("Foundry ML")

training_ds = st.file_uploader("Upload TRAINING dataset", type=["csv"])
prompt = st.text_area("Enter the prompt for the desired ML model")

if "train_df" not in st.session_state:
    st.session_state.train_df = None
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
if "preprocess_started" not in st.session_state:
    st.session_state.preprocess_started = False
if "streaming_in_progress" not in st.session_state:
    st.session_state.streaming_in_progress = False
if "plan_successful" not in st.session_state:
    st.session_state.plan_successful = False
if "training_started" not in st.session_state:
    st.session_state.training_started = False
if "training_done" not in st.session_state:
    st.session_state.training_done = False
if "evaluation_started" not in st.session_state:
    st.session_state.evaluation_started = False
if "evaluation_done" not in st.session_state:
    st.session_state.evaluation_done = False

if st.button("Start foundry process"):
    if training_ds is None:
        st.error("Please upload the training dataset, and fill in the prompt.")
    else:
        st.session_state.train_df = pd.read_csv(training_ds)
        st.session_state.prompt = prompt

if st.session_state.train_df is not None:
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
            raw_train_ds=st.session_state.train_df,
            fe_train_ds=st.session_state.train_df.copy()
        )
            
        with st.spinner("Planning in progress...", show_time=True):
            agents.planner_agent(st.session_state.state, reasoning_stream, plan_stream)

            if st.session_state.state.stage == "failed":
                status_box.error(f"Error during planning: {st.session_state.state.error}")
            else:
                st.session_state.plan_done = True
                st.session_state.plan_successful = True
            
            st.session_state.streaming_in_progress = False
            reasoning_stream.empty()
            plan_stream.empty()

if st.session_state.state.stage == "failed" or st.session_state.state.stage == "success":
    db.log_task(st.session_state.state)

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
        st.session_state.preprocess_started = True


if st.session_state.preprocess_started:
    st.subheader("Preprocessing Agent")
    if not st.session_state.preprocess_done:
        with st.spinner("Preprocessing in progress...", show_time=True):
            agents.preprocessing_agent(st.session_state.state)
        if st.session_state.state.stage == "failed":
            st.error(f"Error during preprocessing: {st.session_state.state.error}")
        else:
            st.session_state.preprocess_done = True

    if st.session_state.preprocess_done:
        st.markdown(f"```json\n{json.dumps(st.session_state.state.preprocess_spec, indent=2)}\n```")
        st.subheader("Preprocessed Training Data Preview")
        st.dataframe(st.session_state.state.x_train.head())
        st.success("Preprocessing completed successfully!")

    if st.button("Proceed to Training Agent"):
        st.session_state.training_started = True
        st.session_state.training_done = False
        
if st.session_state.training_started:
    st.subheader("Training Agent")
    if not st.session_state.training_done:
        with st.spinner("Training in progress...", show_time=True):
            agents.training_agent(st.session_state.state)
        if st.session_state.state.stage == "failed":
            st.error(f"Error during training: {st.session_state.state.error}")
        else:
            st.session_state.training_done = True
            st.success("Training plan created successfully!")

    if st.session_state.training_done:
        st.subheader("Training Plan")
        st.markdown(f"```json\n{json.dumps(st.session_state.state.training_plan, indent=2)}\n```")
        
        st.subheader("Best Model and Scores")
        st.write(st.session_state.state.model)
        st.markdown(f"```json\n{json.dumps(st.session_state.state.best_model_scores, indent=2)}\n```")
        
    if st.button("Prepare Trained Model for Download"):
        with st.spinner("Packaging model...", show_time=True):
            agents.package_agent(st.session_state.state)
        if st.session_state.state.stage == "failed":
            st.error(f"Error during model packaging: {st.session_state.state.error}")
        else:
            with open(st.session_state.state.model_package_path, "rb") as model_file:
                st.download_button(
                    label="Download Model Package",
                    data=model_file,
                    file_name=st.session_state.state.model_package_path,
                    mime="application/octet-stream"
                )
        
        


    
