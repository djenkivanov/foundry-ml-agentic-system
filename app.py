import streamlit as st
import pandas as pd
import logic

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
        st.session_state.insights = logic.get_data_insight(st.session_state.train_df, st.session_state.test_df)
        st.session_state.plan = logic.create_initial_plan(st.session_state.prompt, st.session_state.insights, reasoning_stream, plan_stream)
        
