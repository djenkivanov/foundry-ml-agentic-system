import streamlit as st
import pandas as pd
import logic

st.title("Foundry ML")

training_ds = st.file_uploader("Upload training dataset", type=["csv"])
test_ds = st.file_uploader("Upload test dataset", type=["csv"])
prompt = st.text_area("Enter the prompt for the desired ML model")

if "df_train" not in st.session_state:
    st.session_state.df_train = None
if "df_test" not in st.session_state:
    st.session_state.df_test = None
if "prompt" not in st.session_state:
    st.session_state.prompt = None
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = None

if st.button("Generate ML Model"):
    if training_ds is None or test_ds is None or prompt is None:
        st.error("Please upload both training and test datasets, and fill in the prompt.")
    else:
        st.session_state.df_train = pd.read_csv(training_ds)
        st.session_state.df_test = pd.read_csv(test_ds)
        st.session_state.prompt = prompt

if st.session_state.df_train is not None and st.session_state.df_test is not None and st.session_state.prompt is not None:
    st.subheader("Preview")
    st.dataframe(st.session_state.df_train.head())

    if st.button("Run data diagnostics check"):
        st.session_state.diagnostics = logic.get_data_insight(st.session_state.df_train)

if st.session_state.diagnostics is not None:
    st.subheader("Diagnostics")
    st.write(st.session_state.diagnostics)
