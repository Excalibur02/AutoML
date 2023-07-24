import pandas as pd
import streamlit as st
import os

import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
# classification for classification model, regression for regression model
from pycaret.regression import setup, compare_models, pull, save_model
with st.sidebar:
    st.image("https://yt3.googleusercontent.com/ytc/AGIKgqNFSgMOqjbzDdGdy70mqD9WRRqhy9z2UZg8nhXsGw=s900-c-k-c0x00ffffff-no-rj")
    st.title("AutoStreamML")
    choice = st.radio("Navigation",["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to make an Automated ML pipeline usig Streamlit, Pandas Profiling and PyCaret and it is straightforward magical !")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)
if choice == "Upload":
    st.title("Upload you DATA...")
    file = st.file_uploader("Upload your Dataset here")
    if file:
        df = pd.read_csv(file)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
if choice == "Profiling":
    st.title("Automated Exploratory data analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)
if choice == "ML":
    st.title("Machine Learning goes BRR....")
    target = st.selectbox("Select your target", df.columns)
    if st.button("Train Model"):
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info("This is the ML experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, "best_model")
if choice == "Download":
    pass