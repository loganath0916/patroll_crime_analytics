import streamlit as st
import pandas as pd

df = pd.read_csv("../data/crime_feature_engineered.csv")

st.title("Chicago Crime Analytics Dashboard")

st.metric("Total Crimes", len(df))
st.metric("Crime Types", df["Primary Type"].nunique())
st.metric("Districts", df["District"].nunique())

arrest_rate = round(df["Arrest"].mean() * 100, 2)
st.metric("Arrest Rate (%)", arrest_rate)