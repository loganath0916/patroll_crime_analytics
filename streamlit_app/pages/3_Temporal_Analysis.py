import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Temporal Crime Analysis")

# Load dataset
df = pd.read_csv("../data/crime_feature_engineered.csv")

# -----------------------------
# Hourly Crime Pattern
# -----------------------------
st.subheader("🕒 Crimes by Hour")

hourly_crime = (
    df.groupby("Hour")
    .size()
    .reset_index(name="Crime_Count")
)

fig_hour = px.line(
    hourly_crime,
    x="Hour",
    y="Crime_Count",
    markers=True,
    title="Crime Distribution by Hour"
)

st.plotly_chart(fig_hour, use_container_width=True)

# -----------------------------
# Monthly Crime Pattern
# -----------------------------
st.subheader("📅 Crimes by Month")

monthly_crime = (
    df.groupby("Month")
    .size()
    .reset_index(name="Crime_Count")
)

fig_month = px.bar(
    monthly_crime,
    x="Month",
    y="Crime_Count",
    title="Crime Distribution by Month"
)

st.plotly_chart(fig_month, use_container_width=True)

# -----------------------------
# Weekday vs Weekend
# -----------------------------
st.subheader("📊 Weekday vs Weekend")

weekend_count = (
    df.groupby("IsWeekend")
    .size()
    .reset_index(name="Crime_Count")
)

weekend_count["Day_Type"] = weekend_count["IsWeekend"].map(
    {0: "Weekday", 1: "Weekend"}
)

fig_weekend = px.pie(
    weekend_count,
    names="Day_Type",
    values="Crime_Count",
    title="Weekday vs Weekend Crimes"
)

st.plotly_chart(fig_weekend, use_container_width=True)

# -----------------------------
# Crime Type Trends
# -----------------------------
st.subheader("🚨 Crime Types Distribution")

if "Primary Type" in df.columns:

    crime_type = (
        df["Primary Type"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    crime_type.columns = ["Crime_Type", "Count"]

    fig_crime = px.bar(
        crime_type,
        x="Count",
        y="Crime_Type",
        orientation="h",
        title="Top 15 Crime Types"
    )

    st.plotly_chart(fig_crime, use_container_width=True)

# -----------------------------
# Heatmap Hour vs Month
# -----------------------------
st.subheader("🔥 Crime Density Heatmap")

heatmap_data = (
    df.groupby(["Month", "Hour"])
    .size()
    .reset_index(name="Crime_Count")
)

fig_heatmap = px.density_heatmap(
    heatmap_data,
    x="Hour",
    y="Month",
    z="Crime_Count",
    title="Crime Density by Month and Hour"
)

st.plotly_chart(fig_heatmap, use_container_width=True)
