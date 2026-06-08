import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Temporal Analysis",
    page_icon="📅",
    layout="wide"
)

# Load Dataset
DATA_PATH = Path(__file__).parents[2] / "data" / "crime_deployment.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()

    st.title("📅 Temporal Crime Analysis")

    # Crime by Hour
    st.subheader("🕒 Crimes by Hour")

    hour_counts = (
        df.groupby("Hour")
        .size()
        .reset_index(name="Crime_Count")
        .sort_values("Hour")
    )

    fig_hour = px.line(
        hour_counts,
        x="Hour",
        y="Crime_Count",
        markers=True,
        title="Crime Distribution by Hour"
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )

    # Crime by Day of Week
    st.subheader("📆 Crimes by Day of Week")

    day_counts = (
        df.groupby("Day_of_Week")
        .size()
        .reset_index(name="Crime_Count")
    )

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    day_counts["Day_of_Week"] = pd.Categorical(
        day_counts["Day_of_Week"],
        categories=day_order,
        ordered=True
    )

    day_counts = day_counts.sort_values(
        "Day_of_Week"
    )

    fig_day = px.bar(
        day_counts,
        x="Day_of_Week",
        y="Crime_Count",
        title="Crime Distribution by Day"
    )

    st.plotly_chart(
        fig_day,
        use_container_width=True
    )

    # Crime by Month
    st.subheader("📈 Crimes by Month")

    month_counts = (
        df.groupby("Month")
        .size()
        .reset_index(name="Crime_Count")
        .sort_values("Month")
    )

    fig_month = px.line(
        month_counts,
        x="Month",
        y="Crime_Count",
        markers=True,
        title="Crime Trend by Month"
    )

    st.plotly_chart(
        fig_month,
        use_container_width=True
    )

except Exception as e:
    st.error(f"Error loading data: {e}")