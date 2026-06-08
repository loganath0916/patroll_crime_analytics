import streamlit as st
import pandas as pd
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Crime Analytics Dashboard",
    page_icon="🚔",
    layout="wide"
)

# Load Dataset
DATA_PATH = Path(__file__).parents[2] / "data" / "crime_deployment.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()

    st.title("🚔 Chicago Crime Analytics Dashboard")
    st.markdown("### Project Overview")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Crimes",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Crime Types",
            df["Primary Type"].nunique()
        )

    with col3:
        st.metric(
            "Districts",
            df["District"].nunique()
        )

    with col4:
        arrest_rate = round(
            df["Arrest"].mean() * 100,
            2
        )

        st.metric(
            "Arrest Rate %",
            arrest_rate
        )

    st.divider()

    # Dataset Preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # Dataset Information
    st.subheader("📊 Dataset Information")

    info_df = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing Values"
        ],
        "Value": [
            df.shape[0],
            df.shape[1],
            df.isnull().sum().sum()
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

except Exception as e:
    st.error(f"Error loading dataset: {e}")