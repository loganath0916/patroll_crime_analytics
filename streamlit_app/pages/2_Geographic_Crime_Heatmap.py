import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Geographic Crime Heatmap",
    page_icon="🗺️",
    layout="wide"
)

# Load Dataset
DATA_PATH = Path(__file__).parents[2] / "data" / "crime_deployment.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()

    st.title("🗺️ Geographic Crime Heatmap")

    # Remove missing coordinates
    map_df = df.dropna(
        subset=["Latitude", "Longitude"]
    )

    st.write(
        f"Showing {len(map_df):,} crime records"
    )

    # Sample for faster rendering
    if len(map_df) > 10000:
        map_df = map_df.sample(
            10000,
            random_state=42
        )

    fig = px.scatter_map(
        map_df,
        lat="Latitude",
        lon="Longitude",
        color="Primary Type",
        zoom=9,
        height=700,
        title="Crime Distribution Across Chicago"
    )

    fig.update_layout(
        map_style="open-street-map",
        margin=dict(
            l=0,
            r=0,
            t=50,
            b=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except Exception as e:
    st.error(
        f"Error loading heatmap: {e}"
    )