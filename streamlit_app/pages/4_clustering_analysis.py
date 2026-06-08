import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Clustering Analysis",
    page_icon="🎯",
    layout="wide"
)

# Load Dataset
DATA_PATH = Path(__file__).parents[2] / "data" / "crime_deployment.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()

    st.title("🎯 Crime Clustering Analysis")

    # ===============================
    # KMEANS HOTSPOTS
    # ===============================
    st.subheader("📍 K-Means Crime Hotspots")

    if "KMeans_Cluster" in df.columns:

        sample_df = df.sample(
            min(10000, len(df)),
            random_state=42
        )

        fig = px.scatter_mapbox(
            sample_df,
            lat="Latitude",
            lon="Longitude",
            color="KMeans_Cluster",
            zoom=9,
            height=700,
            title="K-Means Crime Hotspots"
        )

        fig.update_layout(
            mapbox_style="open-street-map"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.warning("KMeans_Cluster column not found.")

    # ===============================
    # DBSCAN HOTSPOTS
    # ===============================
    st.subheader("📍 DBSCAN Crime Hotspots")

    if "DBSCAN_Cluster" in df.columns:

        sample_df = df.sample(
            min(10000, len(df)),
            random_state=42
        )

        fig = px.scatter_mapbox(
            sample_df,
            lat="Latitude",
            lon="Longitude",
            color="DBSCAN_Cluster",
            zoom=9,
            height=700,
            title="DBSCAN Crime Hotspots"
        )

        fig.update_layout(
            mapbox_style="open-street-map"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.warning("DBSCAN_Cluster column not found.")

    # ===============================
    # HIERARCHICAL HOTSPOTS
    # ===============================
    st.subheader("📍 Hierarchical Crime Hotspots")

    if "Hierarchical_Cluster" in df.columns:

        sample_df = df.sample(
            min(10000, len(df)),
            random_state=42
        )

        fig = px.scatter_mapbox(
            sample_df,
            lat="Latitude",
            lon="Longitude",
            color="Hierarchical_Cluster",
            zoom=9,
            height=700,
            title="Hierarchical Crime Hotspots"
        )

        fig.update_layout(
            mapbox_style="open-street-map"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.warning("Hierarchical_Cluster column not found.")

    # ===============================
    # KMEANS DISTRIBUTION
    # ===============================
    st.subheader("📊 K-Means Cluster Distribution")

    if "KMeans_Cluster" in df.columns:

        cluster_counts = (
            df["KMeans_Cluster"]
            .value_counts()
            .reset_index()
        )

        cluster_counts.columns = [
            "Cluster",
            "Crime_Count"
        ]

        fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Crime_Count",
            color="Crime_Count",
            title="Crime Count Per K-Means Cluster"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

except Exception as e:
    st.error(f"Error loading clustering analysis: {e}")