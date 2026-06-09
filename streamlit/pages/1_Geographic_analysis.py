import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Geographic Crime Analysis",
    page_icon="🗺️",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    data_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "crime_clustered_final.csv"
    )

    return pd.read_csv(data_path)

df = load_data()

# ==================================================
# TITLE
# ==================================================

st.title("🗺️ Geographic Crime Analysis")
st.markdown(
    "Geographic crime hotspot analysis using clustering techniques."
)

st.divider()

# ==================================================
# KPI METRICS
# ==================================================

total_crimes = len(df)

total_clusters = (
    df["DBSCAN_Cluster"]
    .nunique()
)

noise_points = (
    (df["DBSCAN_Cluster"] == -1)
    .sum()
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Crimes",
        f"{total_crimes:,}"
    )

with col2:
    st.metric(
        "DBSCAN Clusters",
        total_clusters
    )

with col3:
    st.metric(
        "Noise Points",
        f"{noise_points:,}"
    )

st.divider()

# ==================================================
# CLUSTER DISTRIBUTION
# ==================================================

st.subheader("📊 Cluster Distribution")

cluster_counts = (
    df["DBSCAN_Cluster"]
    .value_counts()
    .sort_index()
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
    text="Crime_Count",
    title="Crime Count by DBSCAN Cluster"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==================================================
# HOTSPOT MAP
# ==================================================

st.subheader("🔥 Crime Hotspot Map")

sample_df = df.sample(
    min(5000, len(df)),
    random_state=42
)

fig = px.scatter_mapbox(
    sample_df,
    lat="Latitude",
    lon="Longitude",
    color="DBSCAN_Cluster",
    zoom=9,
    height=700,
    title="Geographic Crime Hotspots"
)

fig.update_layout(
    mapbox_style="open-street-map"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==================================================
# CLUSTER CENTER ANALYSIS
# ==================================================

st.subheader("📍 Cluster Centers")

cluster_centers = (
    df[df["DBSCAN_Cluster"] != -1]
    .groupby("DBSCAN_Cluster")
    [["Latitude", "Longitude"]]
    .mean()
    .reset_index()
)

st.dataframe(
    cluster_centers,
    use_container_width=True
)

st.divider()

# ==================================================
# CLUSTER CENTER MAP
# ==================================================

st.subheader("🗺️ Cluster Center Locations")

fig = px.scatter_mapbox(
    cluster_centers,
    lat="Latitude",
    lon="Longitude",
    color="DBSCAN_Cluster",
    zoom=9,
    height=600
)

fig.update_layout(
    mapbox_style="open-street-map"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==================================================
# TOP CRIME LOCATIONS
# ==================================================

if "Location Description" in df.columns:

    st.subheader("🏢 Top Crime Locations")

    top_locations = (
        df["Location Description"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_locations.columns = [
        "Location",
        "Crime_Count"
    ]

    fig = px.bar(
        top_locations,
        x="Crime_Count",
        y="Location",
        orientation="h",
        title="Top 10 Crime Locations"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ==================================================
# SUMMARY
# ==================================================

st.subheader("🎯 Geographic Insights")

st.success(
    """
    ✅ Geographic hotspots identified

    ✅ Crime concentration zones visualized

    ✅ Noise points detected by DBSCAN

    ✅ High-risk locations highlighted

    ✅ Spatial crime patterns analyzed

    ✅ Supports patrol planning and resource allocation
    """
)