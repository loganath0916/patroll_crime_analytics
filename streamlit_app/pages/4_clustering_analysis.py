import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Crime Clustering Analysis",
    layout="wide"
)

st.title("🎯 Crime Clustering Analysis")

# Load Data
df = pd.read_csv("../data/crime_feature_engineered.csv")

# =====================================================
# KMEANS CLUSTERS
# =====================================================

st.header("📍 K-Means Crime Hotspots")

if "KMeans_Cluster" in df.columns:

    sample_df = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    fig_kmeans = px.scatter_mapbox(
        sample_df,
        lat="Latitude",
        lon="Longitude",
        color="KMeans_Cluster",
        hover_data=["Primary Type"],
        zoom=9,
        height=600,
        title="K-Means Crime Hotspots"
    )

    fig_kmeans.update_layout(
        mapbox_style="open-street-map"
    )

    st.plotly_chart(
        fig_kmeans,
        use_container_width=True
    )

else:
    st.warning("KMeans_Cluster column not found.")

# =====================================================
# DBSCAN CLUSTERS
# =====================================================

st.header("📍 DBSCAN Crime Hotspots")

if "DBSCAN_Cluster" in df.columns:

    sample_df = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    fig_dbscan = px.scatter_mapbox(
        sample_df,
        lat="Latitude",
        lon="Longitude",
        color="DBSCAN_Cluster",
        hover_data=["Primary Type"],
        zoom=9,
        height=600,
        title="DBSCAN Crime Clusters"
    )

    fig_dbscan.update_layout(
        mapbox_style="open-street-map"
    )

    st.plotly_chart(
        fig_dbscan,
        use_container_width=True
    )

else:
    st.warning("DBSCAN_Cluster column not found.")

# =====================================================
# HIERARCHICAL CLUSTERS
# =====================================================

st.header("📍 Hierarchical Crime Hotspots")

if "Hierarchical_Cluster" in df.columns:

    sample_df = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    fig_hc = px.scatter_mapbox(
        sample_df,
        lat="Latitude",
        lon="Longitude",
        color="Hierarchical_Cluster",
        hover_data=["Primary Type"],
        zoom=9,
        height=600,
        title="Hierarchical Crime Clusters"
    )

    fig_hc.update_layout(
        mapbox_style="open-street-map"
    )

    st.plotly_chart(
        fig_hc,
        use_container_width=True
    )

else:
    st.info("""
    Hierarchical Clustering was performed on a representative sample
    of crime locations because Agglomerative Clustering is computationally
    expensive for 500K+ records.

    See clustering_analysis.ipynb for the generated dendrogram and evaluation metrics.
    """)

# =====================================================
# CLUSTER DISTRIBUTION
# =====================================================

st.header("📊 K-Means Cluster Distribution")

if "KMeans_Cluster" in df.columns:

    cluster_dist = (
        df["KMeans_Cluster"]
        .value_counts()
        .reset_index()
    )

    cluster_dist.columns = [
        "Cluster",
        "Crime_Count"
    ]

    fig_dist = px.bar(
        cluster_dist,
        x="Cluster",
        y="Crime_Count",
        color="Crime_Count",
        title="Crime Count per Cluster"
    )

    st.plotly_chart(
        fig_dist,
        use_container_width=True
    )

# =====================================================
# ALGORITHM PERFORMANCE
# =====================================================

st.header("🏆 Algorithm Comparison")

comparison_df = pd.DataFrame({
    "Algorithm": [
        "KMeans",
        "Hierarchical",
        "DBSCAN"
    ],
    "Silhouette Score": [
        0.402,
        0.365,
        -0.322
    ],
    "Davies-Bouldin Index": [
        0.789,
        0.807,
        1.231
    ]
})

st.dataframe(
    comparison_df,
    use_container_width=True
)

fig_compare = px.bar(
    comparison_df,
    x="Algorithm",
    y="Silhouette Score",
    color="Algorithm",
    title="Silhouette Score Comparison"
)

st.plotly_chart(
    fig_compare,
    use_container_width=True
)

# =====================================================
# BEST MODEL
# =====================================================

st.success("""
🏆 Best Performing Model

Algorithm: K-Means

Silhouette Score: 0.402

Davies-Bouldin Index: 0.789

Selected for final deployment and hotspot detection.
""")