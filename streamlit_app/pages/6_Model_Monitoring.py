import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Monitoring Dashboard")

# ==========================================
# MODEL COMPARISON
# ==========================================

st.subheader("🏆 Clustering Model Comparison")

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

# ==========================================
# SILHOUETTE SCORE
# ==========================================

st.subheader("📈 Silhouette Score Comparison")

fig_silhouette = px.bar(
    comparison_df,
    x="Algorithm",
    y="Silhouette Score",
    color="Algorithm",
    title="Silhouette Score Comparison"
)

st.plotly_chart(
    fig_silhouette,
    use_container_width=True
)

# ==========================================
# DAVIES BOULDIN
# ==========================================

st.subheader("📉 Davies-Bouldin Index Comparison")

fig_db = px.bar(
    comparison_df,
    x="Algorithm",
    y="Davies-Bouldin Index",
    color="Algorithm",
    title="Davies-Bouldin Index Comparison"
)

st.plotly_chart(
    fig_db,
    use_container_width=True
)

# ==========================================
# BEST MODEL
# ==========================================

st.subheader("🥇 Best Model Selection")

st.success("""
Best Performing Model: KMeans

Silhouette Score: 0.402

Davies-Bouldin Index: 0.789

Selected as the final production model for crime hotspot detection.
""")

# ==========================================
# PCA PERFORMANCE
# ==========================================

st.subheader("🔍 Dimensionality Reduction Performance")

pca_df = pd.DataFrame({
    "Component": ["PC1", "PC2", "PC3"],
    "Variance Explained (%)": [
        89.14,
        5.41,
        2.39
    ]
})

fig_pca = px.bar(
    pca_df,
    x="Component",
    y="Variance Explained (%)",
    color="Component",
    title="PCA Explained Variance"
)

st.plotly_chart(
    fig_pca,
    use_container_width=True
)

# ==========================================
# PROJECT SUMMARY
# ==========================================

st.subheader("📋 Project Summary")

summary = pd.DataFrame({
    "Task": [
        "Data Preprocessing",
        "Feature Engineering",
        "KMeans Clustering",
        "DBSCAN Clustering",
        "Hierarchical Clustering",
        "PCA",
        "t-SNE",
        "MLflow Tracking",
        "Streamlit Deployment"
    ],
    "Status": [
        "Completed",
        "Completed",
        "Completed",
        "Completed",
        "Completed",
        "Completed",
        "Completed",
        "Completed",
        "Completed"
    ]
})

st.dataframe(
    summary,
    use_container_width=True
)

# ==========================================
# FINAL INSIGHTS
# ==========================================

st.subheader("📌 Final Insights")

st.info("""
• Geographic location is the strongest factor influencing crime patterns.

• PCA retained over 96% of the original variance.

• KMeans achieved the best clustering performance.

• Crime hotspots were successfully identified across Chicago.

• Interactive dashboards enable exploration of spatial and temporal crime trends.
""")