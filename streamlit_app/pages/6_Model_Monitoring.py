import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow

st.set_page_config(
    page_title="MLflow Monitoring",
    layout="wide"
)

st.title("📊 MLflow Model Monitoring")

# ==========================================
# MODEL COMPARISON
# ==========================================

st.header("🏆 Clustering Model Comparison")

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
# SILHOUETTE SCORE COMPARISON
# ==========================================

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
# DB INDEX COMPARISON
# ==========================================

fig_dbi = px.bar(
    comparison_df,
    x="Algorithm",
    y="Davies-Bouldin Index",
    color="Algorithm",
    title="Davies-Bouldin Index Comparison"
)

st.plotly_chart(
    fig_dbi,
    use_container_width=True
)

# ==========================================
# BEST MODEL
# ==========================================

st.header("🥇 Selected Model")

st.success("""
Best Model: KMeans

Silhouette Score: 0.402

Davies-Bouldin Index: 0.789

Selected for crime hotspot detection and deployment.
""")

# ==========================================
# PCA PERFORMANCE
# ==========================================

st.header("📈 PCA Performance")

variance_df = pd.DataFrame({
    "Component": ["PC1", "PC2", "PC3"],
    "Variance": [89.14, 5.41, 2.39]
})

fig_var = px.bar(
    variance_df,
    x="Component",
    y="Variance",
    color="Component",
    title="PCA Explained Variance"
)

st.plotly_chart(
    fig_var,
    use_container_width=True
)

st.success(
    "Total Variance Explained: 96.94%"
)

# ==========================================
# MLFLOW DETAILS
# ==========================================

st.header("🔬 Experiment Tracking")

st.info("""
MLflow was used for:

• Experiment Tracking

• Parameter Logging

• Metric Tracking

• Model Comparison

• Version Management

• Best Model Selection
""")

# ==========================================
# EXPERIMENT SUMMARY
# ==========================================

st.header("📋 Experiment Summary")

summary_df = pd.DataFrame({
    "Experiment": [
        "KMeans_K8",
        "DBSCAN",
        "Hierarchical",
        "PCA"
    ],
    "Status": [
        "Completed",
        "Completed",
        "Completed",
        "Completed"
    ]
})

st.dataframe(
    summary_df,
    use_container_width=True
)

# ==========================================
# MLFLOW LINK
# ==========================================

st.header("🌐 MLflow Dashboard")

st.markdown(
    "[Open MLflow Dashboard](http://127.0.0.1:5000)"
)

st.success(
    "All experiments successfully tracked using MLflow."
)