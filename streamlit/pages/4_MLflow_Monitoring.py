import streamlit as st
import pandas as pd
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="MLflow Monitoring",
    page_icon="🔬",
    layout="wide"
)

# ==================================================
# TITLE
# ==================================================

st.title("🔬 MLflow Monitoring & Model Tracking")

st.markdown("""
Monitor experiment results, compare algorithms,
and review model performance metrics.
""")

st.divider()

# ==================================================
# BEST MODEL SECTION
# ==================================================

st.subheader("🏆 Best Model Selection")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Best Algorithm",
        "KMeans"
    )

with col2:
    st.metric(
        "Silhouette Score",
        "0.410"
    )

with col3:
    st.metric(
        "DBI",
        "0.773"
    )

st.divider()

# ==================================================
# ALGORITHM COMPARISON
# ==================================================

st.subheader("📊 Clustering Algorithm Comparison")

comparison_df = pd.DataFrame({
    "Algorithm": [
        "KMeans",
        "Hierarchical",
        "DBSCAN"
    ],
    "Silhouette Score": [
        0.410,
        0.359,
        0.276
    ],
    "Davies-Bouldin Index": [
        0.773,
        0.823,
        0.417
    ]
})

st.dataframe(
    comparison_df,
    use_container_width=True
)

# --------------------------------------------------

fig = px.bar(
    comparison_df,
    x="Algorithm",
    y="Silhouette Score",
    color="Algorithm",
    title="Silhouette Score Comparison"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------

fig = px.bar(
    comparison_df,
    x="Algorithm",
    y="Davies-Bouldin Index",
    color="Algorithm",
    title="Davies-Bouldin Index Comparison"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==================================================
# PCA TRACKING
# ==================================================

st.subheader("📉 PCA Experiment Tracking")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Components Selected",
        5
    )

with col2:
    st.metric(
        "Variance Retained",
        "70.4%"
    )

pca_df = pd.DataFrame({
    "Metric": [
        "Features Used",
        "Components Selected",
        "Variance Retained"
    ],
    "Value": [
        9,
        5,
        "70.4%"
    ]
})

st.dataframe(
    pca_df,
    use_container_width=True
)

st.divider()

# ==================================================
# MLFLOW RUNS
# ==================================================

st.subheader("🧪 Experiment Runs Logged")

runs_df = pd.DataFrame({
    "Run Name": [
        "KMeans_Geographic",
        "DBSCAN_Geographic",
        "Hierarchical_Geographic",
        "PCA_Analysis",
        "Best_KMeans_Model"
    ],
    "Status": [
        "Completed",
        "Completed",
        "Completed",
        "Completed",
        "Registered"
    ]
})

st.dataframe(
    runs_df,
    use_container_width=True
)

st.divider()

# ==================================================
# MODEL REGISTRY
# ==================================================

st.subheader("📦 Model Registry")

registry_df = pd.DataFrame({
    "Model Name": [
        "best_kmeans_model"
    ],
    "Version": [
        "1"
    ],
    "Stage": [
        "Production"
    ]
})

st.dataframe(
    registry_df,
    use_container_width=True
)

st.divider()

# ==================================================
# MLFLOW INFORMATION
# ==================================================

st.subheader("🌐 MLflow Dashboard")

st.info("""
MLflow was used for:

• Experiment Tracking

• Parameter Logging

• Metric Logging

• Model Comparison

• Model Registration

• Version Control
""")

st.code(
    "http://127.0.0.1:5000",
    language="text"
)

st.divider()

# ==================================================
# FINAL CONCLUSION
# ==================================================

st.subheader("🎯 Final Conclusion")

st.success("""
✅ KMeans selected as the final deployment model

✅ Highest Silhouette Score achieved

✅ Geographic hotspots successfully identified

✅ Temporal crime patterns discovered

✅ PCA retained 70.4% variance

✅ MLflow used for experiment tracking and model management

✅ PATROLL platform successfully developed
""")