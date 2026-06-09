import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Dimensionality Reduction",
    page_icon="📉",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv("../data/crime_clustered_final.csv")

df = load_data()

# ==================================================
# TITLE
# ==================================================

st.title("📉 Dimensionality Reduction")
st.markdown(
    "PCA and t-SNE Analysis for Crime Pattern Visualization"
)

st.divider()

# ==================================================
# FEATURES USED
# ==================================================

features = [
    "Hour",
    "Month",
    "IsWeekend",
    "Crime_Severity_Score",
    "Crime_Type_Encoded",
    "Location_Encoded",
    "District_Encoded",
    "Latitude_Scaled",
    "Longitude_Scaled"
]

X = df[features]

# ==================================================
# STANDARDIZATION
# ==================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==================================================
# PCA
# ==================================================

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

components_needed = (
    np.argmax(cumulative_variance >= 0.70) + 1
)

variance_retained = (
    cumulative_variance[components_needed - 1]
)

# ==================================================
# KPI METRICS
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Features Used",
        len(features)
    )

with col2:
    st.metric(
        "Components Selected",
        components_needed
    )

with col3:
    st.metric(
        "Variance Retained",
        f"{variance_retained:.2%}"
    )

st.divider()

# ==================================================
# VARIANCE EXPLAINED
# ==================================================

st.subheader("📊 PCA Variance Explained")

variance_df = pd.DataFrame({
    "Components": range(
        1,
        len(cumulative_variance)+1
    ),
    "Cumulative Variance":
    cumulative_variance
})

fig = px.line(
    variance_df,
    x="Components",
    y="Cumulative Variance",
    markers=True,
    title="Cumulative Variance Explained"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==================================================
# PCA SCATTER
# ==================================================

st.subheader("🎯 PCA 2D Visualization")

pca_2d = PCA(
    n_components=2
)

X_pca_2d = pca_2d.fit_transform(
    X_scaled
)

pca_df = pd.DataFrame(
    X_pca_2d,
    columns=["PC1", "PC2"]
)

if "Temporal_Cluster" in df.columns:
    pca_df["Cluster"] = (
        df["Temporal_Cluster"]
    )

fig = px.scatter(
    pca_df.sample(
        min(5000, len(pca_df)),
        random_state=42
    ),
    x="PC1",
    y="PC2",
    color="Cluster",
    title="PCA Projection of Crime Data"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

st.subheader("🏆 Feature Importance")

pca_final = PCA(
    n_components=components_needed
)

pca_final.fit(
    X_scaled
)

loadings = pd.DataFrame(
    pca_final.components_.T,
    index=features
)

feature_importance = (
    loadings.abs()
    .sum(axis=1)
    .sort_values(
        ascending=False
    )
)

importance_df = (
    feature_importance
    .reset_index()
)

importance_df.columns = [
    "Feature",
    "Importance"
]

fig = px.bar(
    importance_df,
    x="Feature",
    y="Importance",
    title="Feature Importance Ranking"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    importance_df,
    use_container_width=True
)

st.divider()

# ==================================================
# TOP 5 FEATURES
# ==================================================

st.subheader(
    "⭐ Top 5 Important Features"
)

st.table(
    importance_df.head(5)
)

st.divider()

# ==================================================
# t-SNE
# ==================================================

st.subheader(
    "🧠 t-SNE Visualization"
)

sample_df = df.sample(
    min(3000, len(df)),
    random_state=42
)

X_tsne = sample_df[
    features
]

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

X_tsne_2d = tsne.fit_transform(
    X_tsne
)

tsne_df = pd.DataFrame(
    X_tsne_2d,
    columns=[
        "TSNE1",
        "TSNE2"
    ]
)

if "Temporal_Cluster" in sample_df.columns:
    tsne_df["Cluster"] = (
        sample_df["Temporal_Cluster"]
    )

fig = px.scatter(
    tsne_df,
    x="TSNE1",
    y="TSNE2",
    color="Cluster",
    title="t-SNE Crime Pattern Visualization"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==================================================
# INSIGHTS
# ==================================================

st.subheader("🎯 Key Findings")

st.success(
    """
    ✅ PCA successfully reduced dimensionality

    ✅ More than 70% variance retained

    ✅ Most influential crime-driving features identified

    ✅ PCA 2D visualization generated

    ✅ t-SNE visualization generated

    ✅ Crime patterns visualized in lower-dimensional space
    """
)