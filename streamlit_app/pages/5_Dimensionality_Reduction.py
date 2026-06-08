import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Dimensionality Reduction",
    page_icon="🔍",
    layout="wide"
)

# File Paths
BASE_DIR = Path(__file__).parents[2]

PCA_PATH = BASE_DIR / "data" / "pca_results.csv"
TSNE_PATH = BASE_DIR / "data" / "tsne_results.csv"
VAR_PATH = BASE_DIR / "data" / "pca_variance.csv"
IMP_PATH = BASE_DIR / "data" / "pca_feature_importance.csv"

try:

    st.title("🔍 Dimensionality Reduction Analysis")

    # ==========================
    # PCA VISUALIZATION
    # ==========================

    st.subheader("📊 PCA Visualization")

    pca_df = pd.read_csv(PCA_PATH)

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        title="PCA Crime Pattern Visualization"
    )

    st.plotly_chart(
        fig_pca,
        use_container_width=True
    )

    # ==========================
    # PCA VARIANCE
    # ==========================

    st.subheader("📈 Explained Variance")

    variance_df = pd.read_csv(VAR_PATH)

    fig_var = px.bar(
        variance_df,
        x="Component",
        y="Variance",
        color="Component",
        title="Explained Variance by PCA Components"
    )

    st.plotly_chart(
        fig_var,
        use_container_width=True
    )

    # ==========================
    # FEATURE IMPORTANCE
    # ==========================

    st.subheader("🏆 Top Features Driving Crime Patterns")

    importance_df = pd.read_csv(IMP_PATH)

    fig_imp = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importance"
    )

    st.plotly_chart(
        fig_imp,
        use_container_width=True
    )

    # ==========================
    # TSNE VISUALIZATION
    # ==========================

    st.subheader("🎯 t-SNE Visualization")

    tsne_df = pd.read_csv(TSNE_PATH)

    fig_tsne = px.scatter(
        tsne_df,
        x="TSNE1",
        y="TSNE2",
        title="t-SNE Crime Pattern Visualization"
    )

    st.plotly_chart(
        fig_tsne,
        use_container_width=True
    )

    # ==========================
    # KEY INSIGHTS
    # ==========================

    st.subheader("📌 Key Insights")

    st.success("""
    ✅ PCA reduced high-dimensional crime data into 3 principal components.

    ✅ More than 90% of variance was retained.

    ✅ Location and Crime Type were the strongest factors influencing crime patterns.

    ✅ t-SNE revealed clear cluster separations and hidden structures in crime data.
    """)

except Exception as e:
    st.error(f"Error loading dimensionality reduction results: {e}")