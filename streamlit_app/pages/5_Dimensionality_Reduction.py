import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Dimensionality Reduction",
    layout="wide"
)

st.title("🔍 Dimensionality Reduction Analysis")

# =====================================================
# PCA VISUALIZATION
# =====================================================

st.header("📊 PCA Visualization")

pca_df = pd.read_csv("../data/pca_results.csv")

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

# =====================================================
# EXPLAINED VARIANCE
# =====================================================

st.header("📈 PCA Explained Variance")

variance_df = pd.DataFrame({
    "Component": ["PC1", "PC2", "PC3"],
    "Variance (%)": [89.14, 5.41, 2.39]
})

fig_var = px.bar(
    variance_df,
    x="Component",
    y="Variance (%)",
    color="Component",
    title="Variance Explained by Principal Components"
)

st.plotly_chart(
    fig_var,
    use_container_width=True
)

st.success(
    "Total Variance Explained = 96.94%"
)

# =====================================================
# TOP FEATURES
# =====================================================

st.header("🏆 Top Features Driving Crime Patterns")

importance_df = pd.DataFrame({
    "Feature": [
        "Location_Encoded",
        "Crime_Type_Encoded",
        "District_Encoded",
        "Latitude_Scaled",
        "Longitude_Scaled"
    ],
    "Importance": [
        0.9995,
        0.9935,
        0.8640,
        0.45,
        0.42
    ]
})

fig_imp = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top PCA Features"
)

st.plotly_chart(
    fig_imp,
    use_container_width=True
)

# =====================================================
# TSNE VISUALIZATION
# =====================================================

st.header("🎯 t-SNE Visualization")

tsne_df = pd.read_csv("../data/tsne_results.csv")

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

# =====================================================
# INSIGHTS
# =====================================================

st.header("📌 Key Insights")

st.info("""
• PCA reduced the dataset to 3 principal components.

• 96.94% of the original variance was retained.

• Location was the strongest driver of crime patterns.

• Crime Type and District were the next most influential factors.

• t-SNE revealed distinct groupings in high-dimensional crime data.
""")