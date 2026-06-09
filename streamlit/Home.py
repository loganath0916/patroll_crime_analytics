import streamlit as st
import pandas as pd
from pathlib import Path

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="PATROLL Crime Analytics",
    page_icon="🚔",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "crime_clustered_final.csv"
    )

    return pd.read_csv(data_path)

df = load_data()

# ==================================================
# TITLE
# ==================================================

st.title("🚔 PATROLL Crime Analytics Platform")

st.markdown("""
### Intelligent Crime Pattern Discovery using Machine Learning

This project analyzes Chicago crime data using:

- Geographic Hotspot Clustering
- Temporal Crime Pattern Analysis
- PCA Dimensionality Reduction
- t-SNE Visualization
- MLflow Experiment Tracking

Navigate using the sidebar to explore the analysis.
""")

st.divider()

# ==================================================
# DATASET OVERVIEW
# ==================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Features",
        df.shape[1]
    )

with col3:
    st.metric(
        "Crime Types",
        df["Primary Type"].nunique()
        if "Primary Type" in df.columns
        else "N/A"
    )

with col4:
    st.metric(
        "Districts",
        df["District"].nunique()
        if "District" in df.columns
        else "N/A"
    )

st.divider()

# ==================================================
# DATA PREVIEW
# ==================================================

st.subheader("🔍 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.divider()

# ==================================================
# PROJECT OBJECTIVES
# ==================================================

st.subheader("🎯 Project Objectives")

st.markdown("""
### Geographic Analysis
- Identify crime hotspots using clustering.
- Detect high-risk zones.
- Discover spatial crime patterns.

### Temporal Analysis
- Identify peak crime hours.
- Analyze seasonal crime trends.
- Discover weekend vs weekday patterns.

### Dimensionality Reduction
- Reduce feature dimensions using PCA.
- Retain maximum variance.
- Visualize hidden crime patterns using t-SNE.

### MLflow Integration
- Track experiments.
- Compare clustering algorithms.
- Select best model for deployment.
""")

st.divider()

# ==================================================
# PROJECT PIPELINE
# ==================================================

st.subheader("⚙️ Project Workflow")

st.markdown("""
1. Data Collection & Cleaning

2. Exploratory Data Analysis (EDA)

3. Feature Engineering

4. Geographic Clustering
   - KMeans
   - DBSCAN
   - Hierarchical Clustering

5. Temporal Pattern Clustering

6. PCA Dimensionality Reduction

7. t-SNE Visualization

8. MLflow Experiment Tracking

9. Streamlit Dashboard Development
""")

st.divider()

# ==================================================
# SUCCESS MESSAGE
# ==================================================

st.success(
    "PATROLL Crime Analytics Platform Successfully Loaded."
)