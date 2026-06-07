import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🗺️ Geographic Crime Analysis")

# Load data
df = pd.read_csv("../data/crime_feature_engineered.csv")

# Check columns
st.write("Dataset Shape:", df.shape)

# Sidebar filters
districts = st.sidebar.multiselect(
    "Select District",
    options=sorted(df["District"].dropna().unique()),
    default=sorted(df["District"].dropna().unique())[:5]
)

filtered_df = df[df["District"].isin(districts)]

st.subheader("Crime Locations")

# Interactive Map
fig = px.scatter_mapbox(
    filtered_df.sample(min(5000, len(filtered_df))),
    lat="Latitude",
    lon="Longitude",
    color="Primary Type",
    zoom=10,
    height=700,
    title="Crime Distribution Across Chicago"
)

fig.update_layout(
    mapbox_style="open-street-map",
    margin=dict(l=0, r=0, t=40, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# Top Crime Areas
st.subheader("Top Crime Districts")

district_count = (
    filtered_df["District"]
    .value_counts()
    .reset_index()
)

district_count.columns = ["District", "Crime Count"]

bar_fig = px.bar(
    district_count.head(10),
    x="District",
    y="Crime Count",
    title="Top 10 Crime Districts"
)

st.plotly_chart(bar_fig, use_container_width=True)

# Cluster Visualization
if "KMeans_Cluster" in filtered_df.columns:

    st.subheader("K-Means Crime Clusters")

    cluster_fig = px.scatter_mapbox(
        filtered_df.sample(min(5000, len(filtered_df))),
        lat="Latitude",
        lon="Longitude",
        color="KMeans_Cluster",
        zoom=10,
        height=700,
        title="Crime Hotspot Clusters"
    )

    cluster_fig.update_layout(
        mapbox_style="open-street-map"
    )

    st.plotly_chart(cluster_fig, use_container_width=True)

else:
    st.warning(
        "KMeans_Cluster column not found. Run clustering notebook first."
    )