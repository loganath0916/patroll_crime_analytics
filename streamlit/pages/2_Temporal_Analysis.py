import streamlit as st
import pandas as pd
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Temporal Crime Analysis",
    page_icon="⏰",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "crime_clustered_final.csv"
    return pd.read_csv(data_path)

df = load_data()

# ==================================================
# TITLE
# ==================================================

st.title("⏰ Temporal Crime Analysis")
st.markdown(
    "Analyze crime patterns across hours, months, seasons and temporal clusters."
)

st.divider()

# ==================================================
# KPI METRICS
# ==================================================

peak_hour = df["Hour"].value_counts().idxmax()

peak_month = df["Month"].value_counts().idxmax()

total_temporal_clusters = (
    df["Temporal_Cluster"].nunique()
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Peak Crime Hour",
        peak_hour
    )

with col2:
    st.metric(
        "Peak Month",
        peak_month
    )

with col3:
    st.metric(
        "Temporal Clusters",
        total_temporal_clusters
    )

st.divider()

# ==================================================
# TEMPORAL CLUSTER DISTRIBUTION
# ==================================================

st.subheader("📊 Temporal Cluster Distribution")

cluster_counts = (
    df["Temporal_Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_counts.columns = [
    "Temporal_Cluster",
    "Crime_Count"
]

fig = px.bar(
    cluster_counts,
    x="Temporal_Cluster",
    y="Crime_Count",
    text="Crime_Count",
    title="Crime Distribution Across Temporal Clusters"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==================================================
# HOURLY CRIME ANALYSIS
# ==================================================

st.subheader("🕒 Crime Pattern by Hour")

hourly_crime = (
    df.groupby("Hour")
    .size()
    .reset_index(name="Crime_Count")
)

fig = px.line(
    hourly_crime,
    x="Hour",
    y="Crime_Count",
    markers=True,
    title="Hourly Crime Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==================================================
# TOP 10 CRIME HOURS
# ==================================================

st.subheader("🔥 Top Crime Hours")

top_hours = (
    hourly_crime
    .sort_values(
        "Crime_Count",
        ascending=False
    )
    .head(10)
)

fig = px.bar(
    top_hours,
    x="Hour",
    y="Crime_Count",
    text="Crime_Count",
    title="Top 10 Crime Hours"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==================================================
# MONTHLY CRIME ANALYSIS
# ==================================================

st.subheader("📅 Monthly Crime Trend")

monthly_crime = (
    df.groupby("Month")
    .size()
    .reset_index(name="Crime_Count")
)

fig = px.line(
    monthly_crime,
    x="Month",
    y="Crime_Count",
    markers=True,
    title="Monthly Crime Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==================================================
# DAY OF WEEK ANALYSIS
# ==================================================

if "DayOfWeek" in df.columns:

    st.subheader("📆 Crime by Day of Week")

    day_crime = (
        df["DayOfWeek"]
        .value_counts()
        .reset_index()
    )

    day_crime.columns = [
        "Day",
        "Crime_Count"
    ]

    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    day_crime["Day"] = pd.Categorical(
        day_crime["Day"],
        categories=order,
        ordered=True
    )

    day_crime = day_crime.sort_values("Day")

    fig = px.bar(
        day_crime,
        x="Day",
        y="Crime_Count",
        title="Crime Distribution by Day"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ==================================================
# SEASONAL ANALYSIS
# ==================================================

if "Season" in df.columns:

    st.subheader("🌦 Seasonal Crime Analysis")

    seasonal_crime = (
        df.groupby("Season")
        .size()
        .reset_index(name="Crime_Count")
    )

    fig = px.pie(
        seasonal_crime,
        names="Season",
        values="Crime_Count",
        title="Seasonal Crime Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ==================================================
# WEEKEND VS WEEKDAY
# ==================================================

if "IsWeekend" in df.columns:

    st.subheader("📌 Weekend vs Weekday Crimes")

    weekend_data = (
        df.groupby("IsWeekend")
        .size()
        .reset_index(name="Crime_Count")
    )

    weekend_data["Type"] = (
        weekend_data["IsWeekend"]
        .map({
            0: "Weekday",
            1: "Weekend"
        })
    )

    fig = px.bar(
        weekend_data,
        x="Type",
        y="Crime_Count",
        text="Crime_Count",
        title="Weekend vs Weekday Crime Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ==================================================
# TEMPORAL CLUSTER PROFILE
# ==================================================

st.subheader("📋 Temporal Cluster Profiles")

profile = (
    df.groupby("Temporal_Cluster")
    [["Hour", "Month", "IsWeekend"]]
    .mean()
    .round(2)
)

st.dataframe(
    profile,
    use_container_width=True
)

st.divider()

# ==================================================
# INSIGHTS
# ==================================================

st.subheader("🎯 Key Temporal Insights")

st.success("""
✅ Temporal crime patterns successfully identified

✅ Peak crime hours discovered

✅ Monthly crime trends analyzed

✅ Seasonal crime behavior identified

✅ Weekend and weekday patterns compared

✅ Temporal clusters support patrol planning and resource allocation
""")