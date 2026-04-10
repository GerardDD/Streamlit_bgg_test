
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🎮 Estadístiques de Partides")

# Load df_2 (plays)
df_2 = pd.read_csv("pages/playsMrbrussels.csv", sep=",", engine="python")

# Clean Players column (remove numbers and brackets)
df_2["Players"] = (
    df_2["Players"]
    .str.replace(r"\(\d+\)", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Convert date if exists
if "Date" in df_2.columns:
    df_2["Date"] = pd.to_datetime(df_2["Date"], errors="coerce")

st.header("📌 KPIs principals")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎲 Total de partides registrades", len(df_2))

with col2:
    if "Name" in df_2.columns:
        st.metric("🧩 Jocs diferents jugats", df_2["Name"].nunique())
    else:
        st.metric("🧩 Jocs diferents jugats", "—")

with col3:
    if "Players" in df_2.columns:
        total_players = df_2["Players"].str.split(",").explode().str.strip().nunique()
        st.metric("👥 Jugadors diferents", total_players)
    else:
        st.metric("👥 Jugadors diferents", "—")

st.subheader("📅 Evolució de partides en el temps")

if "Date" in df_2.columns:
    df_time = df_2.groupby(df_2["Date"].dt.to_period("M")).size().reset_index(name="count")
    df_time["Date"] = df_time["Date"].dt.to_timestamp()

    fig_time = px.line(df_time, x="Date", y="count", title="Partides per mes")
    st.plotly_chart(fig_time, use_container_width=True)
else:
    st.info("No hi ha columna 'Date' al CSV.")

st.subheader("🏆 Jocs més jugats")

if "Name" in df_2.columns:
    top_games = df_2["Name"].value_counts().head(10).reset_index()
    top_games.columns = ["Name", "count"]

    fig_games = px.bar(
        top_games,
        x="count",
        y="Name",
        orientation="h",
        title="Top 10 jocs més jugats",
        text="count"
    )
    fig_games.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_games, use_container_width=True)

st.subheader("👤 Jugadors més actius")

if "Players" in df_2.columns:
    players = (
        df_2["Players"]
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(10)
        .reset_index()
    )
    players.columns = ["Player", "count"]

    fig_players = px.bar(
        players,
        x="count",
        y="Player",
        orientation="h",
        title="Top 10 jugadors més actius",
        text="count"
    )
    fig_players.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_players, use_container_width=True)
