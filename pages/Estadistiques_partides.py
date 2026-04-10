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
    .str.replace(r"\(-\d+\)", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Convert date if exists
if "Date" in df_2.columns:
    df_2["Date"] = pd.to_datetime(df_2["Date"], errors="coerce")

# ============================
# 🔵 SIDEBAR FILTERS
# ============================

st.sidebar.header("Filtres")

# --- Date filter ---
if "Date" in df_2.columns:
    min_date = df_2["Date"].min()
    max_date = df_2["Date"].max()

    date_range = st.sidebar.date_input(
        "Rang de dates:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    date_range = None

# --- Players filter ---
if "Players" in df_2.columns:
    all_players = (
        df_2["Players"]
        .str.split(",")
        .explode()
        .str.strip()
        .dropna()
        .unique()
    )

    players_sel = st.sidebar.multiselect(
        "Jugadors:",
        options=sorted(all_players),
        default=sorted(all_players)
    )
else:
    players_sel = None

# ============================
# 🔵 APPLY FILTERS
# ============================

df_filtered = df_2.copy()

# Filter by date
if date_range and len(date_range) == 2:
    start, end = date_range
    df_filtered = df_filtered[
        (df_filtered["Date"] >= pd.to_datetime(start)) &
        (df_filtered["Date"] <= pd.to_datetime(end))
    ]

# Filter by players
if players_sel:
    df_filtered = df_filtered[
        df_filtered["Players"].apply(
            lambda x: any(p.strip() in players_sel for p in str(x).split(","))
        )
    ]

# ============================
# 🔵 KPIs (all use df_filtered)
# ============================

st.header("📌 KPIs principals")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎲 Total de partides registrades", len(df_filtered))

# Detect game column automatically
game_col = [c for c in df_filtered.columns if "Name" in c or "object" in c.lower()]
game_col = game_col[0] if game_col else None

with col2:
    if game_col:
        st.metric("🧩 Jocs diferents jugats", df_filtered[game_col].nunique())
    else:
        st.metric("🧩 Jocs diferents jugats", "—")

with col3:
    total_players = (
        df_filtered["Players"]
        .str.split(",")
        .explode()
        .str.strip()
        .nunique()
    )
    st.metric("👥 Jugadors diferents", total_players)

# ============================
# 🔵 VISUALITZACIONS (all use df_filtered)
# ============================

st.subheader("📅 Evolució de partides en el temps")

if "Date" in df_filtered.columns:
    df_time = df_filtered.groupby(df_filtered["Date"].dt.to_period("M")).size().reset_index(name="count")
    df_time["Date"] = df_time["Date"].dt.to_timestamp()

    fig_time = px.line(df_time, x="Date", y="count", title="Partides per mes")
    st.plotly_chart(fig_time, use_container_width=True)

st.subheader("🏆 Jocs més jugats")

if game_col:
    top_games = df_filtered[game_col].value_counts().head(10).reset_index()
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

players = (
    df_filtered["Players"]
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
