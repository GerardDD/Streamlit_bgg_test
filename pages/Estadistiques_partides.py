import streamlit as st
import pandas as pd
import plotly.express as px
from itertools import combinations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date

st.set_page_config(layout="wide")

st.title("🎮 Estadístiques de Partides")

# ============================
# LOAD DATA
# ============================

# Load bgg csv
df = pd.read_csv("pages/collection.csv", sep=",", engine="python")

# Load df_2 (plays)
df_2 = pd.read_csv("pages/playsMrbrussels.csv", sep=",", engine="python")

# Assegurem que la columna Players existeix
if "Players" not in df_2.columns:
    df_2["Players"] = ""

# Clean Players column (remove numbers and brackets)
df_2["Players"] = (
    df_2["Players"]
    .astype(str)
    .str.replace(r"\(\d+\)", "", regex=True)
    .str.replace(r"\(-\d+\)", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Convert Date if exists
if "Date" in df_2.columns:
    df_2["Date"] = pd.to_datetime(df_2["Date"], errors="coerce")
    df_2 = df_2.dropna(subset=["Date"])

# ============================
# 🔵 SIDEBAR FILTERS
# ============================

st.sidebar.header("Filtres")

# --- Date filter ---
if "Date" in df_2.columns and not df_2["Date"].isna().all():
    # Convert and clean dates (per si de cas)
    df_2["Date"] = pd.to_datetime(df_2["Date"], errors="coerce")
    df_2 = df_2.dropna(subset=["Date"])

    data_min = df_2["Date"].min().date()
    data_max = df_2["Date"].max().date()

    today = date.today()
    picker_max = max(data_max, today)

    date_range = st.sidebar.date_input(
        "Rang de dates:",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=picker_max
    )
else:
    date_range = None
    data_min = None
    data_max = None

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

# Filter by date (robust + clamp)
if date_range is not None and data_min is not None and data_max is not None:
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:
        start = end = date_range

    start = max(start, data_min)
    end = min(end, data_max)

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    df_filtered = df_filtered[
        (df_filtered["Date"] >= start) &
        (df_filtered["Date"] <= end)
    ]

# Filter by players
if players_sel and "Players" in df_filtered.columns:
    df_filtered = df_filtered[
        df_filtered["Players"].astype(str).apply(
            lambda x: any(p.strip() in players_sel for p in x.split(","))
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
    if game_col and not df_filtered.empty:
        st.metric("🧩 Jocs diferents jugats", df_filtered[game_col].nunique())
    else:
        st.metric("🧩 Jocs diferents jugats", "—")

with col3:
    if "Players" in df_filtered.columns and not df_filtered.empty:
        total_players = (
            df_filtered["Players"]
            .str.split(",")
            .explode()
            .str.strip()
            .nunique()
        )
        st.metric("👥 Jugadors diferents", total_players)
    else:
        st.metric("👥 Jugadors diferents", "—")

# ============================
# 🔵 VISUALITZACIONS (all use df_filtered)
# ============================

st.subheader("📅 Evolució de partides en el temps")

if "Date" in df_filtered.columns and not df_filtered.empty:

    days_range = (df_filtered["Date"].max() - df_filtered["Date"].min()).days

    if days_range <= 31:
        df_time = df_filtered.groupby(df_filtered["Date"].dt.date).size().reset_index(name="count")
        df_time["Date"] = pd.to_datetime(df_time["Date"])
        tick_format = "%d/%m"
    elif days_range <= 370:
        df_time = df_filtered.groupby(df_filtered["Date"].dt.to_period("M")).size().reset_index(name="count")
        df_time["Date"] = df_time["Date"].dt.to_timestamp()
        tick_format = "%m/%Y"
    else:
        df_time = df_filtered.groupby(df_filtered["Date"].dt.to_period("Y")).size().reset_index(name="count")
        df_time["Date"] = df_time["Date"].dt.to_timestamp()
        tick_format = "%Y"

    fig_time = px.line(
        df_time,
        x="Date",
        y="count",
        markers=True,
        title="Evolució de partides en el temps"
    )

    fig_time.update_xaxes(
        tickformat=tick_format,
        nticks=12,
        showgrid=True
    )

    fig_time.update_traces(line=dict(width=3))

    st.plotly_chart(fig_time, use_container_width=True)

st.subheader("🏆 Jocs més jugats")

if game_col and not df_filtered.empty:
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

if "Players" in df_filtered.columns and not df_filtered.empty:
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
else:
    st.info("No hi ha dades de jugadors per mostrar el rànquing.")

# ============================
# 🔵 HEATMAP: Jugadors que juguen junts (només jugadors filtrats)
# ============================

st.subheader("🤝 Jugadors que juguen junts (Heatmap)")

if players_sel and "Players" in df_filtered.columns and not df_filtered.empty:
    pairs_df = (
        df_filtered["Players"]
        .str.split(",")
        .apply(lambda lst: [p.strip() for p in lst if p.strip()])
    )

    pairs_df = pairs_df.apply(lambda lst: [p for p in lst if p in players_sel])

    pair_counts = {}

    for players_list in pairs_df:
        if len(players_list) > 1:
            for p1, p2 in combinations(sorted(players_list), 2):
                pair_counts[(p1, p2)] = pair_counts.get((p1, p2), 0) + 1

    all_players = sorted(players_sel)
    matrix = pd.DataFrame(0, index=all_players, columns=all_players)

    for (p1, p2), count in pair_counts.items():
        matrix.loc[p1, p2] = count
        matrix.loc[p2, p1] = count

    for p in all_players:
        matrix.loc[p, p] = sum(p in lst for lst in pairs_df)

    if matrix.values.sum() > 0:
        fig_heatmap = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Freqüència de jugadors que coincideixen en partides (filtrat)"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No hi ha prou dades per generar el heatmap amb els jugadors filtrats.")
else:
    st.info("Selecciona algun jugador i assegura’t que hi hagi dades filtrades per veure el heatmap.")


