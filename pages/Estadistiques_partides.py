import streamlit as st
import pandas as pd
import plotly.express as px
from itertools import combinations
from datetime import date, timedelta

st.set_page_config(layout="wide")
st.title("🎮 Estadístiques de Partides")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("pages/collection.csv", sep=",", engine="python")
df_2 = pd.read_csv("pages/playsMrbrussels.csv", sep=",", engine="python")

# Ensure Players column exists
df_2["Players"] = df_2.get("Players", "").astype(str)

# Clean Players
df_2["Players"] = (
    df_2["Players"]
    .str.replace(r"\(\d+\)", "", regex=True)
    .str.replace(r"\(-\d+\)", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Clean Date
df_2["Date"] = pd.to_datetime(df_2["Date"], errors="coerce")
df_2 = df_2.dropna(subset=["Date"])

# ============================================================
# 🔵 SIDEBAR FILTERS (UX MILLORADA)
# ============================================================

st.sidebar.header("Filtres")

# --- Date presets ---
preset = st.sidebar.selectbox(
    "Període:",
    ["Tot", "Última setmana", "Últim mes", "Últim any", "Personalitzat"]
)

data_min = df_2["Date"].min().date()
data_max = df_2["Date"].max().date()
today = date.today()

if preset == "Tot":
    start, end = data_min, data_max

elif preset == "Última setmana":
    start, end = today - timedelta(days=7), today

elif preset == "Últim mes":
    start, end = today - timedelta(days=30), today

elif preset == "Últim any":
    start, end = today - timedelta(days=365), today

else:
    # Personalitzat
    start, end = st.sidebar.date_input(
        "Rang de dates:",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=max(today, data_max)
    )

# Clamp to dataset
start = max(start, data_min)
end = min(end, data_max)

start = pd.to_datetime(start)
end = pd.to_datetime(end)

# --- Players filter ---
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

# --- Grouping selector ---
grouping = st.sidebar.radio(
    "Agrupa per:",
    ["Dia", "Setmana", "Mes", "Any"],
    horizontal=True
)

# ============================================================
# 🔵 APPLY FILTERS
# ============================================================

df_filtered = df_2[
    (df_2["Date"] >= start) &
    (df_2["Date"] <= end)
].copy()

if players_sel:
    df_filtered = df_filtered[
        df_filtered["Players"].str.contains("|".join(players_sel))
    ]

# ============================================================
# 🔵 KPIs
# ============================================================

st.header("📌 KPIs principals")

col1, col2, col3 = st.columns(3)

col1.metric("🎲 Total de partides", len(df_filtered))

game_col = next((c for c in df_filtered.columns if "Name" in c or "object" in c.lower()), None)

if game_col:
    col2.metric("🧩 Jocs diferents", df_filtered[game_col].nunique())
else:
    col2.metric("🧩 Jocs diferents", "—")

col3.metric(
    "👥 Jugadors diferents",
    df_filtered["Players"].str.split(",").explode().str.strip().nunique()
)

# ============================================================
# 🔵 EVOLUCIÓ TEMPORAL (GRÀFIC NET)
# ============================================================

st.subheader("📅 Evolució de partides en el temps")

if df_filtered.empty:
    st.info("No hi ha dades per mostrar en aquest rang.")
else:
    if grouping == "Dia":
        df_time = df_filtered.groupby(df_filtered["Date"].dt.date).size().reset_index(name="count")
        df_time["Date"] = pd.to_datetime(df_time["Date"])
        tick_format = "%d/%m"

    elif grouping == "Setmana":
        df_time = df_filtered.groupby(df_filtered["Date"].dt.to_period("W")).size().reset_index(name="count")
        df_time["Date"] = df_time["Date"].dt.start_time
        tick_format = "%d/%m"

    elif grouping == "Mes":
        df_time = df_filtered.groupby(df_filtered["Date"].dt.to_period("M")).size().reset_index(name="count")
        df_time["Date"] = df_time["Date"].dt.to_timestamp()
        tick_format = "%m/%Y"

    else:  # Any
        df_time = df_filtered.groupby(df_filtered["Date"].dt.to_period("Y")).size().reset_index(name="count")
        df_time["Date"] = df_time["Date"].dt.to_timestamp()
        tick_format = "%Y"

    fig_time = px.line(
        df_time,
        x="Date",
        y="count",
        markers=True,
        title=f"Evolució de partides ({grouping.lower()})"
    )

    fig_time.update_xaxes(tickformat=tick_format, nticks=12)
    fig_time.update_traces(line=dict(width=3), hovertemplate="Data: %{x}<br>Partides: %{y}")

    st.plotly_chart(fig_time, use_container_width=True)

# ============================================================
# 🔵 TOP JOCS
# ============================================================

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

# ============================================================
# 🔵 TOP JUGADORS
# ============================================================

st.subheader("👤 Jugadors més actius")

if not df_filtered.empty:
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

# ============================================================
# 🔵 HEATMAP
# ============================================================

st.subheader("🤝 Jugadors que juguen junts (Heatmap)")

if players_sel and not df_filtered.empty:
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
            title="Freqüència de jugadors que coincideixen en partides"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No hi ha prou dades per generar el heatmap.")
