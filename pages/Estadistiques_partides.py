import streamlit as st
import pandas as pd
import plotly.express as px
from itertools import combinations
from datetime import timedelta

from utils import (
    ACCENT,
    PLAYS_CSV,
    UPLOADED_PLAYS_KEY,
    forget_upload,
    load_csv,
    load_csv_bytes,
    remember_upload,
    split_players,
    style_plotly,
    uploaded_name,
)

st.set_page_config(page_title="Estadístiques de partides", page_icon="🎮", layout="wide")
st.title("🎮 Estadístiques de Partides")

# ============================================================
# LOAD DATA
# ============================================================


@st.cache_data(show_spinner=False)
def prepare_plays(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    if "Date" not in df.columns:
        st.error("⚠️ El CSV de partides no té la columna 'Date'.")
        st.stop()

    if "Players" not in df.columns:
        df["Players"] = ""
    if "Winner" not in df.columns:
        df["Winner"] = ""

    # Treure puntuacions "(12)" / "(-3)" i espais duplicats
    df["Players"] = (
        df["Players"].fillna("").astype(str)
        .str.replace(r"\(-?\d+\)", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Llistes de jugadors i guanyadors ("-" = guanyador no informat)
    df["player_list"] = df["Players"].apply(split_players)
    df["winner_list"] = df["Winner"].apply(
        lambda w: [] if str(w).strip() in ("", "-", "nan") else split_players(w)
    )
    return df


# --- Font de dades: CSV per defecte o CSV pujat (es manté en canviar de pàgina) ---
with st.sidebar.expander("📂 Dades de partides", expanded=bool(uploaded_name(UPLOADED_PLAYS_KEY))):
    fitxer_partides = st.file_uploader(
        "Puja un CSV de partides (mateix format que l'exportació de plays):",
        type=["csv"],
        key=f"plays_uploader_{st.session_state.get('plays_uploader_n', 0)}",
    )
    if fitxer_partides is not None:
        remember_upload(UPLOADED_PLAYS_KEY, fitxer_partides)
    nom_partides = uploaded_name(UPLOADED_PLAYS_KEY)
    if nom_partides:
        st.success(f"Fent servir **{nom_partides}**")
        if st.button("↩️ Tornar a les partides per defecte"):
            forget_upload(UPLOADED_PLAYS_KEY)
            # Canviem la clau del file_uploader perquè es buidi
            st.session_state['plays_uploader_n'] = st.session_state.get('plays_uploader_n', 0) + 1
            st.rerun()
    else:
        st.caption(f"Fent servir {PLAYS_CSV}")

if uploaded_name(UPLOADED_PLAYS_KEY):
    raw_plays = load_csv_bytes(st.session_state[UPLOADED_PLAYS_KEY]["data"])
else:
    raw_plays = load_csv(PLAYS_CSV)

df_2 = prepare_plays(raw_plays)

if df_2.empty:
    st.warning("El CSV de partides no té cap partida amb data vàlida.")
    st.stop()

game_col = next((c for c in df_2.columns if "Name" in c or "object" in c.lower()), None)

# ============================================================
# 🔵 SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Filtres")

data_min = df_2["Date"].min().date()
data_max = df_2["Date"].max().date()

# --- Date presets (relatius a l'última partida registrada) ---
preset = st.sidebar.selectbox(
    "Període:",
    ["Tot", "Última setmana", "Últim mes", "Últim any", "Personalitzat"]
)
st.sidebar.caption(f"Última partida registrada: {data_max.strftime('%d/%m/%Y')}")

if preset == "Tot":
    start, end = data_min, data_max
elif preset == "Última setmana":
    start, end = data_max - timedelta(days=7), data_max
elif preset == "Últim mes":
    start, end = data_max - timedelta(days=30), data_max
elif preset == "Últim any":
    start, end = data_max - timedelta(days=365), data_max
else:
    # Personalitzat: mentre s'escull la segona data, date_input retorna una sola data
    rang = st.sidebar.date_input(
        "Rang de dates:",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=data_max
    )
    if isinstance(rang, (list, tuple)) and len(rang) == 2:
        start, end = rang
    else:
        st.sidebar.info("Escull la data final del rang.")
        start = rang[0] if isinstance(rang, (list, tuple)) else rang
        end = data_max

start = pd.to_datetime(max(start, data_min))
end = pd.to_datetime(min(end, data_max))

# --- Players filter ---
player_counts = df_2["player_list"].explode().dropna().value_counts()
all_players = sorted(player_counts.index)

players_sel = st.sidebar.multiselect(
    "Jugadors:",
    options=all_players,
    default=[],
    placeholder="Tots els jugadors"
)

mode_jugadors = st.sidebar.radio(
    "Partides on hi ha:",
    ["Algun dels seleccionats", "Tots els seleccionats"],
    disabled=len(players_sel) < 2
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
    sel = set(players_sel)
    if mode_jugadors == "Tots els seleccionats":
        mask = df_filtered["player_list"].apply(lambda lst: sel.issubset(lst))
    else:
        mask = df_filtered["player_list"].apply(lambda lst: bool(sel.intersection(lst)))
    df_filtered = df_filtered[mask]

# ============================================================
# 🔵 KPIs
# ============================================================

st.header("📌 KPIs principals")

col1, col2, col3 = st.columns(3)

col1.metric("🎲 Total de partides", len(df_filtered))

if game_col:
    col2.metric("🧩 Jocs diferents", df_filtered[game_col].nunique())
else:
    col2.metric("🧩 Jocs diferents", "—")

col3.metric(
    "👥 Jugadors diferents",
    df_filtered["player_list"].explode().dropna().nunique()
)

# ============================================================
# 🔵 EVOLUCIÓ TEMPORAL
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
        title=f"Evolució de partides ({grouping.lower()})",
        color_discrete_sequence=[ACCENT]
    )

    fig_time.update_xaxes(tickformat=tick_format, nticks=12)
    fig_time.update_traces(line=dict(width=3), hovertemplate="Data: %{x}<br>Partides: %{y}")

    st.plotly_chart(style_plotly(fig_time), width="stretch")

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
        text="count",
        labels={"count": "Partides", "Name": ""},
        color_discrete_sequence=[ACCENT]
    )
    fig_games.update_traces(textposition="outside", cliponaxis=False)
    fig_games.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(style_plotly(fig_games), width="stretch")

# ============================================================
# 🔵 TOP JUGADORS
# ============================================================

st.subheader("👤 Jugadors més actius")

if not df_filtered.empty:
    players = (
        df_filtered["player_list"]
        .explode()
        .dropna()
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
        text="count",
        labels={"count": "Partides", "Player": ""},
        color_discrete_sequence=[ACCENT]
    )
    fig_players.update_traces(textposition="outside", cliponaxis=False)
    fig_players.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(style_plotly(fig_players), width="stretch")

# ============================================================
# 🔵 % DE VICTÒRIES
# ============================================================

st.subheader("🥇 Percentatge de victòries")

# Només comptem partides amb guanyador informat
df_wins = df_filtered[df_filtered["winner_list"].str.len() > 0]

# Als cooperatius normalment només s'informen les victòries (guanyen tots),
# i les derrotes queden sense guanyador → inflarien el % de victòries.
excloure_coop = st.checkbox(
    "Excloure partides on guanyen tots els jugadors (cooperatius)",
    value=True,
    key="wr_excloure_coop"
)
if excloure_coop and not df_wins.empty:
    tots_guanyen = df_wins.apply(
        lambda r: len(r["player_list"]) > 1 and set(r["player_list"]) <= set(r["winner_list"]),
        axis=1
    )
    df_wins = df_wins[~tots_guanyen]

if df_wins.empty:
    st.info("No hi ha partides amb guanyador informat en aquest filtre.")
else:
    wr_col1, wr_col2 = st.columns([1, 1])
    with wr_col1:
        min_partides = st.slider(
            "Mínim de partides (amb guanyador) per aparèixer:",
            1, 50, 10, key="wr_min_partides"
        )
    with wr_col2:
        excloure = st.multiselect(
            "Excloure de la classificació:",
            options=all_players,
            default=[p for p in ["Anonymous player", "Automa"] if p in all_players],
            key="wr_excloure"
        )

    jugades = df_wins["player_list"].explode().dropna().value_counts()
    victories = (
        df_wins.apply(
            lambda r: [w for w in set(r["winner_list"]) if w in r["player_list"]],
            axis=1
        )
        .explode()
        .dropna()
        .value_counts()
    )

    df_wr = pd.DataFrame({"Partides": jugades, "Victòries": victories}).fillna(0)
    df_wr["Victòries"] = df_wr["Victòries"].astype(int)
    df_wr["% victòries"] = (df_wr["Victòries"] / df_wr["Partides"] * 100).round(1)
    df_wr = df_wr[(df_wr["Partides"] >= min_partides) & (~df_wr.index.isin(excloure))]
    if players_sel:
        df_wr = df_wr[df_wr.index.isin(players_sel)]
    df_wr = df_wr.rename_axis("Jugador").reset_index().sort_values("% victòries", ascending=False)

    st.caption(
        f"Basat en {len(df_wins)} partides amb guanyador informat "
        f"(s'ignoren {len(df_filtered) - len(df_wins)} sense guanyador"
        f"{' o cooperatives' if excloure_coop else ''}). "
        "En partides amb diversos guanyadors (empats o cooperatius) tots compten com a victòria."
    )

    if df_wr.empty:
        st.info("Cap jugador arriba al mínim de partides. Baixa el mínim.")
    else:
        df_wr_top = df_wr.head(20)
        fig_wr = px.bar(
            df_wr_top,
            x="% victòries",
            y="Jugador",
            orientation="h",
            text=df_wr_top.apply(
                lambda r: f"{r['% victòries']:.0f}% ({r['Victòries']}/{int(r['Partides'])})", axis=1
            ),
            title="Jugadors amb més % de victòries",
            labels={"Jugador": ""},
            color_discrete_sequence=[ACCENT],
            range_x=[0, 115],
            height=120 + 28 * len(df_wr_top),
        )
        fig_wr.update_traces(textposition="outside", cliponaxis=False)
        fig_wr.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_plotly(fig_wr), width="stretch")

    # --- % de victòries per joc d'un jugador ---
    if game_col:
        jugadors_wr = sorted(jugades.index)
        default_jugador = jugades.index[0]  # el jugador amb més partides
        jugador = st.selectbox(
            "Veure % de victòries per joc de:",
            jugadors_wr,
            index=jugadors_wr.index(default_jugador),
            key="wr_jugador"
        )
        min_partides_joc = st.slider(
            "Mínim de partides al joc:", 1, 20, 3, key="wr_min_partides_joc"
        )

        df_j = df_wins[df_wins["player_list"].apply(lambda lst: jugador in lst)].copy()
        df_j["guanya"] = df_j["winner_list"].apply(lambda lst: jugador in lst)
        per_joc = (
            df_j.groupby(game_col)["guanya"]
            .agg(Partides="count", Victòries="sum")
            .reset_index()
            .rename(columns={game_col: "Joc"})
        )
        per_joc["Victòries"] = per_joc["Victòries"].astype(int)
        per_joc["% victòries"] = (per_joc["Victòries"] / per_joc["Partides"] * 100).round(1)
        per_joc = per_joc[per_joc["Partides"] >= min_partides_joc]

        if per_joc.empty:
            st.info(f"{jugador} no té cap joc amb {min_partides_joc} o més partides amb guanyador.")
        else:
            millors, pitjors = st.columns(2)
            with millors:
                st.markdown(f"**💪 On guanya més {jugador}**")
                st.dataframe(
                    per_joc.sort_values(["% victòries", "Partides"], ascending=[False, False]).head(10),
                    hide_index=True, width="stretch"
                )
            with pitjors:
                st.markdown(f"**😓 On li costa més a {jugador}**")
                st.dataframe(
                    per_joc.sort_values(["% victòries", "Partides"], ascending=[True, False]).head(10),
                    hide_index=True, width="stretch"
                )

# ============================================================
# 🔵 HEATMAP
# ============================================================

st.subheader("🤝 Jugadors que juguen junts (Heatmap)")

if not df_filtered.empty:
    # Sense selecció: els 15 jugadors més actius del filtre
    if players_sel:
        heat_players = sorted(players_sel)
    else:
        heat_players = sorted(
            df_filtered["player_list"].explode().dropna().value_counts().head(15).index
        )
        st.caption("Es mostren els 15 jugadors més actius. Selecciona jugadors a la barra lateral per canviar-ho.")

    pairs_df = df_filtered["player_list"].apply(lambda lst: [p for p in lst if p in heat_players])

    pair_counts = {}
    for players_list in pairs_df:
        if len(players_list) > 1:
            for p1, p2 in combinations(sorted(set(players_list)), 2):
                pair_counts[(p1, p2)] = pair_counts.get((p1, p2), 0) + 1

    matrix = pd.DataFrame(0.0, index=heat_players, columns=heat_players)

    for (p1, p2), count in pair_counts.items():
        matrix.loc[p1, p2] = count
        matrix.loc[p2, p1] = count

    if pair_counts:
        # Posem la diagonal a NaN perquè no es pinti
        for p in heat_players:
            matrix.loc[p, p] = None

        fig_heatmap = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Freqüència de jugadors que coincideixen en partides"
        )
        st.plotly_chart(style_plotly(fig_heatmap), width="stretch")
    else:
        st.info("No hi ha prou dades per generar el heatmap.")
