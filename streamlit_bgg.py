import os
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    ACCENT,
    COLLECTION_CSV,
    UPLOADED_COLLECTION_KEY,
    bgg_url,
    clean_mechanics,
    forget_upload,
    load_csv,
    load_csv_bytes,
    remember_upload,
    style_plotly,
    uploaded_name,
)

st.set_page_config(page_title="Ludoteca", page_icon="📚", layout="wide")

# ============================================================
# 🔵 PRIMER PAS: ELECCIÓ DE MÈTODE DE CÀRREGA DEL CSV
# ============================================================

st.sidebar.subheader("📂 Dades d'entrada")

OPCIO_PUJAR = "📁 Pujar un CSV manualment"
OPCIO_DEFECTE = "💾 Veure la ludo de Mrbrussels per defecte"

# Si a la sessió ja hi ha un CSV pujat (p. ex. tornant d'una altra pàgina), el mantenim
metode = st.sidebar.radio(
    "Com vols carregar les dades?",
    (OPCIO_PUJAR, OPCIO_DEFECTE),
    index=0 if uploaded_name(UPLOADED_COLLECTION_KEY) else 1,
)

df = None

if metode == OPCIO_PUJAR:
    fitxer = st.sidebar.file_uploader(
        "Puja el teu CSV:", type=["csv"],
        key=f"collection_uploader_{st.session_state.get('collection_uploader_n', 0)}",
    )
    if fitxer is not None:
        remember_upload(UPLOADED_COLLECTION_KEY, fitxer)

    nom_pujat = uploaded_name(UPLOADED_COLLECTION_KEY)
    if nom_pujat:
        df = load_csv_bytes(st.session_state[UPLOADED_COLLECTION_KEY]["data"]).copy()
        st.sidebar.success(f"Fent servir **{nom_pujat}** (també al recomanador).")
        if st.sidebar.button("🗑️ Oblidar el CSV pujat"):
            forget_upload(UPLOADED_COLLECTION_KEY)
            # Canviem la clau del file_uploader perquè es buidi
            st.session_state['collection_uploader_n'] = st.session_state.get('collection_uploader_n', 0) + 1
            st.rerun()
else:
    # Tornem a la col·lecció per defecte a totes les pàgines
    forget_upload(UPLOADED_COLLECTION_KEY)
    if os.path.exists(COLLECTION_CSV):
        df = load_csv(COLLECTION_CSV).copy()
        st.sidebar.info(f"S'està utilitzant el CSV per defecte: {COLLECTION_CSV}")
    else:
        st.sidebar.error(f"No s'ha trobat el fitxer per defecte: {COLLECTION_CSV}")

st.title("📚 Ludoteca de dumito")

# Si encara no tenim dades, aturem l'app
if df is None:
    st.info(
        "Puja el CSV de la teva col·lecció exportat de BGG "
        "(*Collection → Export → CSV*) per continuar."
    )
    st.stop()

# Neteja de la mecànica principal (tant per al CSV per defecte com per al pujat)
if "comment" in df.columns:
    df["comment"] = clean_mechanics(df["comment"])

# 🔥 Renombrar columnes
df = df.rename(columns={
    "objectname": "nom_del_joc",
    "numplays": "num_partides",
    "avgweight": "pes",
    "average": "nota_bgg",
    "yearpublished": "any_publicació"
})

columnes_necessaries = ["nom_del_joc", "num_partides", "pes", "nota_bgg", "any_publicació"]
falten = [c for c in columnes_necessaries if c not in df.columns]
if falten:
    st.error(
        "⚠️ Al CSV li falten columnes necessàries: "
        + ", ".join(falten)
        + ". Exporta la col·lecció des de BGG (objectname, numplays, avgweight, average, yearpublished)."
    )
    st.stop()

# Eliminem columnes buides (excepte les necessàries, que es queden encara que siguin buides)
buides = [c for c in df.columns if df[c].isna().all() and c not in columnes_necessaries]
df = df.drop(columns=buides)

for col in ["num_partides", "pes", "nota_bgg", "any_publicació"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["num_partides"] = df["num_partides"].fillna(0)

# Idiomes sense informar els passem tots a Desconegut
te_idiomes = "version_languages" in df.columns
if te_idiomes:
    df["version_languages"] = (
        df["version_languages"].fillna("Desconegut").replace("", "Desconegut")
    )

te_itemtype = "itemtype" in df.columns

# --- Crear columna d'èpoques ---
df["epocas"] = pd.cut(
    df["any_publicació"],
    bins=[-3000, 1900, 2020, 2050],
    labels=["Antiguitat", "Moderna", "Post-Pandèmia"]
)

# ============================
# 🔵 FILTRES INTERACTIUS
# ============================

st.sidebar.header("Filtres")

epocas_sel = st.sidebar.multiselect(
    "Selecciona epoques:",
    options=df["epocas"].dropna().unique(),
    default=df["epocas"].dropna().unique()
)

# Rang d'anys (només si hi ha anys vàlids i més d'un valor)
anys_valids = df["any_publicació"].dropna()
year_range = None
if anys_valids.empty:
    st.sidebar.caption("Cap joc té any de publicació informat.")
elif anys_valids.min() == anys_valids.max():
    st.sidebar.caption(f"Tots els jocs són de l'any {int(anys_valids.min())}.")
else:
    min_year, max_year = int(anys_valids.min()), int(anys_valids.max())
    year_range = st.sidebar.slider(
        "Rang d'anys:",
        min_year, max_year, (min_year, max_year)
    )

idioma_sel = []
if te_idiomes:
    idiomas = df["version_languages"].dropna().unique()
    idioma_sel = st.sidebar.multiselect(
        "Idioma:",
        options=idiomas,
        default=idiomas
    )

if "own" in df.columns:
    filtre_own = st.sidebar.checkbox("Només jocs en propietat", value=False)
else:
    filtre_own = False

df_filtrado = df.copy()

if epocas_sel:
    df_filtrado = df_filtrado[df_filtrado["epocas"].isin(epocas_sel)]

if year_range is not None:
    df_filtrado = df_filtrado[
        df_filtrado["any_publicació"].between(year_range[0], year_range[1], inclusive="both")
    ]

if te_idiomes and idioma_sel:
    df_filtrado = df_filtrado[df_filtrado["version_languages"].isin(idioma_sel)]

if filtre_own:
    df_filtrado = df_filtrado[pd.to_numeric(df_filtrado["own"], errors="coerce") == 1]

# ============================
# 🔵 KPIs PRINCIPALS
# ============================

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("🎯 Jocs filtrats", len(df_filtrado))
col2.metric(
    "⚖️ Pes mitjà",
    round(df_filtrado["pes"].mean(), 2) if df_filtrado["pes"].notna().any() else "—"
)
col3.metric(
    "⭐ Nota BGG mitjana",
    round(df_filtrado["nota_bgg"].mean(), 2) if df_filtrado["nota_bgg"].notna().any() else "—"
)
col4.metric(
    "🎲 Total de partides",
    int(df_filtrado["num_partides"].sum()) if not df_filtrado.empty else "—"
)
anys_filtrats = df_filtrado["any_publicació"].dropna()
col5.metric(
    "📅 Any amb més jocs",
    int(anys_filtrats.value_counts().idxmax()) if not anys_filtrats.empty else "—"
)

tab_col, tab_graf, tab_pend = st.tabs(["📋 Col·lecció", "📊 Gràfics", "😅 Pendents d'estrenar"])

# ============================================================
# 📋 PESTANYA 1: TAULA DE LA COL·LECCIÓ
# ============================================================

COLUMNES_TAULA = [
    "bgg", "nom_del_joc", "any_publicació", "comment", "pes", "nota_bgg", "rating",
    "num_partides", "minplayers", "maxplayers", "playingtime", "version_languages", "own",
]

config_columnes = {
    "bgg": st.column_config.LinkColumn("BGG", display_text="🔗", width="small"),
    "nom_del_joc": st.column_config.TextColumn("Joc", width="large"),
    "any_publicació": st.column_config.NumberColumn("Any", format="%d"),
    "comment": st.column_config.TextColumn("Mecànica"),
    "pes": st.column_config.ProgressColumn("Pes", min_value=1, max_value=5, format="%.2f"),
    "nota_bgg": st.column_config.ProgressColumn("Nota BGG", min_value=0, max_value=10, format="%.1f"),
    "rating": st.column_config.NumberColumn("La meva nota", format="%.1f"),
    "num_partides": st.column_config.NumberColumn("Partides", format="%d"),
    "minplayers": st.column_config.NumberColumn("Mín. jug.", format="%d"),
    "maxplayers": st.column_config.NumberColumn("Màx. jug.", format="%d"),
    "playingtime": st.column_config.NumberColumn("Durada", format="%d min"),
    "version_languages": st.column_config.TextColumn("Idioma"),
    "own": st.column_config.CheckboxColumn("En propietat"),
}

with tab_col:
    df_taula = df_filtrado.copy()
    if "objectid" in df_taula.columns:
        df_taula["bgg"] = df_taula["objectid"].map(bgg_url)
    if "rating" in df_taula.columns:
        # BGG exporta 0 o "N/A" quan no has puntuat el joc
        df_taula["rating"] = pd.to_numeric(df_taula["rating"], errors="coerce").replace(0, pd.NA)
    if "own" in df_taula.columns:
        df_taula["own"] = pd.to_numeric(df_taula["own"], errors="coerce") == 1

    totes = st.toggle("Mostrar totes les columnes del CSV", value=False)
    if totes:
        st.dataframe(df_filtrado, hide_index=True)
    else:
        cols = [c for c in COLUMNES_TAULA if c in df_taula.columns]
        st.dataframe(
            df_taula[cols].sort_values("nom_del_joc"),
            column_config=config_columnes,
            hide_index=True,
        )

# ============================================================
# 📊 PESTANYA 2: GRÀFICS
# ============================================================

with tab_graf:
    if df_filtrado.empty:
        st.info("Cap joc compleix els filtres.")
    else:
        g1, g2 = st.columns(2)

        # --- Jocs per època ---
        with g1:
            conteo_epocas = df_filtrado["epocas"].value_counts().reset_index()
            conteo_epocas.columns = ["epoca", "cantidad"]
            fig1 = px.pie(
                conteo_epocas,
                names="epoca",
                values="cantidad",
                title="Distribució de jocs per època",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig1.update_traces(textposition="inside", textinfo="percent+label")
            fig1.update_layout(showlegend=False)
            st.plotly_chart(style_plotly(fig1), width="stretch")

        # --- Histograma d'anys ---
        with g2:
            fig2 = px.histogram(
                df_filtrado,
                x="any_publicació",
                nbins=30,
                title="Distribució d'anys de publicació",
                color_discrete_sequence=[ACCENT]
            )
            fig2.update_layout(xaxis_title="Any de publicació", yaxis_title="Recompte")
            st.plotly_chart(style_plotly(fig2), width="stretch")

        # Els gràfics de barres van a amplada completa (els noms dels jocs són llargs)

        # --- Top 10 jocs més jugats ---
        top10 = df_filtrado.sort_values("num_partides", ascending=False).head(10)
        fig5 = px.bar(
            top10,
            x="num_partides",
            y="nom_del_joc",
            orientation="h",
            text="num_partides",
            title="🏆 Top 10 jocs més jugats",
            color_discrete_sequence=[ACCENT]
        )
        fig5.update_traces(textposition="outside", cliponaxis=False)
        fig5.update_layout(
            xaxis_title="Nombre de partides",
            yaxis_title=None,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(style_plotly(fig5), width="stretch")

        # --- Top 10 mecàniques amb més partides ---
        if "comment" in df_filtrado.columns:
            mecaniques_top = (
                df_filtrado.groupby("comment")["num_partides"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            mecaniques_top.columns = ["Mecànica", "Partides"]
            fig_mec = px.bar(
                mecaniques_top,
                x="Partides",
                y="Mecànica",
                orientation="h",
                text="Partides",
                title="🧠 Top 10 mecàniques amb més partides",
                color_discrete_sequence=[ACCENT]
            )
            fig_mec.update_traces(textposition="outside", cliponaxis=False)
            fig_mec.update_layout(
                yaxis=dict(autorange="reversed"),
                xaxis_title="Total de partides",
                yaxis_title=None,
            )
            st.plotly_chart(style_plotly(fig_mec), width="stretch")
        else:
            st.info("El CSV no té la columna de mecànica principal (comment).")

        g5, g6 = st.columns(2)

        # --- Relació entre variables i partides ---
        with g5:
            etiquetes_scatter = {
                "pes": "Pes (complexitat)",
                "nota_bgg": "Nota BGG"
            }
            opcion_x = st.selectbox(
                "Variable a comparar amb el nombre de partides:",
                options=list(etiquetes_scatter),
                format_func=etiquetes_scatter.get
            )
            etiqueta_x = etiquetes_scatter[opcion_x]
            df_scatter = df_filtrado[df_filtrado["num_partides"] > 0]
            fig4 = px.scatter(
                df_scatter,
                x=opcion_x,
                y="num_partides",
                hover_name="nom_del_joc",
                title=f"{etiqueta_x} vs Nombre de partides",
                labels={opcion_x: etiqueta_x, "num_partides": "Nombre de partides"},
            )
            fig4.update_traces(marker=dict(size=10))
            st.plotly_chart(style_plotly(fig4), width="stretch")

        # --- Jocs per idioma ---
        with g6:
            if te_idiomes:
                mostrar_desconegut = st.checkbox("Mostrar 'Desconegut' al gràfic d'idiomes", value=False)
                df_idiomas_graf = df_filtrado
                if not mostrar_desconegut:
                    df_idiomas_graf = df_idiomas_graf[df_idiomas_graf["version_languages"] != "Desconegut"]
                conteo_idiomas = df_idiomas_graf["version_languages"].value_counts().reset_index()
                conteo_idiomas.columns = ["idioma", "cantidad"]
                fig3 = px.pie(
                    conteo_idiomas,
                    names="idioma",
                    values="cantidad",
                    title="🌍 Jocs per idioma"
                )
                st.plotly_chart(style_plotly(fig3), width="stretch")

# ============================================================
# 😅 PESTANYA 3: LLISTA DE LA VERGONYA
# ============================================================

with tab_pend:
    df_vergonya = df_filtrado[df_filtrado["num_partides"] == 0].copy()

    if te_itemtype:
        mostrar_expansions = st.checkbox("Mostrar expansions a la llista", value=False)
        if not mostrar_expansions:
            df_vergonya = df_vergonya[df_vergonya["itemtype"] != "expansion"]

    df_vergonya = df_vergonya.sort_values("nom_del_joc")

    if df_vergonya.empty:
        st.success("🎉 No tens cap joc sense jugar! Bona feina!")
    else:
        st.warning(f"Tens {len(df_vergonya)} jocs sense estrenar...")
        if "objectid" in df_vergonya.columns:
            df_vergonya["bgg"] = df_vergonya["objectid"].map(bgg_url)
        cols_vergonya = ["bgg", "nom_del_joc", "any_publicació", "pes", "version_languages", "itemtype"]
        st.dataframe(
            df_vergonya[[c for c in cols_vergonya if c in df_vergonya.columns]],
            column_config={**config_columnes, "itemtype": st.column_config.TextColumn("Tipus")},
            hide_index=True,
        )
