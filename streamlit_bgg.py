import os
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import COLLECTION_CSV, clean_mechanics, fix_plotly_colors, load_csv, load_csv_bytes

st.set_page_config(layout="wide")

# ============================================================
# 🔵 PRIMER PAS: ELECCIÓ DE MÈTODE DE CÀRREGA DEL CSV
# ============================================================

st.sidebar.subheader("📂 Dades d'entrada")

metode = st.sidebar.radio(
    "Com vols carregar les dades?",
    ("📁 Pujar un CSV manualment", "💾 Veure la ludo de Mrbrussels per defecte")
)

df = None

if metode == "📁 Pujar un CSV manualment":
    fitxer = st.sidebar.file_uploader("Puja el teu CSV:", type=["csv"])
    if fitxer is not None:
        df = load_csv_bytes(fitxer.getvalue()).copy()
        st.sidebar.success("CSV carregat correctament!")
else:
    if os.path.exists(COLLECTION_CSV):
        df = load_csv(COLLECTION_CSV).copy()
        st.sidebar.info(f"S'està utilitzant el CSV per defecte: {COLLECTION_CSV}")
    else:
        st.sidebar.error(f"No s'ha trobat el fitxer per defecte: {COLLECTION_CSV}")

# Si encara no tenim dades, aturem l'app
if df is None:
    st.warning("Carrega un CSV per continuar.")
    st.stop()

# Neteja de la mecànica principal (tant per al CSV per defecte com per al pujat)
if "comment" in df.columns:
    df["comment"] = clean_mechanics(df["comment"])


# ============================
# 🔵 ESTILO PERSONALIZADO
# ============================
st.markdown("""
<style>

    /* ============================
       🔵 FONDO GENERAL (modo claro)
       ============================ */
    .stApp {
        background-color: #FFF4E6 !important;
        color: #000000 !important;
    }

    html, body {
        background-color: #FFF4E6 !important;
        color: #000000 !important;
    }

    /* ============================
       🔵 SIDEBAR
       ============================ */
    section[data-testid="stSidebar"] {
        background-color: #FFE8D1 !important;
        color: #000000 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    /* ============================
       🔵 BORDES REDONDEADOS
       ============================ */
    .stButton>button,
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div {
        border-radius: 8px !important;
    }

    /* ============================
       🔵 TEXTO GLOBAL (forzar negro)
       ============================ */
    *, html, body, [class*="css"] {
        color: #000000 !important;
    }

    /* ============================
       🔵 WIDGETS
       ============================ */
    .stTextInput input,
    .stSelectbox div,
    .stNumberInput input,
    .stCheckbox label,
    .stRadio label,
    .stMetric label,
    .stMetric div {
        color: #000000 !important;
    }

    /* ============================
       🔵 BOTONES
       ============================ */
    .stButton button {
        color: #000000 !important;
        background-color: #FFD8A8 !important;
        border-radius: 8px !important;
    }

    /* ============================
       🔵 TABLAS
       ============================ */
    .dataframe tbody tr td,
    .dataframe thead tr th {
        color: #000000 !important;
    }

    /* ============================
       🔵 PLOTLY (títulos, ejes, labels)
       ============================ */
    .js-plotly-plot .plotly .main-svg {
        color: #000000 !important;
        fill: #000000 !important;
    }

    /* ============================
       🔵 RESPONSIVE PARA MÓVIL
       ============================ */
    @media (max-width: 768px) {

        /* Texto más grande y negro */
        body, html, .stApp {
            font-size: 18px !important;
            color: #000000 !important;
        }

        /* Títulos */
        h1, h2, h3, h4 {
            color: #000000 !important;
        }

        /* Inputs más grandes */
        .stTextInput input,
        .stSelectbox div {
            font-size: 18px !important;
        }

        /* Botones más accesibles */
        .stButton button {
            font-size: 18px !important;
            padding: 12px 20px !important;
        }

        /* Sidebar texto más grande */
        section[data-testid="stSidebar"] * {
            font-size: 18px !important;
        }
    }

</style>
""", unsafe_allow_html=True)

st.title("📚 Ludoteca de dumito")

# --- Carga de CSV ---
st.header("Llista completa")
# 🔥 Renombrar columnas
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

# --- Crear columna de épocas ---
df["epocas"] = pd.cut(
    df["any_publicació"],
    bins=[-3000, 1900, 2020, 2050],
    labels=["Antiguitat", "Moderna", "Post-Pandèmia"]
)

# ============================
# 🔵 FILTROS INTERACTIVOS
# ============================

st.sidebar.header("Filtres")

# Filtro por época
epocas_sel = st.sidebar.multiselect(
    "Selecciona epoques:",
    options=df["epocas"].dropna().unique(),
    default=df["epocas"].dropna().unique()
)

# Filtro por rango de años (només si hi ha anys vàlids i més d'un valor)
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

# Filtro por idioma
idioma_sel = []
if te_idiomes:
    idiomas = df["version_languages"].dropna().unique()
    idioma_sel = st.sidebar.multiselect(
        "Idioma:",
        options=idiomas,
        default=idiomas
    )

# 🔵 Nou filtre: jocs en propietat
if "own" in df.columns:
    filtre_own = st.sidebar.checkbox("Només jocs en propietat", value=False)
else:
    filtre_own = False

# Aplicar filtros
df_filtrado = df.copy()

if epocas_sel:
    df_filtrado = df_filtrado[df_filtrado["epocas"].isin(epocas_sel)]

if year_range is not None:
    df_filtrado = df_filtrado[
        df_filtrado["any_publicació"].between(year_range[0], year_range[1], inclusive="both")
    ]

if te_idiomes and idioma_sel:
    df_filtrado = df_filtrado[df_filtrado["version_languages"].isin(idioma_sel)]

# Aplicar filtre de jocs en propietat
if filtre_own:
    df_filtrado = df_filtrado[pd.to_numeric(df_filtrado["own"], errors="coerce") == 1]

st.subheader("La col·lecció")

# ============================
# 🔵 KPIs PRINCIPALS
# ============================

st.subheader("📌 Indicadors principals del filtre actual")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="🎯 Jocs filtrats",
        value=len(df_filtrado)
    )

with col2:
    st.metric(
        label="⚖️ Pes mitjà",
        value=round(df_filtrado["pes"].mean(), 2) if not df_filtrado.empty else "—"
    )

with col3:
    st.metric(
        label="⭐ Nota BGG mitjana",
        value=round(df_filtrado["nota_bgg"].mean(), 2) if not df_filtrado.empty else "—"
    )

with col4:
    st.metric(
        label="🎲 Total de partides",
        value=int(df_filtrado["num_partides"].sum()) if not df_filtrado.empty else "—"
    )

with col5:
    anys_filtrats = df_filtrado["any_publicació"].dropna()
    if not anys_filtrats.empty:
        any_mes_jocs = anys_filtrats.value_counts().idxmax()
        st.metric(
            label="📅 Any amb més jocs",
            value=int(any_mes_jocs)
        )
    else:
        st.metric(label="📅 Any amb més jocs", value="—")

st.dataframe(df_filtrado)

# ============================
# 🔵 GRÁFICO 1: Juegos por época
# ============================

conteo_epocas = df_filtrado["epocas"].value_counts().reset_index()
conteo_epocas.columns = ["epoca", "cantidad"]

st.subheader("📊 Distribució de jocs per època")

fig1 = px.pie(
    conteo_epocas,
    names="epoca",
    values="cantidad",
    title="Distribució de jocs per època",
    color="epoca",
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig1.update_traces(
    textposition="inside",
    textinfo="percent+label",
    pull=[0.05] * len(conteo_epocas)
)

fig1.update_layout(
    showlegend=False,
    plot_bgcolor="#FFF4E6",
    paper_bgcolor="#FFF4E6"
)
fig1 = fix_plotly_colors(fig1)
st.plotly_chart(fig1)

# ============================
# 🔵 GRÁFICO 2: Histograma de años
# ============================

st.subheader("📈 Histograma per any de publicació")

fig2 = px.histogram(
    df_filtrado,
    x="any_publicació",
    nbins=30,
    title="Distribució d'anys de publicació"
)

fig2.update_layout(
    xaxis_title="Any de publicació",
    yaxis_title="Recompte",
    plot_bgcolor="#FFF4E6",
    paper_bgcolor="#FFF4E6"
)
fig2 = fix_plotly_colors(fig2)

st.plotly_chart(fig2)

# ============================
# 🔵 GRÁFICO 5: Top 10 jocs més jugats (versió millorada)
# ============================

st.subheader("🏆 Top 10 jocs més jugats")

top10 = df_filtrado.sort_values("num_partides", ascending=False).head(10)

fig5 = px.bar(
    top10,
    x="num_partides",
    y="nom_del_joc",
    orientation="h",
    text="num_partides",
    title="Top 10 jocs més jugats",
    color="num_partides",
    color_continuous_scale="Blues"
)

fig5.update_traces(
    textposition="outside",
    marker=dict(line=dict(color="black", width=0.5))
)

fig5.update_layout(
    xaxis_title="Nombre de partides",
    yaxis_title="Joc",
    yaxis=dict(autorange="reversed"),
    coloraxis_showscale=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)
fig5 = fix_plotly_colors(fig5)
st.plotly_chart(fig5, width="stretch")

# ============================
# 🔵 GRÀFIC NOU: Top 10 mecàniques amb més partides jugades
# ============================

st.subheader("🧠 Top 10 mecàniques amb més partides jugades")

if not df_filtrado.empty and "comment" in df_filtrado.columns:

    # Agrupar per mecànica i sumar partides
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
        title="Top 10 mecàniques amb més partides jugades",
        color="Partides",
        color_continuous_scale="Oranges"
    )

    fig_mec.update_traces(
        textposition="outside",
        marker=dict(line=dict(color="black", width=0.5))
    )

    fig_mec.update_layout(
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Total de partides",
        yaxis_title="Mecànica"
    )

    fig_mec = fix_plotly_colors(fig_mec)
    st.plotly_chart(fig_mec, width="stretch")

else:
    st.info("No hi ha dades suficients per generar el gràfic de mecàniques.")


# ============================
# 🔵 GRÁFICO 3: Juegos por idioma
# ============================

if te_idiomes:
    st.subheader("🌍 Jocs per idioma")

    mostrar_desconegut = st.checkbox("Mostrar 'Desconegut' al gràfic", value=False)

    df_idiomas_graf = df_filtrado.copy()
    if not mostrar_desconegut:
        df_idiomas_graf = df_idiomas_graf[df_idiomas_graf["version_languages"] != "Desconegut"]

    conteo_idiomas = df_idiomas_graf["version_languages"].value_counts().reset_index()
    conteo_idiomas.columns = ["idioma", "cantidad"]

    fig3 = px.pie(
        conteo_idiomas,
        names="idioma",
        values="cantidad",
        title="Distribución por idioma"
    )

    fig3.update_layout(
        plot_bgcolor="#FFF4E6",
        paper_bgcolor="#FFF4E6"
    )
    fig3 = fix_plotly_colors(fig3)
    st.plotly_chart(fig3)

# ============================
# 🔵 GRÁFICO 4: Scatter interactiu
# ============================

st.subheader("📌 Relació entre variables i nombre de partides")

etiquetes_scatter = {
    "pes": "Pes (complexitat)",
    "nota_bgg": "Nota BGG"
}
opcion_x = st.selectbox(
    "Selecciona la variable per comparar amb nombre de partides:",
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
    labels={
        opcion_x: etiqueta_x,
        "num_partides": "Nombre de partides"
    },
)

fig4.update_traces(marker=dict(size=10, opacity=1))
fig4.update_layout(
    xaxis_title=etiqueta_x,
    yaxis_title="Nombre de partides",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)
fig4 = fix_plotly_colors(fig4)
st.plotly_chart(fig4, width="stretch")

# ============================
# 🔵 LISTA DE LA VERGÜENZA
# ============================

st.subheader("😅 Llista de la vergonya (jocs sense jugar)")

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
    cols_vergonya = ["nom_del_joc", "any_publicació", "pes", "version_languages", "itemtype"]
    st.dataframe(df_vergonya[[c for c in cols_vergonya if c in df_vergonya.columns]])

# ============================
# 🔵 MISSATGES DELS USUARIS
# ============================

