import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(layout="wide")
st.title("🎯 Recomanador personalitzat de jocs")

# ============================================================
# FUNCIONS DE MECÀNIQUES BGG
# ============================================================

BGG_CACHE_FILE = "pages/bgg_mechanics_cache.csv"

def load_mechanics_cache() -> dict:
    if not os.path.exists(BGG_CACHE_FILE):
        st.error("⚠️ No s'ha trobat bgg_mechanics_cache.csv al repositori!")
        st.stop()
    try:
        cache_df = pd.read_csv(BGG_CACHE_FILE)
        if cache_df.empty or "objectid" not in cache_df.columns or "mechanics" not in cache_df.columns:
            st.error("⚠️ El fitxer bgg_mechanics_cache.csv està buit o mal format!")
            st.stop()
        return dict(zip(
            cache_df["objectid"].astype(int),
            cache_df["mechanics"].apply(eval)
        ))
    except Exception as e:
        st.error(f"⚠️ Error llegint bgg_mechanics_cache.csv: {e}")
        st.stop()


# ============================================================
# LOAD COLLECTION DATA
# ============================================================

df = pd.read_csv("pages/collection.csv", sep=",", engine="python")

if "playingtime" not in df.columns:
    st.error("⚠️ La columna 'playingtime' no existeix al CSV!")
    st.stop()

df["playingtime"] = df["playingtime"].fillna(df["playingtime"].median())
df.columns = df.columns.str.strip()

df = df.rename(columns={
    "objectname": "nom_del_joc",
    "avgweight": "pes",
    "average": "nota_bgg",
    "yearpublished": "any_publicació"
})

required_cols = ["pes", "nota_bgg", "minplayers", "maxplayers"]
df = df.dropna(subset=required_cols)

df["comment"] = df["comment"].replace(r"^\s*$", "No informat", regex=True)
df["comment"] = df["comment"].fillna("No informat")
df["comment"] = df["comment"].replace("Selecció accions", "Selecció d'accions", regex=True)
df["comment"] = df["comment"].replace("Car-driven", "Card-driven", regex=True)
df["comment"] = df["comment"].replace("Col·locació daus", "Col·locació de daus", regex=True)
df["comment"] = df["comment"].replace("Construcció motor", "Construcció de motor", regex=True)
df["comment"] = df["comment"].replace("Draft pùblic", "Draft públic", regex=True)
df["comment"] = df["comment"].replace("Gestió de la mà", "Gestió de mà", regex=True)
df["comment"] = df["comment"].replace("Pseudo-abstracte", "Pseudo-Abstracte", regex=True)
df["comment"] = df["comment"].replace("Pseudo-wargame", "Pseudo-Wargame", regex=True)
df["comment"] = df["comment"].replace("Pseudo-wargames", "Pseudo-Wargame", regex=True)
df["comment"] = df["comment"].replace("Pseudo-Wargames", "Pseudo-Wargame", regex=True)
df = df.rename(columns={"comment": "Mecànica_principal"})

if "own" not in df.columns:
    st.warning("⚠️ El CSV no té la columna 'own'. Es crearà amb valor 0.")
    df["own"] = 0
else:
    df["own"] = pd.to_numeric(df["own"], errors="coerce").fillna(0).astype(int)


# ============================================================
# CÀRREGA DE MECÀNIQUES DES DE CACHÉ LOCAL
# ============================================================

if "mechanics_cache" not in st.session_state:
    disc_cache = load_mechanics_cache()
    st.session_state["mechanics_cache"] = disc_cache

mechanics_cache = st.session_state["mechanics_cache"]

# Assignar mecàniques a cada fila
if "objectid" in df.columns:
    df["objectid"] = pd.to_numeric(df["objectid"], errors="coerce")
    df["all_mechanics"] = df["objectid"].map(
        lambda oid: mechanics_cache.get(int(oid), []) if pd.notna(oid) else []
    )
else:
    df["all_mechanics"] = df["Mecànica_principal"].apply(lambda x: [x])

# One-hot encoding
all_mec_names = sorted(set(m for mecs in df["all_mechanics"] for m in mecs if m))

mec_data = {
    f"mec_{mec.replace(' ', '_')}": df["all_mechanics"].apply(lambda mecs: 1 if mec in mecs else 0)
    for mec in all_mec_names
}
mec_cols = pd.DataFrame(mec_data, index=df.index)
df = pd.concat([df, mec_cols], axis=1)

# ============================================================
# 1️⃣ USER PREFERENCES
# ============================================================

st.header("🧩 Preferències del jugador")

# Botó de reset
if st.button("🔄 Restablir preferències per defecte"):
    st.session_state["pes_pref"] = 2.5
    st.session_state["nota_pref"] = 6.5
    st.session_state["num_jugadors"] = 3
    st.session_state["durada_pref"] = 60
    st.session_state["mecanica_pref"] = "Qualsevol"
    st.rerun()

pes_pref = st.slider("Pes preferit (complexitat):", 0.0, 5.0, 
    st.session_state.get("pes_pref", 2.5), 0.1, key="pes_pref")
nota_pref = st.slider("Nota mínima BGG:", 0.0, 10.0, 
    st.session_state.get("nota_pref", 6.5), 0.1, key="nota_pref")
num_jugadors = st.slider("Nombre de jugadors preferit:", 1, 10, 
    st.session_state.get("num_jugadors", 3), key="num_jugadors")
durada_pref = st.slider("Durada preferida (minuts):", 10, 300, 
    st.session_state.get("durada_pref", 60), 5, key="durada_pref")

mecaniques = ["Qualsevol"] + all_mec_names
mecanica_pref = st.selectbox("Mecànica preferida:", mecaniques,
    index=mecaniques.index(st.session_state.get("mecanica_pref", "Qualsevol")) 
    if st.session_state.get("mecanica_pref", "Qualsevol") in mecaniques else 0,
    key="mecanica_pref")

#if st.button("↩️ Restablir preferències per defecte"):
 #   st.session_state["pes_pref"] = 2.5
  #  st.session_state["nota_pref"] = 6.5
   # st.session_state["num_jugadors"] = 3
    #st.session_state["durada_pref"] = 60
    #st.session_state["mecanica_pref"] = "Qualsevol"
    #st.rerun()



# ============================================================
# 2️⃣ USER RATINGS FOR SAMPLE GAMES
# ============================================================

st.subheader("⚙️ Opcions del recomanador")
filter_owned = st.checkbox("Només mostrar jocs que tinc en propietat", value=False)

st.header("⭐ Avalua alguns jocs")
st.subheader("🚥 Puntua del 1 al 10 fins a quin punt s'ajusten al que estàs buscant")
if "sample_games" not in st.session_state:
    st.session_state.sample_games = df.sample(10)[
        ["nom_del_joc", "pes", "nota_bgg", "minplayers", "maxplayers", "Mecànica_principal"]
    ]

if st.button("🔄 Tornar a mostrar altres jocs"):
    st.session_state.sample_games = df.sample(10)[
        ["nom_del_joc", "pes", "nota_bgg", "minplayers", "maxplayers", "Mecànica_principal"]
    ]

sample_games = st.session_state.sample_games

ignore_flags = {}
user_ratings = {}

for idx, row in sample_games.iterrows():
    col1, col2 = st.columns([3, 1])
    with col1:
        rating = st.slider(
            f"Valoració per **{row['nom_del_joc']}**:",
            1, 10, 5,
            key=f"rating_{row['nom_del_joc']}"
        )
    with col2:
        ignore = st.checkbox("Ignorar", key=f"ignore_{row['nom_del_joc']}")

    user_ratings[row["nom_del_joc"]] = rating
    ignore_flags[row["nom_del_joc"]] = ignore

# ============================================================
# 3️⃣ BUILD USER PROFILE VECTOR
# ============================================================

PLAYTIME_WEIGHT = 2.0
MECHANICS_WEIGHT = 2.3

numeric_cols = ["pes", "nota_bgg", "minplayers", "maxplayers", "playingtime"]

scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[numeric_cols])
X_numeric[:, -1] *= PLAYTIME_WEIGHT
X = np.hstack([X_numeric, mec_cols.values * MECHANICS_WEIGHT])

rated_games = df[df["nom_del_joc"].isin([g for g in user_ratings if not ignore_flags[g]])].copy()
rated_games["user_rating"] = rated_games["nom_del_joc"].map(user_ratings)

user_profile_numeric = np.average(
    scaler.transform(rated_games[numeric_cols]),
    axis=0,
    weights=rated_games["user_rating"]
)

rated_mec_matrix = mec_cols.loc[rated_games.index].values
user_mec_from_ratings = np.average(
    rated_mec_matrix,
    axis=0,
    weights=rated_games["user_rating"]
)

pref_df = pd.DataFrame([{
    "pes": pes_pref,
    "nota_bgg": nota_pref,
    "minplayers": num_jugadors,
    "maxplayers": num_jugadors,
    "playingtime": durada_pref
}])[numeric_cols]

pref_numeric_vector = scaler.transform(pref_df)[0]
pref_numeric_vector[-1] *= PLAYTIME_WEIGHT

user_mec_vector = np.zeros(len(mec_cols.columns))
if mecanica_pref != "Qualsevol":
    colname = f"mec_{mecanica_pref.replace(' ', '_')}"
    if colname in mec_cols.columns:
        user_mec_vector[mec_cols.columns.get_loc(colname)] = 1
else:
    MECHANICS_WEIGHT = 0

user_profile = np.concatenate([
    (user_profile_numeric + pref_numeric_vector) / 2,
    ((user_mec_from_ratings + user_mec_vector) / 2) * MECHANICS_WEIGHT
])

# ============================================================
# 4️⃣ COMPUTE SIMILARITY AND RECOMMEND
# ============================================================

similarities = cosine_similarity([user_profile], X)[0]
df["similarity"] = similarities

df_filtered = df[
    (df["nota_bgg"] >= nota_pref) &
    (df["minplayers"] <= num_jugadors) &
    (df["maxplayers"] >= num_jugadors)
]


if filter_owned:
    df_filtered = df_filtered[df_filtered["own"] == 1]
    df_filtered = df_filtered["itemtype"] != "expansion"]
recommendations = df_filtered.sort_values("similarity", ascending=False).head(50)

# ============================================================
# OUTPUT
# ============================================================

st.subheader("✨ Recomanacions de jocs")

if recommendations.empty:
    st.success("🎉 No hi ha recomanacions noves!")
else:
    st.write(
        recommendations[
            ["nom_del_joc", "similarity", "Mecànica_principal", "pes", "nota_bgg", "playingtime", "minplayers", "maxplayers"]
        ]
    )

# ============================================================
# 6️⃣ CLUSTERING DE JOCS
# ============================================================

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

st.divider()
st.header("🔬 Agrupació de jocs per similitud")

n_clusters = st.slider("Nombre de grups (clusters):", 2, 10, 5, key="n_clusters")

@st.cache_data(show_spinner=False)
def compute_clusters(n: int, matrix_hash: str) -> pd.DataFrame:
    return None  # placeholder, la lògica real és a sota

# Construir matriu de features per clustering (tots els jocs, no només filtrats)
cluster_features = df[numeric_cols].copy()
cluster_features = cluster_features.fillna(cluster_features.median())

# Afegir mecàniques
mec_matrix = mec_cols.values

# Normalitzar numèrics
scaler_cluster = StandardScaler()
X_cluster_numeric = scaler_cluster.fit_transform(cluster_features)
X_cluster = np.hstack([X_cluster_numeric, mec_matrix * 0.5])

# KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster)
df_plot = df[["nom_del_joc", "pes", "nota_bgg", "playingtime", "Mecànica_principal", "minplayers", "maxplayers"]].copy()
df_plot["Cluster"] = cluster_labels.astype(str)

# PCA per reduir a 2D per visualitzar
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_cluster)
df_plot["PC1"] = coords[:, 0]
df_plot["PC2"] = coords[:, 1]

# Variança explicada
var_explicada = pca.explained_variance_ratio_
st.caption(f"Els dos eixos expliquen el {var_explicada[0]*100:.1f}% + {var_explicada[1]*100:.1f}% = {sum(var_explicada)*100:.1f}% de la variabilitat total")

fig_cluster = px.scatter(
    df_plot,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_name="nom_del_joc",
    hover_data={
        "pes": True,
        "nota_bgg": True,
        "playingtime": True,
        "Mecànica_principal": True,
        "PC1": False,
        "PC2": False
    },
    title=f"Agrupació de jocs en {n_clusters} clusters (PCA 2D)",
    color_discrete_sequence=px.colors.qualitative.Set2,
)

fig_cluster.update_traces(marker=dict(size=8, opacity=0.8))
fig_cluster.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#000000"),
    legend_title_text="Grup",
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
)

st.plotly_chart(fig_cluster, use_container_width=True)

# Taula resum per cluster
st.subheader("📊 Característiques mitjanes per grup")
cluster_summary = df_plot.groupby("Cluster").agg(
    Jocs=("nom_del_joc", "count"),
    Pes_mitjà=("pes", "mean"),
    Nota_BGG_mitjana=("nota_bgg", "mean"),
    Durada_mitjana=("playingtime", "mean"),
).round(2).reset_index()
cluster_summary.columns = ["Grup", "Nº Jocs", "Pes mitjà", "Nota BGG", "Durada (min)"]
st.dataframe(cluster_summary, use_container_width=True)


# ============================================================
# 5️⃣ MISUT MEEPLE - RESENYA DEL JOC MÉS RECOMANAT
# ============================================================

@st.cache_data(show_spinner=False)
def get_misutmeeple_summary(game_name: str) -> dict:
    search_url = f"https://misutmeeple.com/?s={game_name.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        link = soup.find("a", href=re.compile(r"misutmeeple\.com/\d{4}/"))
        if not link:
            return {"found": False, "url": "", "summary": "", "image": "", "sello": ""}

        page_url = link["href"]
        r2 = requests.get(page_url, headers=headers, timeout=10)
        soup2 = BeautifulSoup(r2.text, "html.parser")

        paragraphs = soup2.select("article p")[:3]
        summary = "\n\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        if not summary.strip():
            return {"found": False, "url": "", "summary": "", "image": "", "sello": ""}

        image_url = ""
        img_tag = soup2.select_one("article img")
        if img_tag:
            image_url = img_tag.get("src", "") or img_tag.get("data-src", "")

        sello_url = ""
        for img in soup2.select("article img"):
            src = img.get("src", "") or img.get("data-src", "")
            if "sello" in src.lower():
                sello_url = src
                break

        return {"found": True, "url": page_url, "summary": summary, "image": image_url, "sello": sello_url}

    except Exception as e:
        return {"found": False, "url": "", "summary": f"Error: {e}", "image": "", "sello": ""}


st.divider()
st.header("📖 Resenya a Misut Meeple")

if not recommendations.empty:
    top_game = recommendations.iloc[0]["nom_del_joc"]
    st.markdown(f"Cercant informació sobre **{top_game}** a Misut Meeple...")

    with st.spinner(f"Buscant resenya de '{top_game}'..."):
        result = get_misutmeeple_summary(top_game)

    if result["found"]:
        st.success(f"✅ Resenya trobada per a **{top_game}**!")
        st.markdown(f"🔗 [Llegir la resenya completa a Misut Meeple]({result['url']})")
        col1, col2 = st.columns([1, 2])
        with col1:
            if result["image"]:
                st.image(result["image"], use_container_width=True)
            if result["sello"]:
                st.image(result["sello"], use_container_width=True)
        with col2:
            st.markdown(result["summary"])
    else:
        st.info(f"ℹ️ No s'ha trobat cap resenya de **{top_game}** a Misut Meeple.")
else:
    st.info("No hi ha recomanacions per mostrar.")
