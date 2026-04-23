import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import os
import time
import xml.etree.ElementTree as ET

BGG_CACHE_FILE = "pages/bgg_mechanics_cache.csv"

def load_mechanics_cache() -> dict:
    if os.path.exists(BGG_CACHE_FILE):
        cache_df = pd.read_csv(BGG_CACHE_FILE)
        return dict(zip(
            cache_df["objectid"].astype(int),
            cache_df["mechanics"].apply(eval)
        ))
    return {}

def fetch_mechanics_for_ids(object_ids: list, existing_cache: dict) -> dict:
    cache = dict(existing_cache)
    ids_to_fetch = [oid for oid in object_ids if oid not in cache]

    if not ids_to_fetch:
        return cache

    batch_size = 20
    batches = [ids_to_fetch[i:i+batch_size] for i in range(0, len(ids_to_fetch), batch_size)]
    progress = st.progress(0, text="Carregant mecàniques de BGG...")

    for i, batch in enumerate(batches):
        ids_str = ",".join(str(oid) for oid in batch)
        url = f"https://boardgamegeek.com/xmlapi2/thing?id={ids_str}&type=boardgame"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                root = ET.fromstring(r.content)
                for item in root.findall("item"):
                    oid = int(item.get("id"))
                    mechanics = [
                        link.get("value")
                        for link in item.findall("link")
                        if link.get("type") == "boardgamemechanic"
                    ]
                    cache[oid] = mechanics
            elif r.status_code == 429:
                time.sleep(5)
        except Exception:
            pass
        time.sleep(0.5)
        progress.progress((i + 1) / len(batches), text=f"Carregant mecàniques... ({i+1}/{len(batches)})")

    progress.empty()

    # Guardar a disc
    cache_rows = [{"objectid": k, "mechanics": str(v)} for k, v in cache.items()]
    pd.DataFrame(cache_rows).to_csv(BGG_CACHE_FILE, index=False)

    return cache

# ============================================================
# CÀRREGA DE MECÀNIQUES — només es fa UNA vegada per sessió
# ============================================================

if "mechanics_cache" not in st.session_state:
    if "objectid" in df.columns:
        df["objectid"] = pd.to_numeric(df["objectid"], errors="coerce")
        df = df.dropna(subset=["objectid"])
        df["objectid"] = df["objectid"].astype(int)
        object_ids = df["objectid"].tolist()

        disc_cache = load_mechanics_cache()
        full_cache = fetch_mechanics_for_ids(object_ids, disc_cache)
        st.session_state["mechanics_cache"] = full_cache
    else:
        st.session_state["mechanics_cache"] = {}

mechanics_cache = st.session_state["mechanics_cache"]

# Assignar mecàniques a cada fila
if "objectid" in df.columns:
    df["all_mechanics"] = df["objectid"].map(
        lambda oid: mechanics_cache.get(oid, [])
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


st.set_page_config(layout="wide")

st.title("🎯 Recomanador personalitzat de jocs")

# ============================================================
# LOAD COLLECTION DATA
# ============================================================

df = pd.read_csv("pages/collection.csv", sep=",", engine="python")

# Ensure playingtime exists
if "playingtime" not in df.columns:
    st.error("⚠️ La columna 'playingtime' no existeix al CSV!")
else:
    df["playingtime"] = df["playingtime"].fillna(df["playingtime"].median())

# Clean columns
df.columns = df.columns.str.strip()

# Rename if needed
df = df.rename(columns={
    "objectname": "nom_del_joc",
    "avgweight": "pes",
    "average": "nota_bgg",
    "yearpublished": "any_publicació"
})

# Ensure required columns exist
required_cols = ["pes", "nota_bgg", "minplayers", "maxplayers"]
df = df.dropna(subset=required_cols)

# Clean mechanics text
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

# Ensure 'own' exists
if "own" not in df.columns:
    st.warning("⚠️ El CSV no té la columna 'own'. Es crearà amb valor 0.")
    df["own"] = 0

# ============================================================
# CÀRREGA DE MECÀNIQUES COMPLETES DES DE BGG API
# ============================================================

import os

if "objectid" in df.columns:
    df["objectid"] = pd.to_numeric(df["objectid"], errors="coerce").dropna().astype(int)
    object_ids = df["objectid"].dropna().astype(int).tolist()

    with st.spinner("Carregant mecàniques des de BGG (primera vegada pot trigar uns minuts)..."):
        mechanics_cache = load_mechanics_cache()
        mechanics_cache = fetch_mechanics_for_ids(object_ids, mechanics_cache)

    # Assignar mecàniques a cada joc
    df["all_mechanics"] = df["objectid"].map(
        lambda oid: mechanics_cache.get(int(oid), []) if pd.notna(oid) else []
    )
else:
    st.warning("⚠️ No s'ha trobat la columna 'objectid'. S'usarà només la mecànica principal.")
    df["all_mechanics"] = df["Mecànica_principal"].apply(lambda x: [x])

# One-hot encoding amb totes les mecàniques
all_mec_names = sorted(set(m for mecs in df["all_mechanics"] for m in mecs))
mec_data = {}
for mec in all_mec_names:
    col = f"mec_{mec.replace(' ', '_')}"
    mec_data[col] = df["all_mechanics"].apply(lambda mecs: 1 if mec in mecs else 0)

mec_cols = pd.DataFrame(mec_data, index=df.index)
df = pd.concat([df, mec_cols], axis=1)

# ============================================================
# 1️⃣ USER PREFERENCES
# ============================================================

st.header("🧩 Preferències del jugador")

pes_pref = st.slider("Pes preferit (complexitat):", 0.0, 5.0, 2.5, 0.1)
nota_pref = st.slider("Nota mínima BGG:", 0.0, 10.0, 6.5, 0.1)

num_jugadors = st.slider("Nombre de jugadors preferit:", 1, 10, 3)

durada_pref = st.slider("Durada preferida (minuts):", 10, 300, 60, 5)

# Mecànica preferida
mecaniques = ["Qualsevol"] + all_mec_names
mecanica_pref = st.selectbox("Mecànica preferida:", mecaniques)


# ============================================================
# 2️⃣ USER RATINGS FOR SAMPLE GAMES
# ============================================================

st.subheader("⚙️ Opcions del recomanador")
filter_owned = st.checkbox("Només mostrar jocs que tinc en propietat", value=False)

st.header("⭐ Avalua alguns jocs")

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
    col1, col2 = st.columns([3,1])

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

PLAYTIME_WEIGHT = 2.1
MECHANICS_WEIGHT = 1.95

numeric_cols = ["pes", "nota_bgg", "minplayers", "maxplayers", "playingtime"]

scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[numeric_cols])

# Apply weight to playingtime
X_numeric[:, -1] *= PLAYTIME_WEIGHT

X = np.hstack([X_numeric, mec_cols.values * MECHANICS_WEIGHT])

# Rated games
rated_games = df[df["nom_del_joc"].isin([g for g in user_ratings if not ignore_flags[g]])].copy()
rated_games["user_rating"] = rated_games["nom_del_joc"].map(user_ratings)

# Numeric profile
user_profile_numeric = np.average(
    scaler.transform(rated_games[numeric_cols]),
    axis=0,
    weights=rated_games["user_rating"]
)

# Mecànica profile
rated_mec_matrix = mec_cols.loc[rated_games.index].values
user_mec_from_ratings = np.average(
    rated_mec_matrix,
    axis=0,
    weights=rated_games["user_rating"]
)

# Explicit preferences
pref_df = pd.DataFrame([{
    "pes": pes_pref,
    "nota_bgg": nota_pref,
    "minplayers": num_jugadors,
    "maxplayers": num_jugadors,
    "playingtime": durada_pref
}])[numeric_cols]

pref_numeric_vector = scaler.transform(pref_df)[0]
pref_numeric_vector[-1] *= PLAYTIME_WEIGHT

# Mecànica explícita
user_mec_vector = np.zeros(len(mec_cols.columns))
if mecanica_pref != "Qualsevol":
    colname = f"mec_{mecanica_pref.replace(' ', '_')}"
    if colname in mec_cols.columns:
        user_mec_vector[mec_cols.columns.get_loc(colname)] = 1
else:
    MECHANICS_WEIGHT = 0

# Final user profile
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

recommendations = df_filtered.sort_values("similarity", ascending=False).head(50)

# Apply OWN filter
if filter_owned:
    recommendations = recommendations[recommendations["own"] == 1]

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
# 5️⃣ MISUT MEEPLE - RESENYA DEL JOC MÉS RECOMANAT
# ============================================================

import requests
from bs4 import BeautifulSoup
import re

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

        # Extraer intro del artículo (primeros párrafos)
        paragraphs = soup2.select("article p")[:3]
        summary = "\n\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

        if not summary.strip():
            return {"found": False, "url": "", "summary": "", "image": "", "sello": ""}

        # Imagen principal (primera imagen del artículo)
        image_url = ""
        img_tag = soup2.select_one("article img")
        if img_tag:
            image_url = img_tag.get("src", "") or img_tag.get("data-src", "")

        # Sello de valoración (imagen cuyo src contiene "sello")
        sello_url = ""
        for img in soup2.select("article img"):
            src = img.get("src", "") or img.get("data-src", "")
            if "sello" in src.lower():
                sello_url = src
                break

        return {
            "found": True,
            "url": page_url,
            "summary": summary,
            "image": image_url,
            "sello": sello_url
        }

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

        if result["image"]:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(result["image"], use_container_width=True)
            with col2:
                st.markdown(result["summary"])
        else:
            st.markdown(result["summary"])
    else:
        st.info(f"ℹ️ No s'ha trobat cap resenya de **{top_game}** a Misut Meeple.")
else:
    st.info("No hi ha recomanacions per mostrar.")
