import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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

# One-hot encoding
mec_cols = pd.get_dummies(df["Mecànica_principal"], prefix="mec")
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
mecaniques = ["Qualsevol"] + sorted(df["Mecànica_principal"].unique())
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
    colname = f"mec_{mecanica_pref}"
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
# 5️⃣ MISUT MEEPLE - RESEÑA DEL JUEGO MÁS RECOMENDADO
# ============================================================

import anthropic
import re

@st.cache_data(show_spinner=False)
def get_misutmeeple_summary(game_name: str) -> dict:
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    prompt = f"""Busca en el blog Misut Meeple (misutmeeple.com) una reseña del juego de mesa "{game_name}".

Si encuentras la reseña:
1. Indica la URL exacta de la página.
2. Haz un resumen breve (3-5 frases) con: de qué trata el juego, mecánica principal, número de jugadores, duración aproximada y valoración general del autor.

Si NO encuentras ninguna reseña de este juego en misutmeeple.com, responde exactamente con: NO_ENCONTRADO

Responde en catalán."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}]
    )

    full_text = " ".join(
        block.text for block in response.content if hasattr(block, "text")
    ).strip()

    if "NO_ENCONTRADO" in full_text:
        return {"found": False, "url": "", "summary": ""}

    url_match = re.search(r"https?://misutmeeple\.com/\S+", full_text)
    url = url_match.group(0).rstrip(".,)") if url_match else ""
    summary = full_text.replace(url, "").strip(" .,\n")

    return {"found": True, "url": url, "summary": summary}


st.divider()
st.header("📖 Resenya a Misut Meeple")

if not recommendations.empty:
    top_game = recommendations.iloc[0]["nom_del_joc"]
    st.markdown(f"Cercant informació sobre **{top_game}** a Misut Meeple...")

    with st.spinner(f"Buscant resenya de '{top_game}'..."):
        result = get_misutmeeple_summary(top_game)

    if result["found"]:
        st.success(f"✅ Resenya trobada per a **{top_game}**!")
        if result["url"]:
            st.markdown(f"🔗 [Llegir la resenya completa a Misut Meeple]({result['url']})")
        st.markdown(result["summary"])
    else:
        st.info(f"ℹ️ No s'ha trobat cap resenya de **{top_game}** a Misut Meeple.")
else:
    st.info("No hi ha recomanacions per mostrar.")
