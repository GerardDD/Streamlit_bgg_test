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

# ============================================================
# 1️⃣ USER PREFERENCES
# ============================================================

st.header("🧩 Preferències del jugador")

pes_pref = st.slider("Pes preferit (complexitat):", 1.0, 5.0, 2.5, 0.1)
nota_pref = st.slider("Nota mínima BGG:", 1.0, 10.0, 6.5, 0.1)

num_jugadors = st.slider(
    "Nombre de jugadors preferit:",
    1, 10, 3
)

# ============================================================
# 2️⃣ USER RATINGS FOR SAMPLE GAMES (WITH RESHUFFLE)
# ============================================================

st.header("⭐ Avalua alguns jocs")

# Initialize sample in session_state
if "sample_games" not in st.session_state:
    st.session_state.sample_games = df.sample(10, random_state=None)[
        ["nom_del_joc", "pes", "nota_bgg", "minplayers", "maxplayers"]
    ]

# Reshuffle button
if st.button("🔄 Tornar a mostrar altres jocs"):
    st.session_state.sample_games = df.sample(10, random_state=None)[
        ["nom_del_joc", "pes", "nota_bgg", "minplayers", "maxplayers"]
    ]

sample_games = st.session_state.sample_games

user_ratings = {}

for idx, row in sample_games.iterrows():
    rating = st.slider(
        f"Valoració per **{row['nom_del_joc']}**:",
        1, 10, 5,
        key=f"rating_{row['nom_del_joc']}"
    )
    user_ratings[row["nom_del_joc"]] = rating

# ============================================================
# 3️⃣ BUILD USER PROFILE VECTOR
# ============================================================

feature_cols = ["pes", "nota_bgg", "minplayers", "maxplayers"]

# Feature matrix
features = df[feature_cols].copy()

scaler = StandardScaler()
X = scaler.fit_transform(features)

# Rated games
rated_games = df[df["nom_del_joc"].isin(user_ratings.keys())].copy()
rated_games["user_rating"] = rated_games["nom_del_joc"].map(user_ratings)

# Weighted average of rated games
user_profile = np.average(
    scaler.transform(rated_games[feature_cols]),
    axis=0,
    weights=rated_games["user_rating"]
)

# Explicit preferences → convert to same feature space
pref_df = pd.DataFrame([{
    "pes": pes_pref,
    "nota_bgg": nota_pref,
    "minplayers": num_jugadors,
    "maxplayers": num_jugadors
}])[feature_cols]

pref_vector = scaler.transform(pref_df)[0]

# Combine both signals
user_profile = (user_profile + pref_vector) / 2

# ============================================================
# 4️⃣ COMPUTE SIMILARITY AND RECOMMEND
# ============================================================

similarities = cosine_similarity([user_profile], X)[0]
df["similarity"] = similarities

# Filter by preferences
df_filtered = df[
    (df["nota_bgg"] >= nota_pref) &
    (df["minplayers"] <= num_jugadors) &
    (df["maxplayers"] >= num_jugadors)
]

# Sort by similarity
recommendations = df_filtered.sort_values("similarity", ascending=False).head(10)

# ============================================================
# OUTPUT
# ============================================================

st.header("🎉 Recomanacions per tu")

st.dataframe(
    recommendations[["nom_del_joc", "pes", "nota_bgg", "minplayers", "maxplayers", "similarity"]]
)
