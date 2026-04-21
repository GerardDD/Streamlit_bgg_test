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

df = df.rename(columns={"comment": "Mecànica_principal"})

# One-hot encoding
mec_cols = pd.get_dummies(df["Mecànica_principal"], prefix="mec")
df = pd.concat([df, mec_cols], axis=1)

# ============================================================
# 1️⃣ USER PREFERENCES
# ============================================================

st.header("🧩 Preferències del jugador")

pes_pref = st.slider("Pes preferit (complexitat):", 0.0, 5.0, 2.5, 0.1)
nota_pref = st.slider("Nota mínima BGG:", 0.0, 10.0, 6.5, 0.1)

num_jugadors = st.slider(
    "Nombre de jugadors preferit:",
    1, 10, 3
)

# Mecànica preferida
mecaniques = sorted(df["Mecànica_principal"].unique())
mecanica_pref = st.selectbox("Mecànica preferida:", mecaniques)

# ============================================================
# 2️⃣ USER RATINGS FOR SAMPLE GAMES (WITH RESHUFFLE)
# ============================================================

st.header("⭐ Avalua alguns jocs")

if "sample_games" not in st.session_state:
    st.session_state.sample_games = df.sample(10, random_state=None)[
        ["nom_del_joc", "pes", "nota_bgg", "minplayers", "maxplayers", "Mecànica_principal"]
    ]

if st.button("🔄 Tornar a mostrar altres jocs"):
    st.session_state.sample_games = df.sample(10, random_state=None)[
        ["nom_del_joc", "pes", "nota_bgg", "minplayers", "maxplayers", "Mecànica_principal"]
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

MECHANICS_WEIGHT = 2.0

numeric_cols = ["pes", "nota_bgg", "minplayers", "maxplayers"]
feature_cols = numeric_cols + list(mec_cols.columns)

# Scale numeric features
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[numeric_cols])

# Full feature matrix
X = np.hstack([X_numeric, mec_cols.values])

# Rated games
rated_games = df[df["nom_del_joc"].isin(user_ratings.keys())].copy()
rated_games["user_rating"] = rated_games["nom_del_joc"].map(user_ratings)

# Numeric profile from ratings
user_profile_numeric = np.average(
    scaler.transform(rated_games[numeric_cols]),
    axis=0,
    weights=rated_games["user_rating"]
)

# Mecànica profile from ratings
rated_mec_matrix = mec_cols.loc[rated_games.index].values
user_mec_from_ratings = np.average(
    rated_mec_matrix,
    axis=0,
    weights=rated_games["user_rating"]
)

# Explicit preferences → numeric
pref_df = pd.DataFrame([{
    "pes": pes_pref,
    "nota_bgg": nota_pref,
    "minplayers": num_jugadors,
    "maxplayers": num_jugadors
}])[numeric_cols]

pref_numeric_vector = scaler.transform(pref_df)[0]

# Explicit preferences → mecànica
user_mec_vector = np.zeros(len(mec_cols.columns))
if f"mec_{mecanica_pref}" in mec_cols.columns:
    idx = mec_cols.columns.get_loc(f"mec_{mecanica_pref}")
    user_mec_vector[idx] = 1

# Combine numeric + mecànica
user_profile = np.concatenate([
    (user_profile_numeric + pref_numeric_vector) / 2,
    ((user_mec_from_ratings + user_mec_vector) / 2) * MECHANICS_WEIGHT
])




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
    recommendations[[
        "nom_del_joc", "pes", "nota_bgg",
        "minplayers", "maxplayers",
        "Mecànica_principal", "similarity"
    ]]
)
