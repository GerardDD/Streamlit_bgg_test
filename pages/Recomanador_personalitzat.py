import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")

st.title("🎯 Recomanador personalitzat de jocs")

# Load your main collection
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

# Drop games missing key info
df = df.dropna(subset=["pes", "nota_bgg", "any_publicació"])

# ============================================================
# 1️⃣ USER PREFERENCES
# ============================================================

st.header("🧩 Preferències del jugador")

pes_pref = st.slider("Pes preferit (complexitat):", 1.0, 5.0, 2.5, 0.1)
nota_pref = st.slider("Nota mínima BGG:", 1.0, 10.0, 6.5, 0.1)
any_range = st.slider(
    "Rang d'anys de publicació:",
    int(df["any_publicació"].min()),
    int(df["any_publicació"].max()),
    (2000, 2025)
)

# ============================================================
# 2️⃣ USER RATINGS FOR SAMPLE GAMES
# ============================================================

st.header("⭐ Avalua alguns jocs")

sample_games = df.sample(6, random_state=42)[["nom_del_joc", "pes", "nota_bgg", "any_publicació"]]

user_ratings = {}

for idx, row in sample_games.iterrows():
    rating = st.slider(
        f"Valoració per **{row['nom_del_joc']}**:",
        1, 10, 5
    )
    user_ratings[row["nom_del_joc"]] = rating

# ============================================================
# 3️⃣ BUILD USER PROFILE VECTOR
# ============================================================

# Convert df to feature matrix
features = df[["pes", "nota_bgg", "any_publicació"]].copy()

scaler = StandardScaler()
X = scaler.fit_transform(features)

# Build user profile from rated games
rated_games = df[df["nom_del_joc"].isin(user_ratings.keys())].copy()
rated_games["user_rating"] = rated_games["nom_del_joc"].map(user_ratings)

# Weighted average of features
user_profile = np.average(
    scaler.transform(rated_games[["pes", "nota_bgg", "any_publicació"]]),
    axis=0,
    weights=rated_games["user_rating"]
)

# Add explicit preferences as extra weight
pref_vector = scaler.transform(pd.DataFrame([{
    "pes": pes_pref,
    "nota_bgg": nota_pref,
    "any_publicació": np.mean(any_range)
}]))

# Combine both
user_profile = (user_profile + pref_vector[0]) / 2

# ============================================================
# 4️⃣ COMPUTE SIMILARITY AND RECOMMEND
# ============================================================

similarities = cosine_similarity([user_profile], X)[0]

df["similarity"] = similarities

# Filter by preferences
df_filtered = df[
    (df["nota_bgg"] >= nota_pref) &
    (df["any_publicació"].between(any_range[0], any_range[1]))
]

# Sort by similarity
recommendations = df_filtered.sort_values("similarity", ascending=False).head(10)

st.header("🎉 Recomanacions per tu")

st.dataframe(
    recommendations[["nom_del_joc", "pes", "nota_bgg", "any_publicació", "similarity"]]
)
