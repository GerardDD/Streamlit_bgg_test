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
    st.session_state.sample_games = df.sample(6, random_state=None)[
        ["nom_del_joc", "pes", "nota_bgg", "minplayers", "maxplayers"]
    ]

# Reshuffle button
if st.button("🔄 Tornar a mostrar altres jocs"):
    st.session_state.sample_games = df.sample(6, random_state=None)[
        ["nom_del_joc", "pes", "nota_bgg", "minplayers", "maxplayers"]
    ]

sample_games
