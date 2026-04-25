import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(layout="centered")

# ============================================================
# CSS RESPONSIVE
# ============================================================
st.markdown("""
<style>
/* ── Desktop: amplada màxima generosa ────────────────────── */
.block-container {
    max-width: 1200px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ── En mòbil: eliminar padding lateral excessiu ─────────── */
@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }

    /* Sliders: zona tàctil més gran */
    .stSlider > div > div > div {
        height: 44px !important;
    }
    .stSlider [data-testid="stThumbValue"] {
        font-size: 1rem !important;
    }

    /* Botons */
    .stButton > button {
        width: 100% !important;
        min-height: 44px !important;
        font-size: 1rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Checkboxes */
    .stCheckbox label {
        font-size: 1rem !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        min-height: 44px !important;
        font-size: 1rem !important;
    }

    /* Columnes: apilament vertical en mòbil */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* Taules: scroll horitzontal */
    .stDataFrame {
        overflow-x: auto !important;
    }

    /* Imatges */
    img {
        max-width: 100% !important;
        height: auto !important;
    }

    .stCaption, small {
        font-size: 0.85rem !important;
    }
}

/* ── Cards de valoració: separació visual ────────────────── */
[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
    border-bottom: 1px solid rgba(128,128,128,0.15);
    padding-bottom: 0.5rem;
    margin-bottom: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

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
    "average":   "nota_bgg",
    "yearpublished": "any_publicació"
})

required_cols = ["pes", "nota_bgg", "minplayers", "maxplayers"]
df = df.dropna(subset=required_cols)

# Excloure jocs amb pes 0 (sense dades de complexitat a BGG)
df = df[df["pes"] > 0]

df["comment"] = df["comment"].replace(r"^\s*$", "No informat", regex=True)
df["comment"] = df["comment"].fillna("No informat")
df["comment"] = df["comment"].replace("Selecció accions",   "Selecció d'accions", regex=True)
df["comment"] = df["comment"].replace("Car-driven",         "Card-driven",        regex=True)
df["comment"] = df["comment"].replace("Col·locació daus",   "Col·locació de daus",regex=True)
df["comment"] = df["comment"].replace("Construcció motor",  "Construcció de motor",regex=True)
df["comment"] = df["comment"].replace("Draft pùblic",       "Draft públic",       regex=True)
df["comment"] = df["comment"].replace("Gestió de la mà",   "Gestió de mà",       regex=True)
df["comment"] = df["comment"].replace("Pseudo-abstracte",   "Pseudo-Abstracte",   regex=True)
df["comment"] = df["comment"].replace("Pseudo-wargame",     "Pseudo-Wargame",     regex=True)
df["comment"] = df["comment"].replace("Pseudo-wargames",    "Pseudo-Wargame",     regex=True)
df["comment"] = df["comment"].replace("Pseudo-Wargames",    "Pseudo-Wargame",     regex=True)
df = df.rename(columns={"comment": "Mecànica_principal"})

if "own" not in df.columns:
    st.warning("⚠️ El CSV no té la columna 'own'. Es crearà amb valor 0.")
    df["own"] = 0
else:
    df["own"] = pd.to_numeric(df["own"], errors="coerce").fillna(0).astype(int)

# ============================================================
# MILLORA 1 — numplays i rating personal com a features
# ============================================================
# numplays: quantes vegades has jugat → senyal implícita de preferència
# rating:   la teva valoració personal (columna "rating" del CSV de BGG)
#           diferent de "average" que és la mitjana de la comunitat

if "numplays" in df.columns:
    df["numplays"] = pd.to_numeric(df["numplays"], errors="coerce").fillna(0)
    # Transformació log per reduir l'efecte d'outliers (1 partida vs 50)
    df["log_numplays"] = np.log1p(df["numplays"])
else:
    df["numplays"]     = 0
    df["log_numplays"] = 0.0

if "rating" in df.columns:
    # La columna "rating" del CSV de BGG és la teva valoració personal
    # Pot tenir valors "N/A" o buits per jocs no valorats
    df["rating_personal"] = pd.to_numeric(df["rating"], errors="coerce")
    has_personal_ratings  = df["rating_personal"].notna().sum()
else:
    df["rating_personal"] = np.nan
    has_personal_ratings  = 0

# ============================================================
# CÀRREGA DE MECÀNIQUES DES DE CACHÉ LOCAL
# ============================================================

if "mechanics_cache" not in st.session_state:
    disc_cache = load_mechanics_cache()
    st.session_state["mechanics_cache"] = disc_cache

mechanics_cache = st.session_state["mechanics_cache"]

# Estat de variació de recomanació (s'incrementa amb el botó)
if "variation_idx" not in st.session_state:
    st.session_state["variation_idx"] = 0
if "excluded_games" not in st.session_state:
    st.session_state["excluded_games"] = []

if "objectid" in df.columns:
    df["objectid"]      = pd.to_numeric(df["objectid"], errors="coerce")
    df["all_mechanics"] = df["objectid"].map(
        lambda oid: mechanics_cache.get(int(oid), []) if pd.notna(oid) else []
    )
else:
    df["all_mechanics"] = df["Mecànica_principal"].apply(lambda x: [x])

# One-hot encoding mecàniques
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

if st.button("↩️ Restablir preferències per defecte"):
    st.session_state["pes_pref"]      = 2.5
    st.session_state["nota_pref"]     = 6.5
    st.session_state["num_jugadors"]  = 3
    st.session_state["durada_pref"]   = 60
    st.session_state["mecanica_pref"] = "Qualsevol"
    st.rerun()

pes_pref     = st.slider("Pes preferit (complexitat):", 0.0, 5.0,
    st.session_state.get("pes_pref", 2.5), 0.1, key="pes_pref")
nota_pref    = st.slider("Nota mínima BGG:", 0.0, 10.0,
    st.session_state.get("nota_pref", 6.5), 0.1, key="nota_pref")
num_jugadors = st.slider("Nombre de jugadors preferit:", 1, 10,
    st.session_state.get("num_jugadors", 3), key="num_jugadors")
durada_pref  = st.slider("Durada preferida (minuts):", 10, 300,
    st.session_state.get("durada_pref", 60), 5, key="durada_pref")

mecaniques   = ["Qualsevol"] + all_mec_names
mecanica_pref = st.selectbox(
    "Mecànica preferida:", mecaniques,
    index=mecaniques.index(st.session_state.get("mecanica_pref", "Qualsevol"))
          if st.session_state.get("mecanica_pref", "Qualsevol") in mecaniques else 0,
    key="mecanica_pref"
)

# ============================================================
# 2️⃣ USER RATINGS FOR SAMPLE GAMES
# ============================================================

st.subheader("⚙️ Opcions del recomanador")
filter_owned = st.checkbox("Només mostrar jocs que tinc en propietat", value=False)

st.header("⭐ Avalua alguns jocs")
st.subheader("🚥 Puntua del 1 al 10 fins a quin punt s'ajusten al que estàs buscant")

def select_strategic_games(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Selecciona jocs estratègicament per maximitzar la informació del perfil:
    - 1 representant per cluster (cobreix tot l'espai de jocs)
    - 1 per mecànica principal de les més freqüents (diversitat de mecàniques)
    - Resta: jocs amb nota BGG alta i pes variat (qualitat + diversitat)
    Els representants de cluster es trien com el joc més proper al centroide.
    """
    cols_base = ["pes", "nota_bgg", "minplayers", "maxplayers", "playingtime", "log_numplays"]
    cols_available = [c for c in cols_base if c in df.columns]

    selected_ids = set()
    selected_rows = []

    # ── 1. Un representant per cluster (fins a 5) ─────────────────
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler as _SS
        X_s = _SS().fit_transform(df[cols_available].fillna(0))
        n_cl = min(5, len(df) // 3)
        km = KMeans(n_clusters=n_cl, random_state=42, n_init=5)
        labels = km.fit_predict(X_s)
        df_tmp = df.copy()
        df_tmp["_cluster"] = labels
        df_tmp["_X0"] = X_s[:, 0]
        df_tmp["_X1"] = X_s[:, 1]
        for cl in range(n_cl):
            cl_mask = df_tmp["_cluster"] == cl
            cl_df   = df_tmp[cl_mask]
            # joc més proper al centroide del cluster
            centroid = km.cluster_centers_[cl]
            dists = np.linalg.norm(X_s[cl_mask.values] - centroid, axis=1)
            rep_idx = cl_df.index[np.argmin(dists)]
            if rep_idx not in selected_ids:
                selected_ids.add(rep_idx)
                selected_rows.append(df.loc[rep_idx])
    except Exception:
        pass

    # ── 2. Un per mecànica principal (les més freqüents) ──────────
    top_mecs = df["Mecànica_principal"].value_counts().head(8).index.tolist()
    for mec in top_mecs:
        if len(selected_rows) >= n:
            break
        candidates = df[
            (df["Mecànica_principal"] == mec) &
            (~df.index.isin(selected_ids))
        ]
        if candidates.empty:
            continue
        # Millor valorat de la mecànica
        rep = candidates.sort_values("nota_bgg", ascending=False).iloc[0]
        selected_ids.add(rep.name)
        selected_rows.append(rep)

    # ── 3. Completar amb jocs ben valorats no repetits ────────────
    remaining = df[~df.index.isin(selected_ids)].sort_values(
        "nota_bgg", ascending=False
    )
    for _, row in remaining.iterrows():
        if len(selected_rows) >= n:
            break
        selected_rows.append(row)
        selected_ids.add(row.name)

    result = pd.DataFrame(selected_rows[:n])
    return result[["nom_del_joc", "pes", "nota_bgg", "minplayers", "maxplayers", "Mecànica_principal"]]


if "sample_games" not in st.session_state:
    st.session_state.sample_games = select_strategic_games(df, n=10)

col_refresh1, col_refresh2 = st.columns([1, 1])
with col_refresh1:
    if st.button("🎯 Regenerar selecció estratègica", use_container_width=True):
        st.session_state.sample_games = select_strategic_games(df, n=10)
        # Netejar valoracions anteriors
        for name in st.session_state.sample_games["nom_del_joc"]:
            st.session_state.pop(f"rating_{name}", None)
        st.rerun()
with col_refresh2:
    if st.button("🔀 Selecció aleatòria", use_container_width=True):
        st.session_state.sample_games = df.sample(10)[
            ["nom_del_joc", "pes", "nota_bgg", "minplayers", "maxplayers", "Mecànica_principal"]
        ]
        for name in st.session_state.sample_games["nom_del_joc"]:
            st.session_state.pop(f"rating_{name}", None)
        st.rerun()

st.caption(
    "🎯 Els jocs mostrats han estat seleccionats estratègicament per cobrir "
    "diferents tipus de jocs i mecàniques, maximitzant la informació del teu perfil."
)

sample_games = st.session_state.sample_games
ignore_flags = {}
user_ratings = {}

# CSS per als botons de valoració
st.markdown("""
<style>
/* Botons de valoració numerica */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    padding: 0.35rem 0 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    min-height: 44px !important;
    border-radius: 8px !important;
}
/* Botó seleccionat: destacar en taronja */
div[data-testid="stHorizontalBlock"] button[kind="primary"] {
    background-color: #E8703A !important;
    color: white !important;
    padding: 0.35rem 0 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    min-height: 44px !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

for idx, row in sample_games.iterrows():
    game_name = row["nom_del_joc"]
    rating_key = f"rating_{game_name}"
    ignore_key = f"ignore_{game_name}"

    # Valor actual (per defecte 5)
    if rating_key not in st.session_state:
        st.session_state[rating_key] = 5

    with st.container():
        c_name, c_ignore = st.columns([6, 1])
        with c_name:
            st.markdown(
                f"**{game_name}** "
                f"<span style='color:gray;font-size:0.85rem'>"
                f"({row['Mecànica_principal']} · ⚖️{row['pes']:.1f} · ⭐{row['nota_bgg']:.1f})"
                f"</span>",
                unsafe_allow_html=True
            )
        with c_ignore:
            ignore = st.checkbox("✗", key=ignore_key, help="Ignorar aquest joc")

        # 10 botons en una fila — cada botó assigna la valoració
        btn_cols = st.columns(10)
        current_val = st.session_state[rating_key]
        for i, bcol in enumerate(btn_cols):
            val = i + 1
            label = str(val)
            btn_type = "primary" if current_val == val else "secondary"
            with bcol:
                if st.button(label, key=f"btn_{game_name}_{val}", type=btn_type):
                    st.session_state[rating_key] = val
                    st.rerun()

        st.markdown("<hr style='margin:0.5rem 0;opacity:0.15'>", unsafe_allow_html=True)

    user_ratings[game_name]  = st.session_state[rating_key]
    ignore_flags[game_name]  = st.session_state.get(ignore_key, False)

# ============================================================
# 3️⃣ BUILD USER PROFILE VECTOR
# ============================================================

# ── Pesos base ───────────────────────────────────────────────
# La variació modifica els pesos per explorar zones diferents de l'espai
_var = st.session_state["variation_idx"]

# Cada variació desplaça l'èmfasi entre dimensions:
# 0 → configuració base
# 1 → èmfasi en mecàniques
# 2 → èmfasi en durada i complexitat
# 3 → èmfasi en nota BGG
# 4 → combinació mecàniques + nota
# (cicle de 5 variacions, després torna a explorar)
_var_cycle = _var % 5

_weight_profiles = {
    #              PLAYTIME  MECH   NUMPLAYS  RATING
    0: dict(pt=2.0, mec=2.3, np=1.5, rt=2.0),  # base
    1: dict(pt=1.0, mec=4.0, np=1.0, rt=1.5),  # èmfasi mecàniques
    2: dict(pt=3.5, mec=1.5, np=1.0, rt=1.5),  # èmfasi durada/complexitat
    3: dict(pt=1.5, mec=1.5, np=1.0, rt=3.5),  # èmfasi nota BGG
    4: dict(pt=1.0, mec=3.5, np=2.0, rt=3.0),  # mecàniques + nota
}
_wp = _weight_profiles[_var_cycle]

PLAYTIME_WEIGHT  = _wp["pt"]
MECHANICS_WEIGHT = _wp["mec"]
NUMPLAYS_WEIGHT  = _wp["np"]
RATING_WEIGHT    = _wp["rt"]

# ── Espai A: features per a les preferències EXPLÍCITES (sliders)
# Només les 5 columnes que l'usuari pot expressar directament
numeric_cols = ["pes", "nota_bgg", "minplayers", "maxplayers", "playingtime"]

# ── Espai B: features extres per enriquir el perfil DES DELS JOCS VALORATS
# log_numplays i rating_personal NO entren al vector de preferències explícites
# perquè l'usuari no pot expressar "quantes vegades vol haver jugat" un joc nou
extra_cols = ["log_numplays"]
use_personal_rating = has_personal_ratings > 0
if use_personal_rating:
    median_rating = df["rating_personal"].median()
    df["rating_personal_imp"] = df["rating_personal"].fillna(median_rating)
    extra_cols = extra_cols + ["rating_personal_imp"]
    st.caption(
        f"ℹ️ S'han trobat **{has_personal_ratings}** valoracions personals teves al CSV. "
        f"S'usaran per enriquir el perfil de recomanació."
    )

all_num_cols = numeric_cols + extra_cols

# ── Escalar UNA sola vegada (sense aplicar pesos aquí)
scaler  = StandardScaler()
X_num   = scaler.fit_transform(df[all_num_cols])

# Aplicar pesos a la matriu de jocs
X_num_w = X_num.copy()
X_num_w[:, all_num_cols.index("playingtime")] *= PLAYTIME_WEIGHT
X_num_w[:, all_num_cols.index("log_numplays")] *= NUMPLAYS_WEIGHT
if use_personal_rating:
    X_num_w[:, all_num_cols.index("rating_personal_imp")] *= RATING_WEIGHT

X = np.hstack([X_num_w, mec_cols.values * MECHANICS_WEIGHT])

# ── Perfil des dels jocs valorats als sliders (usa TOTS els features)
rated_names = [g for g in user_ratings if not ignore_flags[g]]
rated_games = df[df["nom_del_joc"].isin(rated_names)].copy()
rated_games["user_rating"] = rated_games["nom_del_joc"].map(user_ratings)

rated_num_scaled = scaler.transform(rated_games[all_num_cols])
rated_num_scaled[:, all_num_cols.index("playingtime")] *= PLAYTIME_WEIGHT
rated_num_scaled[:, all_num_cols.index("log_numplays")] *= NUMPLAYS_WEIGHT
if use_personal_rating:
    rated_num_scaled[:, all_num_cols.index("rating_personal_imp")] *= RATING_WEIGHT

user_profile_from_ratings = np.average(
    rated_num_scaled, axis=0, weights=rated_games["user_rating"]
)

rated_mec_matrix      = mec_cols.loc[rated_games.index].values
user_mec_from_ratings = np.average(
    rated_mec_matrix, axis=0, weights=rated_games["user_rating"]
)

# ── Perfil des de les preferències EXPLÍCITES (només numeric_cols base)
# log_numplays i rating_personal s'ometen aquí — valor 0 escalat = neutre
pref_row = {
    "pes": pes_pref, "nota_bgg": nota_pref,
    "minplayers": num_jugadors, "maxplayers": num_jugadors,
    "playingtime": durada_pref,
    "log_numplays": float(df["log_numplays"].median()),
}
if use_personal_rating:
    pref_row["rating_personal_imp"] = median_rating

pref_scaled = scaler.transform(pd.DataFrame([pref_row])[all_num_cols])[0]
pref_scaled[all_num_cols.index("playingtime")] *= PLAYTIME_WEIGHT
# log_numplays i rating NO es ponderen al vector de prefs — evita soroll
pref_scaled[all_num_cols.index("log_numplays")] = 0.0
if use_personal_rating:
    pref_scaled[all_num_cols.index("rating_personal_imp")] = 0.0

# ── Mecànica explícita
user_mec_vector = np.zeros(len(mec_cols.columns))
if mecanica_pref != "Qualsevol":
    colname = f"mec_{mecanica_pref.replace(' ', '_')}"
    if colname in mec_cols.columns:
        user_mec_vector[mec_cols.columns.get_loc(colname)] = 1
else:
    MECHANICS_WEIGHT = 0

# ── Combinació final: 60% valoracions del usuari + 40% preferències explícites
# Les valoracions pesen més perquè contenen informació real de comportament
RATING_BLEND = 0.6
PREF_BLEND   = 0.4

final_num = (user_profile_from_ratings * RATING_BLEND +
             pref_scaled              * PREF_BLEND)
final_mec = ((user_mec_from_ratings * RATING_BLEND +
              user_mec_vector       * PREF_BLEND) * MECHANICS_WEIGHT)

user_profile = np.concatenate([final_num, final_mec])

# ============================================================
# 4️⃣ COMPUTE SIMILARITY AND RECOMMEND
# ============================================================

similarities   = cosine_similarity([user_profile], X)[0]
df["similarity"] = similarities

df_filtered = df[
    (df["nota_bgg"]    >= nota_pref) &
    (df["minplayers"]  <= num_jugadors) &
    (df["maxplayers"]  >= num_jugadors)
]

if filter_owned:
    df_filtered = df_filtered[df_filtered["own"] == 1]
    df_filtered = df_filtered[df_filtered["itemtype"] != "expansion"]

recommendations = df_filtered.sort_values("similarity", ascending=False).head(50)

# ============================================================
# OUTPUT
# ============================================================

st.subheader("✨ Recomanacions de jocs")

# Excloure jocs ja suggerits en variacions anteriors
excluded = st.session_state.get("excluded_games", [])
recommendations_display = recommendations[~recommendations["nom_del_joc"].isin(excluded)]

# Si hem esgotat opcions, netejem exclusions i tornem a mostrar tot
if recommendations_display.empty and not recommendations.empty:
    st.session_state["excluded_games"] = []
    excluded = []
    recommendations_display = recommendations

if recommendations.empty:
    st.success("🎉 No hi ha recomanacions noves!")
else:
    display_cols = ["nom_del_joc", "similarity", "Mecànica_principal",
                    "pes", "nota_bgg", "playingtime", "minplayers", "maxplayers", "numplays"]
    if use_personal_rating:
        display_cols.append("rating_personal")
    st.write(recommendations_display[[c for c in display_cols if c in recommendations_display.columns]])

    st.markdown("---")

    # ── Etiquetes de variació ─────────────────────────────────────
    _var_labels = {
        0: "configuració base",
        1: "èmfasi en mecàniques",
        2: "èmfasi en durada i complexitat",
        3: "èmfasi en nota BGG",
        4: "mecàniques + nota BGG",
    }

    if _var_cycle > 0:
        st.caption(
            f"🔀 Variació {_var_cycle}/4 activa: **{_var_labels[_var_cycle]}** · "
            f"{len(excluded)} joc(s) exclòs(os) de suggeriments anteriors"
        )

    # ── Botons sempre visibles ────────────────────────────────────
    btn_col1, btn_col2 = st.columns([1, 1])

    with btn_col1:
        if st.button("🔀 Suggereix una recomanació diferent", use_container_width=True):
            if not recommendations_display.empty:
                top_to_exclude = recommendations_display.iloc[0]["nom_del_joc"]
                excl = st.session_state.get("excluded_games", [])
                if top_to_exclude not in excl:
                    excl.append(top_to_exclude)
                st.session_state["excluded_games"] = excl
            st.session_state["variation_idx"] = st.session_state.get("variation_idx", 0) + 1
            st.rerun()

    with btn_col2:
        if _var_cycle > 0:
            if st.button("↩️ Tornar a la recomanació principal", use_container_width=True):
                st.session_state["variation_idx"] = 0
                st.session_state["excluded_games"] = []
                st.rerun()

# ============================================================
# 6️⃣ CLUSTERING AMB UMAP
# ============================================================

from sklearn.cluster import KMeans
import plotly.express as px

# Import UMAP amb gestió d'error si no està instal·lat
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

st.divider()
st.header("🔬 Agrupació de jocs per similitud")

if not UMAP_AVAILABLE:
    st.warning(
        "⚠️ La llibreria `umap-learn` no està instal·lada. "
        "Afegeix `umap-learn` al teu `requirements.txt` per activar aquesta secció."
    )
else:
    n_clusters = st.slider("Nombre de grups (clusters):", 2, 10, 5, key="n_clusters")

    # Construir matriu per clustering — tots els jocs
    # rating_personal_imp s'exclou: molts jocs sense valorar distorsionen els grups
    cluster_num_cols = [c for c in all_num_cols if c != "rating_personal_imp"]
    cluster_features = df[cluster_num_cols].copy().fillna(df[cluster_num_cols].median())

    scaler_cluster  = StandardScaler()
    X_cl_numeric    = scaler_cluster.fit_transform(cluster_features)

    X_cl_numeric[:, cluster_num_cols.index("playingtime")] *= PLAYTIME_WEIGHT
    X_cl_numeric[:, cluster_num_cols.index("log_numplays")] *= NUMPLAYS_WEIGHT

    # Mecàniques amb pes reduït per no dominar el clustering
    X_cluster = np.hstack([X_cl_numeric, mec_cols.values * 0.5])

    # KMeans
    kmeans        = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)

    df_plot             = df[["nom_del_joc", "pes", "nota_bgg", "playingtime",
                               "Mecànica_principal", "minplayers", "maxplayers", "numplays"]].copy()
    df_plot["Cluster"]  = cluster_labels.astype(str)
    if use_personal_rating:
        df_plot["rating_personal"] = df["rating_personal"]

    # ── MILLORA 3: UMAP en lloc de PCA ──────────────────────────────────────
    # UMAP preserva millor l'estructura local en espais d'alta dimensió
    # (centenars de columnes de mecàniques) que PCA
    with st.spinner("Calculant UMAP (pot trigar uns segons la primera vegada)..."):
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,    # equilibri entre estructura local i global
            min_dist=0.1,      # com de junts poden estar els punts
            metric="cosine",   # consistent amb el recomanador
            random_state=42
        )
        coords = reducer.fit_transform(X_cluster)

    df_plot["UMAP_1"] = coords[:, 0]
    df_plot["UMAP_2"] = coords[:, 1]

    st.caption(
        "📐 **UMAP** (Uniform Manifold Approximation and Projection) preserva millor "
        "l'estructura dels datos que PCA quan hi ha moltes dimensions (mecàniques). "
        "Jocs propers al gràfic són més similars entre ells."
    )

    hover_data = {
        "pes": True, "nota_bgg": True, "playingtime": True,
        "Mecànica_principal": True, "numplays": True,
        "UMAP_1": False, "UMAP_2": False
    }
    if use_personal_rating:
        hover_data["rating_personal"] = True

    fig_cluster = px.scatter(
        df_plot,
        x="UMAP_1", y="UMAP_2",
        color="Cluster",
        hover_name="nom_del_joc",
        hover_data=hover_data,
        title=f"Agrupació de jocs en {n_clusters} clusters (UMAP 2D)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig_cluster.update_traces(marker=dict(size=8, opacity=0.8))
    fig_cluster.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000"),
        legend_title_text="Grup",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", title="UMAP 1"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", title="UMAP 2"),
    )

    st.plotly_chart(fig_cluster, use_container_width=True)

    # Taula resum per cluster
    st.subheader("📊 Característiques mitjanes per grup")
    agg_dict = {
        "nom_del_joc": "count",
        "pes": "mean",
        "nota_bgg": "mean",
        "playingtime": "mean",
        "numplays": "mean",
    }
    if use_personal_rating:
        agg_dict["rating_personal"] = "mean"

    cluster_summary = df_plot.groupby("Cluster").agg(agg_dict).round(2).reset_index()

    rename_map = {
        "Cluster": "Grup", "nom_del_joc": "Nº Jocs",
        "pes": "Pes mitjà", "nota_bgg": "Nota BGG",
        "playingtime": "Durada (min)", "numplays": "Partides jugades",
    }
    if use_personal_rating:
        rename_map["rating_personal"] = "Rating personal mitjà"

    cluster_summary = cluster_summary.rename(columns=rename_map)
    st.dataframe(cluster_summary, use_container_width=True)

    # Mecànica més freqüent per cluster
    st.subheader("🎲 Mecànica més freqüent per grup")
    df_mec_cluster = df_plot[["Cluster", "Mecànica_principal"]].copy()
    top_mec = (
        df_mec_cluster.groupby(["Cluster", "Mecànica_principal"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .groupby("Cluster")
        .first()
        .reset_index()[["Cluster", "Mecànica_principal", "count"]]
    )
    top_mec.columns = ["Grup", "Mecànica més freqüent", "Nº jocs amb aquesta mecànica"]
    st.dataframe(top_mec, use_container_width=True)

# ============================================================
# 5️⃣ MISUT MEEPLE - RESENYA DEL JOC MÉS RECOMANAT
# ============================================================

@st.cache_data(show_spinner=False)
def get_misutmeeple_summary(game_name: str) -> dict:
    search_url = f"https://misutmeeple.com/?s={game_name.replace(' ', '+')}"
    headers    = {"User-Agent": "Mozilla/5.0"}
    try:
        r    = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        link = soup.find("a", href=re.compile(r"misutmeeple\.com/\d{4}/"))
        if not link:
            return {"found": False, "url": "", "summary": "", "image": "", "sello": ""}

        page_url = link["href"]
        r2       = requests.get(page_url, headers=headers, timeout=10)
        soup2    = BeautifulSoup(r2.text, "html.parser")

        paragraphs = soup2.select("article p")[:3]
        summary    = "\n\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        if not summary.strip():
            return {"found": False, "url": "", "summary": "", "image": "", "sello": ""}

        image_url = ""
        img_tag   = soup2.select_one("article img")
        if img_tag:
            image_url = img_tag.get("src", "") or img_tag.get("data-src", "")

        sello_url = ""
        for img in soup2.select("article img"):
            src = img.get("src", "") or img.get("data-src", "")
            if "sello" in src.lower():
                sello_url = src
                break

        return {"found": True, "url": page_url, "summary": summary,
                "image": image_url, "sello": sello_url}

    except Exception as e:
        return {"found": False, "url": "", "summary": f"Error: {e}", "image": "", "sello": ""}


st.divider()
st.header("📖 Resenya a Misut Meeple")

# Usem recommendations_display per reflectir la variació activa
if not recommendations_display.empty:
    top_game = recommendations_display.iloc[0]["nom_del_joc"]
    st.markdown(f"Cercant informació sobre **{top_game}** a Misut Meeple...")

    with st.spinner(f"Buscant resenya de '{top_game}'..."):
        result = get_misutmeeple_summary(top_game)

    if result["found"]:
        st.success(f"✅ Resenya trobada per a **{top_game}**!")
        st.markdown(f"🔗 [Llegir la resenya completa a Misut Meeple]({result['url']})")
        img_col, txt_col = st.columns([1, 2])
        with img_col:
            if result["image"]:
                st.image(result["image"], use_container_width=True)
            if result["sello"]:
                st.image(result["sello"], use_container_width=True)
        with txt_col:
            st.markdown(result["summary"])
    else:
        st.info(f"ℹ️ No s'ha trobat cap resenya de **{top_game}** a Misut Meeple.")
else:
    st.info("No hi ha recomanacions per mostrar.")
