"""Funcions compartides entre les pàgines de l'app."""
import io

import pandas as pd
import streamlit as st

# Color principal (el mateix que primaryColor a .streamlit/config.toml)
ACCENT = "#E8703A"

COLLECTION_CSV = "pages/collection.csv"
PLAYS_CSV = "pages/playsMrbrussels.csv"

# Correccions de la mecànica principal (columna "comment" de BGG).
# L'ordre importa: "Pseudo-wargames" passa primer per "Pseudo-Wargames".
MECHANIC_FIXES = [
    ("Selecció accions", "Selecció d'accions"),
    ("Car-driven", "Card-driven"),
    ("Col·locació daus", "Col·locació de daus"),
    ("Construcció motor", "Construcció de motor"),
    ("Draft pùblic", "Draft públic"),
    ("Gestió de la mà", "Gestió de mà"),
    ("Pseudo-abstracte", "Pseudo-Abstracte"),
    ("Pseudo-wargame", "Pseudo-Wargame"),
    ("Pseudo-Wargames", "Pseudo-Wargame"),
]


def clean_mechanics(series: pd.Series) -> pd.Series:
    """Normalitza la mecànica principal: buits → 'No informat' i correccions ortogràfiques."""
    s = series.fillna("").astype(str)
    s = s.where(s.str.strip() != "", "No informat")
    for old, new in MECHANIC_FIXES:
        s = s.str.replace(old, new, regex=False)
    return s


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.replace("﻿", "", regex=False)
        .str.replace("\r", "", regex=False)
        .str.replace("\n", "", regex=False)
        .str.strip()
    )
    return df


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """Llegeix un CSV del repositori (amb memòria cau)."""
    return _clean_columns(pd.read_csv(path, sep=",", engine="python"))


@st.cache_data(show_spinner=False)
def load_csv_bytes(data: bytes) -> pd.DataFrame:
    """Llegeix un CSV pujat per l'usuari (amb memòria cau pel contingut)."""
    return _clean_columns(pd.read_csv(io.BytesIO(data), sep=",", engine="python"))


def split_players(value) -> list:
    """'gerard, guillaum' → ['gerard', 'guillaum'] (sense buits ni 'nan')."""
    if pd.isna(value):
        return []
    return [p.strip() for p in str(value).split(",") if p.strip() and p.strip() != "nan"]


# ============================================================
# CSV pujat per l'usuari, compartit entre pàgines
# ============================================================
# Els widgets (file_uploader) perden el valor en canviar de pàgina, per això
# el contingut es guarda en claus pròpies de st.session_state.

UPLOADED_COLLECTION_KEY = "uploaded_collection"
UPLOADED_PLAYS_KEY = "uploaded_plays"


def remember_upload(key: str, fitxer) -> None:
    """Guarda un fitxer pujat (UploadedFile) a la sessió."""
    st.session_state[key] = {"name": fitxer.name, "data": fitxer.getvalue()}


def forget_upload(key: str) -> None:
    st.session_state.pop(key, None)


def uploaded_name(key: str):
    up = st.session_state.get(key)
    return up["name"] if up else None


def load_shared_csv(key: str, default_path: str) -> pd.DataFrame:
    """CSV pujat a la sessió si n'hi ha; si no, el del repositori."""
    up = st.session_state.get(key)
    if up:
        return load_csv_bytes(up["data"])
    return load_csv(default_path)


def bgg_url(objectid) -> str:
    if pd.isna(objectid):
        return None
    return f"https://boardgamegeek.com/boardgame/{int(objectid)}"


def style_plotly(fig):
    """Fons transparent perquè els gràfics agafin el color del tema (clar o fosc)."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
