"""Funcions compartides entre les pàgines de l'app."""
import io

import pandas as pd
import streamlit as st

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


def fix_plotly_colors(fig):
    fig.update_layout(
        font=dict(color="#000000"),
        title_font=dict(color="#000000"),
        legend_font=dict(color="#000000"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(color="#000000", title_font=dict(color="#000000"))
    fig.update_yaxes(color="#000000", title_font=dict(color="#000000"))
    return fig
