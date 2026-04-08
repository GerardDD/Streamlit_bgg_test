import os


import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📚 Ludoteca de dumito")

# --- Carga de CSV ---
st.header("Llista completa")
archivo = True  # para pruebas locales

if archivo:
    #df = pd.read_csv("C:/Users/47173276T/Downloads/collection.csv")
    df = pd.read_csv("collection.csv")
    
    # Eliminem columnes buides
    df = df.dropna(axis=1, how="all")

    # Idiomes sense informar els passem tots a Desconegut
    df["version_languages"] = df["version_languages"].fillna("Desconegut")
    df["version_languages"] = df["version_languages"].replace("", "Desconegut")

    st.subheader("Contenido del archivo")
    st.dataframe(df)

    # --- Convertir año a numérico ---
    df["yearpublished"] = pd.to_numeric(df["yearpublished"], errors="coerce")

    # --- Crear columna de épocas ---
    df["epocas"] = pd.cut(
        df["yearpublished"],
        bins=[-3000, 1900, 2020, 2050],
        labels=["Antigüedad", "Moderno", "PostPandemia"]
    )

    # ============================
    # 🔵 FILTROS INTERACTIVOS
    # ============================

    st.sidebar.header("Filtres")

    # Filtro por época
    epocas_sel = st.sidebar.multiselect(
        "Selecciona épocas:",
        options=df["epocas"].dropna().unique(),
        default=df["epocas"].dropna().unique()
    )

    # Filtro por rango de años
    min_year = int(df["yearpublished"].min(skipna=True))
    max_year = int(df["yearpublished"].max(skipna=True))

    year_range = st.sidebar.slider(
        "Rango de años:",
        min_year, max_year, (min_year, max_year)
    )

    # Filtro por idioma
    idiomas = df["version_languages"].dropna().unique()
    idioma_sel = st.sidebar.multiselect(
        "Idioma:",
        options=idiomas,
        # default=idiomas[:5] if len(idiomas) > 5 else idiomas
        default=idiomas
    )

    # Aplicar filtros
    df_filtrado = df.copy()

    if epocas_sel:
        df_filtrado = df_filtrado[df_filtrado["epocas"].isin(epocas_sel)]

    df_filtrado = df_filtrado[
        df_filtrado["yearpublished"].between(year_range[0], year_range[1], inclusive="both")
    ]

    if idioma_sel:
        df_filtrado = df_filtrado[df_filtrado["version_languages"].isin(idioma_sel)]

    st.subheader("Datos filtrados")
    st.dataframe(df_filtrado)

    # ============================
    # 🔵 GRÁFICO 1: Juegos por época
    # ============================

    conteo_epocas = df_filtrado["epocas"].value_counts().reset_index()
    conteo_epocas.columns = ["epoca", "cantidad"]

    st.subheader("📊 Distribució de jocs per època")

    fig1 = px.bar(
        conteo_epocas,
        x="epoca",
        y="cantidad",
        text="quantitat",
        title="Jocs per època"
    )
    fig1.update_traces(textposition="outside")
    st.plotly_chart(fig1)

    # ============================
    # 🔵 GRÁFICO 2: Histograma de años
    # ============================

    st.subheader("📈 Histograma per any de publicació")

    fig2 = px.histogram(
        df_filtrado,
        x="yearpublished",
        nbins=30,
        title="Distribució d'anys de publicació"
    )

    fig2.update_layout(
        xaxis_title="Any de publicació",
        yaxis_title="Recompte"
    )
    
    st.plotly_chart(fig2)

    # ============================
    # 🔵 GRÁFICO 3: Juegos por idioma
    # ============================

    st.subheader("🌍 Juegos por idioma")

    # Checkbox para incluir/excluir "Desconegut" SOLO en el gráfico
    mostrar_desconegut = st.checkbox("Mostrar 'Desconegut' en el gráfico", value=False)

    # Filtrado local para el gráfico
    df_idiomas_graf = df_filtrado.copy()
    if not mostrar_desconegut:
        df_idiomas_graf = df_idiomas_graf[df_idiomas_graf["version_languages"] != "Desconegut"]

    conteo_idiomas = df_idiomas_graf["version_languages"].value_counts().reset_index()
    conteo_idiomas.columns = ["idioma", "cantidad"]

    fig3 = px.pie(
        conteo_idiomas,
        names="idioma",
        values="cantidad",
        title="Distribución por idioma"
    )
    st.plotly_chart(fig3)


else:
    st.info("Sube un CSV para generar el dashboard.")
