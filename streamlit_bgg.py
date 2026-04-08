import os


import streamlit as st
import pandas as pd
import plotly.express as px

# ============================
# 🔵 CARGA DE DATOS (CSV por defecto + CSV manual)
# ============================

st.sidebar.subheader("📂 Dades d'entrada")

archivo_subido = st.sidebar.file_uploader(
    "Puja un CSV personalitzat (opcional):",
    type=["csv"]
)

if archivo_subido is not None:
    # Leer CSV subido por el usuario
    df = pd.read_csv(archivo_subido)
    st.sidebar.success("CSV carregat correctament!")
else:
    # CSV por defecto
    df = pd.read_csv("collection.csv")
    st.sidebar.info("S'està utilitzant el CSV per defecte.")



# ============================
# 🔵 ESTILO PERSONALIZADO
# ============================
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        /* Fondo general */
        .stApp {
            background-color: #FFF4E6; /* naranja suave pastel */
        }

        /* Opcional: color de los paneles laterales */
        section[data-testid="stSidebar"] {
            background-color: #FFE8D1; /* un poco más oscuro */
        }

        /* Opcional: bordes redondeados en widgets */
        .stButton>button, .stTextInput>div>div>input, .stSelectbox>div>div>div {
            border-radius: 8px;
        }

        /* Force black text everywhere */
        html, body, [class*="css"] {
            color: #000000 !important;
        }

        /* Force black text inside widgets */
        .stTextInput input,
        .stSelectbox div,
        .stButton button,
        .stCheckbox label,
        .stRadio label,
        .stMetric label,
        .stMetric div {
            color: #000000 !important;
        }

        /* Force black text inside dataframes */
        .dataframe tbody tr td,
        .dataframe thead tr th {
            color: #000000 !important;
        }

        /* Force black text in Plotly charts (titles, labels, ticks) */
        .js-plotly-plot .plotly .main-svg {
            color: #000000 !important;
            fill: #000000 !important;
        }
        
    </style>
    """, unsafe_allow_html=True)



st.title("📚 Ludoteca de dumito")

# --- Carga de CSV ---
st.header("Llista completa")
archivo = True  # para pruebas locales

if archivo:
    #df = pd.read_csv("C:/Users/47173276T/Downloads/collection.csv")
    #df = pd.read_csv("collection.csv")
    
    # Eliminem columnes buides
    df = df.dropna(axis=1, how="all")

    # Idiomes sense informar els passem tots a Desconegut
    df["version_languages"] = df["version_languages"].fillna("Desconegut")
    df["version_languages"] = df["version_languages"].replace("", "Desconegut")

    # 🔥 Renombrar columnas
    df = df.rename(columns={
        "objectname": "nom_del_joc",
        "numplays": "num_partides",
        "avgweight": "pes",
        "average": "nota_bgg",
        "yearpublished":"any_publicació"
    })

    #st.subheader("Contenido del archivo")
    #st.dataframe(df)

    # --- Convertir año a numérico ---
    df["any_publicació"] = pd.to_numeric(df["any_publicació"], errors="coerce")

    # --- Crear columna de épocas ---
    df["epocas"] = pd.cut(
        df["any_publicació"],
        bins=[-3000, 1900, 2020, 2050],
        labels=["Antiguitat", "Moderna", "Post-Pandèmia"]
    )

    # ============================
    # 🔵 FILTROS INTERACTIVOS
    # ============================

    st.sidebar.header("Filtres")

    # Filtro por época
    epocas_sel = st.sidebar.multiselect(
        "Selecciona epoques:",
        options=df["epocas"].dropna().unique(),
        default=df["epocas"].dropna().unique()
    )

    # Filtro por rango de años
    min_year = int(df["any_publicació"].min(skipna=True))
    max_year = int(df["any_publicació"].max(skipna=True))

    year_range = st.sidebar.slider(
        "Rang d'anys:",
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
        df_filtrado["any_publicació"].between(year_range[0], year_range[1], inclusive="both")
    ]

    if idioma_sel:
        df_filtrado = df_filtrado[df_filtrado["version_languages"].isin(idioma_sel)]


    st.subheader("La col·lecció")

    # ============================
    # 🔵 KPIs PRINCIPALS
    # ============================

    st.subheader("📌 Indicadors principals del filtre actual")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🎯 Jocs filtrats",
            value=len(df_filtrado)
        )

    with col2:
        st.metric(
            label="⚖️ Pes mitjà",
            value=round(df_filtrado["pes"].mean(), 2) if not df_filtrado.empty else "—"
        )

    with col3:
        st.metric(
            label="⭐ Nota BGG mitjana",
            value=round(df_filtrado["nota_bgg"].mean(), 2) if not df_filtrado.empty else "—"
        )

    with col4:
        st.metric(
            label="🎲 Total de partides",
            value=int(df_filtrado["num_partides"].sum()) if not df_filtrado.empty else "—"
        )


    
    st.dataframe(df_filtrado)

    # ============================
    # 🔵 GRÁFICO 1: Juegos por época
    # ============================

    conteo_epocas = df_filtrado["epocas"].value_counts().reset_index()
    conteo_epocas.columns = ["epoca", "cantidad"]

    st.subheader("📊 Distribució de jocs per època")

    fig1 = px.pie(
        conteo_epocas,
        names="epoca",
        values="cantidad",
        title="Distribució de jocs per època",
        color="epoca",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig1.update_traces(
        textposition="inside",
        textinfo="percent+label",
        pull=[0.05] * len(conteo_epocas)  # pequeño efecto de separación
    )

    fig1.update_layout(
        showlegend=False,
        plot_bgcolor="#FFF4E6",
        paper_bgcolor="#FFF4E6"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # ============================
    # 🔵 GRÁFICO 2: Histograma de años
    # ============================

    st.subheader("📈 Histograma per any de publicació")

    fig2 = px.histogram(
        df_filtrado,
        x="any_publicació",
        nbins=30,
        title="Distribució d'anys de publicació"
    )

    fig2.update_layout(
        xaxis_title="Any de publicació",
        yaxis_title="Recompte",
        plot_bgcolor="#FFF4E6",
        paper_bgcolor="#FFF4E6"
    )
    
    st.plotly_chart(fig2)
    
    # ============================
    # 🔵 GRÁFICO 5: Top 10 jocs més jugats (versió millorada)
    # ============================

    st.subheader("🏆 Top 10 jocs més jugats")

    # Ordenar por número de partidas y seleccionar top 10
    top10 = df_filtrado.sort_values("num_partides", ascending=False).head(10)

    fig5 = px.bar(
        top10,
        x="num_partides",
        y="nom_del_joc",
        orientation="h",
        text="num_partides",
        title="Top 10 jocs més jugats",
        color="num_partides",
        color_continuous_scale="Blues"
    )

    fig5.update_traces(
        textposition="outside",
        marker=dict(line=dict(color="black", width=0.5))
    )

    fig5.update_layout(
        xaxis_title="Nombre de partides",
        yaxis_title="Joc",
        yaxis=dict(autorange="reversed"),  # Para que el más jugado quede arriba
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig5, use_container_width=True)



    # ============================
    # 🔵 GRÁFICO 3: Juegos por idioma
    # ============================

    st.subheader("🌍 Jocs per idioma")

    # Checkbox para incluir/excluir "Desconegut" SOLO en el gráfico
    mostrar_desconegut = st.checkbox("Mostrar 'Desconegut' al gràfic", value=False)

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

    fig3.update_layout(
        # showlegend=False,
        plot_bgcolor="#FFF4E6",
        paper_bgcolor="#FFF4E6"
    )
    
    st.plotly_chart(fig3)
    
    # ============================
    # 🔵 GRÁFICO 4: Scatter interactiu
    # ============================

    st.subheader("📌 Relació entre variables i nombre de partides")

    # Selector de variable para el eje X
    opcion_x = st.selectbox(
        "Selecciona la variable per comparar amb nombre de partides:",
        options={
            "pes": "Pes (complexitat)",
            "nota_bgg": "Nota BGG"
        }
    )

    # Crear scatter dinámico
    fig4 = px.scatter(
        df_filtrado,
        x=opcion_x,
        y="num_partides",
        hover_name="nom_del_joc",
        title=f"{opcion_x} vs Nombre de partides",
        labels={
            opcion_x: opcion_x,
            "num_partides": "Nombre de partides"
        },

    )

    fig4.update_traces(marker=dict(size=10, opacity=1))
    fig4.update_layout(
        xaxis_title=opcion_x,
        yaxis_title="Nombre de partides",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # ============================
    # 🔵 LISTA DE LA VERGÜENZA
    # ============================

    st.subheader("😅 Llista de la vergonya (jocs sense jugar)")

    # Checkbox para incluir/excluir expansiones
    mostrar_expansions = st.checkbox("Mostrar expansions a la llista", value=False)

    df_vergonya = df[df["num_partides"] == 0].copy()

    # Filtrar expansiones si el checkbox está desmarcado
    if not mostrar_expansions:
        df_vergonya = df_vergonya[df_vergonya["itemtype"] != "expansion"]

    df_vergonya = df_vergonya.sort_values("nom_del_joc")

    if df_vergonya.empty:
        st.success("🎉 No tens cap joc sense jugar! Bona feina!")
    else:
        st.warning(f"Tens {len(df_vergonya)} jocs sense estrenar...")
        st.dataframe(df_vergonya[["nom_del_joc", "any_publicació", "pes", "version_languages", "itemtype"]])

    # ============================
    # 🔵 MISSATGES DELS USUARIS
    # ============================

    st.subheader("💬 Deixa un missatge (NO OPERATIU ENCARA)")

    # Inicializar lista de mensajes en la sesión
    if "missatges" not in st.session_state:
        st.session_state.missatges = []

    # Caja de texto
    nou_missatge = st.text_area("Escriu el teu missatge aquí:")

    # Botón para enviar
    if st.button("Enviar missatge"):
        if nou_missatge.strip():
            st.session_state.missatges.append(nou_missatge.strip())
            st.success("Missatge afegit!")
        else:
            st.warning("El missatge no pot estar buit.")

    # Mostrar mensajes enviados
    if st.session_state.missatges:
        st.write("### 📜 Missatges enviats:")
        for idx, msg in enumerate(st.session_state.missatges, 1):
            st.write(f"**{idx}.** {msg}")




else:
    st.info("Sube un CSV para generar el dashboard.")
