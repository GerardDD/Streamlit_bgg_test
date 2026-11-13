"""Proves de fum: cada pàgina carrega sense errors i les interaccions principals funcionen."""
from streamlit.testing.v1 import AppTest

from utils import UPLOADED_COLLECTION_KEY, UPLOADED_PLAYS_KEY

MAIN = "../streamlit_bgg.py"
STATS = "../pages/Estadistiques_partides.py"
REC = "../pages/Recomanador_personalitzat.py"


def run(path, timeout=300, session=None):
    at = AppTest.from_file(path, default_timeout=timeout)
    for k, v in (session or {}).items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def upload(name, text):
    return {"name": name, "data": text.encode("utf-8")}


# ------------------------------------------------------------------ pàgina principal

def test_main_default_collection():
    at = run(MAIN)
    assert not at.error
    assert len(at.metric) == 5
    assert int(at.metric[0].value) > 0
    assert len(at.tabs) == 3


def test_main_scatter_shows_labels():
    at = run(MAIN)
    sb = next(s for s in at.selectbox if "partides" in s.label)
    assert sb.options == ["Pes (complexitat)", "Nota BGG"]
    sb.set_value("nota_bgg").run()
    assert not at.exception


def test_main_uploaded_csv_single_year_and_mechanics_cleanup():
    csv = (
        "objectname,objectid,numplays,avgweight,average,yearpublished,comment\n"
        "A,1,0,2,7,2015,Pseudo-wargames\n"
        "B,2,3,3,8,2015,\n"
    )
    at = run(MAIN, session={UPLOADED_COLLECTION_KEY: upload("meu.csv", csv)})
    assert not at.error
    assert at.metric[0].value == "2"
    assert any("2015" in c.value for c in at.sidebar.caption)


def test_main_uploaded_csv_without_years():
    csv = "objectname,numplays,avgweight,average,yearpublished\nA,0,2,7,\n"
    at = run(MAIN, session={UPLOADED_COLLECTION_KEY: upload("sense_anys.csv", csv)})
    assert not at.error
    assert at.metric[4].value == "—"


def test_main_uploaded_csv_missing_columns_shows_error():
    at = run(MAIN, session={UPLOADED_COLLECTION_KEY: upload("dolent.csv", "objectname,numplays\nA,1\n")})
    assert at.error and "falten" in at.error[0].value


# ------------------------------------------------------------------ estadístiques

def test_stats_default():
    at = run(STATS)
    assert int(at.metric[0].value) > 0
    # Secció de % de victòries
    assert any("guanyador informat" in c.value for c in at.caption)


def test_stats_presets_and_player_filter():
    at = run(STATS)
    total = int(at.metric[0].value)
    for preset in ["Última setmana", "Últim mes", "Últim any", "Personalitzat", "Tot"]:
        at.sidebar.selectbox[0].set_value(preset).run()
        assert not at.exception, preset
        assert int(at.metric[0].value) <= total

    at.sidebar.multiselect[0].set_value(["papa", "mama"]).run()
    algun = int(at.metric[0].value)
    at.sidebar.radio[0].set_value("Tots els seleccionats").run()
    tots = int(at.metric[0].value)
    assert 0 < tots <= algun < total


def test_stats_uploaded_plays_and_win_rate():
    csv = (
        "Name,Date,Players,Winner\n"
        "Joc A,2024-01-01,\"anna (10), bernat (5)\",anna\n"
        "Joc A,2024-01-02,\"anna, bernat\",bernat\n"
        "Joc B,2024-01-03,\"anna, bernat\",anna\n"
        "Coop,2024-01-04,\"anna, bernat\",\"anna, bernat\"\n"
    )
    at = AppTest.from_file(STATS, default_timeout=60)
    at.session_state[UPLOADED_PLAYS_KEY] = upload("partides.csv", csv)
    at.run()
    assert not at.exception
    assert at.metric[0].value == "4"
    at.slider(key="wr_min_partides").set_value(1)
    at.selectbox(key="wr_jugador").set_value("anna")
    at.slider(key="wr_min_partides_joc").set_value(1)
    at.run()
    assert not at.exception
    assert any("s'ignoren 1" in c.value for c in at.caption)  # la cooperativa

    # Per defecte s'exclou la cooperativa: anna guanya 1/1 a Joc B i 1/2 a Joc A
    millors = at.dataframe[0].value.set_index("Joc")
    assert millors.loc["Joc B", "% victòries"] == 100.0
    assert millors.loc["Joc A", "Victòries"] == 1
    assert millors.loc["Joc A", "Partides"] == 2
    assert "Coop" not in millors.index

    # Incloent cooperatius, la partida Coop compta com a victòria
    at.checkbox(key="wr_excloure_coop").uncheck().run()
    millors = at.dataframe[0].value.set_index("Joc")
    assert millors.loc["Coop", "% victòries"] == 100.0


# ------------------------------------------------------------------ recomanador

def test_recommender_default():
    at = run(REC)
    assert not at.error
    rec = next(d.value for d in at.dataframe if "afinitat" in d.value.columns)
    assert rec["afinitat"].between(0, 1).all()
    assert len(at.get("download_button")) == 1

    # Canviar una valoració no ha de fallar i ha de ser ràpid (càlculs en memòria cau)
    seg = at.get("segmented_control")[0]
    seg.set_value(9).run()
    assert not at.exception


def test_recommender_uses_uploaded_collection(root):
    import pandas as pd

    full = pd.read_csv(root / "pages" / "collection.csv")
    subset = full.head(60).to_csv(index=False)
    at = run(REC, session={UPLOADED_COLLECTION_KEY: upload("subset.csv", subset)})
    assert any("subset.csv" in i.value for i in at.info)
    rec = next(d.value for d in at.dataframe if "afinitat" in d.value.columns)
    assert set(rec["nom_del_joc"]) <= set(full.head(60)["objectname"])
