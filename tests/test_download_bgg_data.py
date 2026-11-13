"""Proves de la descàrrega de col·lecció i partides (sense xarxa: sessió simulada)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import download_bgg_data as dl  # noqa: E402

COLLECTION_XML = """<?xml version="1.0" encoding="utf-8"?>
<items totalitems="2">
  <item objecttype="thing" objectid="8203" subtype="boardgame" collid="93247940">
    <name sortindex="1">¡Pingüinos!</name>
    <originalname>Hey, That's My Fish!</originalname>
    <yearpublished>2015</yearpublished>
    <stats minplayers="2" maxplayers="4" minplaytime="20" maxplaytime="20" playingtime="20" numowned="30507">
      <rating value="7.5">
        <average value="6.69825"/><bayesaverage value="6.57108"/>
        <ranks>
          <rank type="subtype" id="1" name="boardgame" value="1171"/>
          <rank type="family" id="5499" name="familygames" value="300"/>
        </ranks>
      </rating>
    </stats>
    <status own="1" prevowned="0" fortrade="0" want="0" wanttoplay="0" wanttobuy="0"
            wishlist="0" wishlistpriority="3" preordered="0"/>
    <numplays>13</numplays>
    <comment>Losetas</comment>
    <version><item type="boardgameversion" id="1">
      <link type="boardgamepublisher" id="5" value="Edge Entertainment"/>
      <link type="language" id="2" value="Spanish"/>
      <name type="primary" value="Edge Spanish edition 2015"/>
      <yearpublished value="2015"/>
    </item></version>
  </item>
  <item objecttype="thing" objectid="146228" subtype="boardgame" collid="84537882">
    <name sortindex="1">1714: The Case of the Catalans</name>
    <stats minplayers="2" maxplayers="5" playingtime="120" numowned="538">
      <rating value="N/A"><average value="6.84859"/><bayesaverage value="5.67719"/>
        <ranks><rank type="subtype" value="Not Ranked"/></ranks>
      </rating>
    </stats>
    <status own="1"/>
    <numplays>0</numplays>
  </item>
</items>""".encode()

EMPTY_COLLECTION = b'<items totalitems="0"></items>'

THING_XML = b"""<items>
  <item type="boardgame" id="8203"><statistics><ratings><averageweight value="1.4447"/></ratings></statistics></item>
  <item type="boardgame" id="146228"><statistics><ratings><averageweight value="2.78260"/></ratings></statistics></item>
</items>"""


def plays_xml(total, plays):
    return f'<plays username="Mrbrussels" total="{total}" page="1">{"".join(plays)}</plays>'.encode()


PLAY_1 = """<play id="117595739" date="2026-08-16" quantity="1" length="0" incomplete="0" location="casa pama">
  <item name="Time's Up Express" objecttype="thing" objectid="462134">
    <subtypes><subtype value="boardgame"/></subtypes></item>
  <comments>#BGCatalog</comments>
  <players>
    <player name="gerard" score="" win="1"/>
    <player name="mama" score="" win="0"/>
  </players>
</play>"""
PLAY_2 = """<play id="108696829" date="2026-01-09" quantity="3" length="60" incomplete="1" location="">
  <item name="Ticket to Ride Map Collection 8" objecttype="thing" objectid="414268">
    <subtypes><subtype value="boardgame"/><subtype value="boardgameexpansion"/></subtypes></item>
  <players>
    <player name="Cristina" score="113" win="0"/>
    <player username="paula" name="" score="166" win="1"/>
  </players>
</play>"""
PLAY_3 = """<play id="1" date="2019-01-01" quantity="1" length="0" incomplete="0" location="x">
  <item name="Solo" objectid="5"><subtypes><subtype value="boardgame"/></subtypes></item>
</play>"""


class FakeResponse:
    def __init__(self, status, content=b""):
        self.status_code = status
        self.content = content


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url.rsplit("/", 1)[-1], params))
        return self.responses.pop(0)


def no_sleep(_):
    pass


def test_parse_collection():
    rows = dl.parse_collection(COLLECTION_XML, "standalone")
    a, b = rows
    assert a["objectname"] == "¡Pingüinos!"
    assert a["objectid"] == "8203"
    assert a["rating"] == "7.5"
    assert a["rank"] == "1171"
    assert a["baverage"] == "6.57108"
    assert a["own"] == "1"
    assert a["comment"] == "Losetas"
    assert a["version_languages"] == "Spanish"
    assert a["version_publishers"] == "Edge Entertainment"
    assert a["version_nickname"] == "Edge Spanish edition 2015"
    assert a["itemtype"] == "standalone"
    assert a["playingtime"] == "20"
    # Valors absents com a l'exportació manual
    assert b["rating"] == "0"
    assert b["rank"] == "0"
    assert b["originalname"] == "1714: The Case of the Catalans"
    assert set(a) == set(dl.COLLECTION_COLUMNS)


def test_parse_plays():
    total, rows = dl.parse_plays(plays_xml(2, [PLAY_1, PLAY_2]))
    assert total == 2
    a, b = rows
    assert a == {
        "Play ID": "117595739", "Game ID": "462134", "Name": "Time's Up Express",
        "Expansion": "-", "Date": "2026-08-16", "Plays": "x1", "Location": "casa pama",
        "Duration": "-", "Incomplete": "-", "Report": "#BGCatalog", "Count": "2",
        "Players": "gerard, mama", "Winner": "gerard",
    }
    assert b["Expansion"] == "Y"
    assert b["Plays"] == "x3"
    assert b["Duration"] == "60"
    assert b["Incomplete"] == "Y"
    assert b["Location"] == "-"
    assert b["Report"] == "-"
    assert b["Players"] == "Cristina (113), paula (166)"
    assert b["Winner"] == "paula"


def test_download_collection_merges_expansions_and_weights():
    session = FakeSession([
        FakeResponse(202),                      # BGG encara prepara la col·lecció
        FakeResponse(200, COLLECTION_XML),
        FakeResponse(200, EMPTY_COLLECTION),
        FakeResponse(200, THING_XML),
    ])
    rows = dl.download_collection(session, "Mrbrussels", sleep=no_sleep)
    assert [r["objectname"] for r in rows] == ["1714: The Case of the Catalans", "¡Pingüinos!"]
    assert {r["objectid"]: r["avgweight"] for r in rows} == {"8203": "1.4447", "146228": "2.7826"}
    assert session.calls[1][1]["excludesubtype"] == "boardgameexpansion"
    assert session.calls[2][1]["subtype"] == "boardgameexpansion"
    assert session.calls[3] == ("thing", {"id": "8203,146228", "stats": 1})


def test_download_plays_paginates_and_sorts():
    session = FakeSession([
        FakeResponse(200, plays_xml(3, [PLAY_2, PLAY_3])),
        FakeResponse(200, plays_xml(3, [PLAY_1])),
    ])
    rows = dl.download_plays(session, "Mrbrussels", sleep=no_sleep)
    assert [r["Play ID"] for r in rows] == ["117595739", "108696829", "1"]
    assert [p["page"] for _, p in session.calls] == [1, 2]


def test_auth_error():
    with pytest.raises(dl.BGGAuthError):
        dl.get_xml(FakeSession([FakeResponse(401)]), "plays", {}, sleep=no_sleep)


def test_written_csvs_are_readable_by_the_app(tmp_path):
    from utils import load_csv

    _, plays = dl.parse_plays(plays_xml(2, [PLAY_1, PLAY_2]))
    path = tmp_path / "plays.csv"
    assert dl.write_csv(plays, dl.PLAYS_COLUMNS, path, quote_all=False)
    assert not dl.write_csv(plays, dl.PLAYS_COLUMNS, path, quote_all=False)  # idèntic → sense canvis
    df = load_csv.__wrapped__(str(path))
    assert list(df.columns) == dl.PLAYS_COLUMNS
    assert df.loc[1, "Players"] == "Cristina (113), paula (166)"

    coll = dl.parse_collection(COLLECTION_XML, "standalone")
    cpath = tmp_path / "collection.csv"
    dl.write_csv(coll, dl.COLLECTION_COLUMNS, cpath, quote_all=True)
    original_header = (Path(__file__).resolve().parent.parent / "pages" / "collection.csv").read_text(
        encoding="utf-8").splitlines()[0]
    assert cpath.read_text(encoding="utf-8").splitlines()[0] == original_header
    assert pd.read_csv(cpath)["objectid"].tolist() == [8203, 146228]


def test_main_does_not_overwrite_with_empty_plays(tmp_path, monkeypatch):
    path = tmp_path / "plays.csv"
    path.write_text("vell", encoding="utf-8")
    monkeypatch.setattr(dl, "PLAYS_CSV", path)
    monkeypatch.setattr(dl, "make_session", lambda: FakeSession([FakeResponse(200, plays_xml(0, []))]))
    monkeypatch.setattr(dl.time, "sleep", no_sleep)
    assert dl.main(["--only", "plays"]) == 1
    assert path.read_text(encoding="utf-8") == "vell"
