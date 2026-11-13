"""Proves del script d'actualització de la caché (sense xarxa: sessió simulada)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import update_mechanics_cache as umc  # noqa: E402

XML = b"""<?xml version="1.0" encoding="utf-8"?>
<items termsofuse="https://boardgamegeek.com/xmlapi/termsofuse">
  <item type="boardgame" id="1">
    <name type="primary" value="Joc U"/>
    <link type="boardgamecategory" id="9" value="Card Game"/>
    <link type="boardgamemechanic" id="2040" value="Hand Management"/>
    <link type="boardgamemechanic" id="2004" value="Set Collection"/>
  </item>
  <item type="boardgameexpansion" id="2">
    <link type="boardgamemechanic" id="2001" value="Action Points"/>
  </item>
</items>"""


class FakeResponse:
    def __init__(self, status, content=b""):
        self.status_code = status
        self.content = content


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return self.responses.pop(0)


def no_sleep(_):
    pass


def test_parse_things():
    assert umc.parse_things(XML) == {
        1: ["Hand Management", "Set Collection"],
        2: ["Action Points"],
    }


def test_ids_to_fetch_includes_missing_and_empty():
    cache = {1: ["X"], 2: []}
    assert umc.ids_to_fetch([1, 2, 3], cache) == [2, 3]
    assert umc.ids_to_fetch([1, 2, 3], cache, refresh_all=True) == [1, 2, 3]


def test_chunks():
    assert list(umc.chunks(list(range(45)), 20)) == [list(range(20)), list(range(20, 40)), list(range(40, 45))]


def test_update_cache_retries_queued_response_and_keeps_good_data():
    cache = {1: [], 5: ["Old"]}
    session = FakeSession([FakeResponse(202), FakeResponse(200, XML)])
    demanats, nous = umc.update_cache([1, 2, 3, 5], cache, session, sleep=no_sleep)
    assert demanats == 3                       # 1 (buit), 2 i 3 (nous); 5 ja tenia dades
    assert nous == 2
    assert cache[1] == ["Hand Management", "Set Collection"]
    assert cache[2] == ["Action Points"]
    assert cache[3] == []                      # BGG no l'ha retornat
    assert cache[5] == ["Old"]
    assert len(session.calls) == 2 and session.calls[0]["id"] == "1,2,3"


def test_auth_error_is_explicit():
    with pytest.raises(umc.BGGAuthError, match="BGG_API_TOKEN"):
        umc.fetch_batch(FakeSession([FakeResponse(401)]), [1], sleep=no_sleep)


def test_cache_roundtrip_is_readable_by_app(tmp_path):
    path = tmp_path / "cache.csv"
    cache = {1: ["Hand Management", "Player's Choice"], 2: []}
    umc.save_cache(cache, path)
    assert umc.load_cache(path) == cache


def test_real_cache_file_is_valid():
    cache = umc.load_cache(umc.CACHE_CSV)
    assert len(cache) > 0
    assert all(isinstance(v, list) for v in cache.values())
