"""Descarrega la col·lecció i les partides d'un usuari de BGG en CSV.

Substitueix l'exportació manual (Collection → Export i l'exportació de Plays):
fa servir l'API XML2 oficial de BGG i escriu els CSV amb les mateixes columnes
que ja llegeix l'app.

- Col·lecció: /xmlapi2/collection (jocs en propietat, amb estadístiques i versió)
  + /xmlapi2/thing per al pes (avgweight), que la col·lecció no inclou.
- Partides: /xmlapi2/plays (100 per pàgina).

BGG demana registrar les aplicacions que fan servir l'API: si tens un token,
posa'l a la variable d'entorn BGG_API_TOKEN (a GitHub, com a secret del repositori).

Ús:
    python scripts/download_bgg_data.py                    # col·lecció i partides
    python scripts/download_bgg_data.py --only plays       # només partides
    python scripts/download_bgg_data.py --only collection  # només col·lecció
    python scripts/download_bgg_data.py --username Altre
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from update_mechanics_cache import BGGAuthError, chunks, make_session  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_USERNAME = "Mrbrussels"
COLLECTION_CSV = Path(os.environ.get("BGG_COLLECTION_CSV", ROOT / "pages" / "collection.csv"))
PLAYS_CSV = Path(os.environ.get("BGG_PLAYS_CSV", ROOT / "pages" / "playsMrbrussels.csv"))

API = "https://boardgamegeek.com/xmlapi2"
THING_BATCH = 20
PAUSE_SECONDS = 5
MAX_RETRIES = 8

# Columnes de l'exportació manual de BGG (mateix ordre)
COLLECTION_COLUMNS = [
    "objectname", "objectid", "rating", "numplays", "weight", "own", "fortrade", "want",
    "wanttobuy", "wanttoplay", "prevowned", "preordered", "wishlist", "wishlistpriority",
    "wishlistcomment", "comment", "conditiontext", "haspartslist", "wantpartslist", "collid",
    "baverage", "average", "avgweight", "rank", "numowned", "objecttype", "originalname",
    "minplayers", "maxplayers", "playingtime", "maxplaytime", "minplaytime", "yearpublished",
    "bggrecplayers", "bggbestplayers", "bggrecagerange", "bgglanguagedependence", "publisherid",
    "imageid", "year", "language", "other", "itemtype", "barcode", "pricepaid", "pp_currency",
    "currvalue", "cv_currency", "acquisitiondate", "acquiredfrom", "quantity", "privatecomment",
    "invlocation", "invdate", "version_publishers", "version_languages", "version_yearpublished",
    "version_nickname",
]
PLAYS_COLUMNS = [
    "Play ID", "Game ID", "Name", "Expansion", "Date", "Plays", "Location", "Duration",
    "Incomplete", "Report", "Count", "Players", "Winner",
]
STATUS_FIELDS = [
    "own", "fortrade", "want", "wanttobuy", "wanttoplay", "prevowned", "preordered",
    "wishlist", "wishlistpriority",
]


# ------------------------------------------------------------------ funcions pures

def _num(value, default="0") -> str:
    """'N/A', buit o None → default; '7.50000' → '7.5'; '8.0' → '8'."""
    if value in (None, "", "N/A"):
        return default
    try:
        f = float(value)
    except ValueError:
        return default
    return str(int(f)) if f.is_integer() else str(f)


def _text(node, path) -> str:
    el = node.find(path)
    return (el.text or "").strip() if el is not None else ""


def _attr(node, path, attr="value") -> str:
    el = node.find(path)
    return el.get(attr, "") if el is not None else ""


def parse_collection(xml_content: bytes, itemtype: str) -> list[dict]:
    """Files (dict amb COLLECTION_COLUMNS) d'una resposta /collection."""
    root = ET.fromstring(xml_content)
    rows = []
    for item in root.findall("item"):
        row = dict.fromkeys(COLLECTION_COLUMNS, "")
        stats = item.find("stats")
        rating = stats.find("rating") if stats is not None else None
        status = item.find("status")
        row.update(
            objectname=_text(item, "name"),
            objectid=item.get("objectid", ""),
            collid=item.get("collid", ""),
            objecttype=item.get("objecttype", "thing"),
            originalname=_text(item, "originalname") or _text(item, "name"),
            yearpublished=_text(item, "yearpublished"),
            numplays=_text(item, "numplays") or "0",
            comment=_text(item, "comment"),
            wishlistcomment=_text(item, "wishlistcomment"),
            conditiontext=_text(item, "conditiontext"),
            itemtype=itemtype,
            weight="0",
            avgweight="0",
        )
        if status is not None:
            for f in STATUS_FIELDS:
                row[f] = status.get(f, "0")
        if stats is not None:
            for f in ("minplayers", "maxplayers", "playingtime", "minplaytime", "maxplaytime", "numowned"):
                row[f] = stats.get(f, "")
        if rating is not None:
            row["rating"] = _num(rating.get("value"))
            row["average"] = _num(_attr(rating, "average"))
            row["baverage"] = _num(_attr(rating, "bayesaverage"))
            rank = next((r for r in rating.findall("ranks/rank") if r.get("type") == "subtype"), None)
            row["rank"] = _num(rank.get("value") if rank is not None else None)
        version = item.find("version/item")
        if version is not None:
            links = version.findall("link")
            row["version_publishers"] = ", ".join(
                l.get("value", "") for l in links if l.get("type") == "boardgamepublisher")
            row["version_languages"] = ", ".join(
                l.get("value", "") for l in links if l.get("type") == "language")
            row["version_yearpublished"] = _attr(version, "yearpublished")
            row["version_nickname"] = _attr(version, "name")
        rows.append(row)
    return rows


def parse_weights(xml_content: bytes) -> dict[str, str]:
    """{objectid: averageweight} d'una resposta /thing?stats=1."""
    root = ET.fromstring(xml_content)
    return {
        item.get("id"): _num(_attr(item, "statistics/ratings/averageweight"))
        for item in root.findall("item")
    }


def parse_plays(xml_content: bytes) -> tuple[int, list[dict]]:
    """(total de partides, files amb PLAYS_COLUMNS) d'una pàgina /plays."""
    root = ET.fromstring(xml_content)
    total = int(root.get("total", "0") or 0)
    rows = []
    for play in root.findall("play"):
        item = play.find("item")
        subtypes = [s.get("value") for s in play.findall("item/subtypes/subtype")]
        players, winners = [], []
        for p in play.findall("players/player"):
            name = (p.get("name") or p.get("username") or "").strip() or "Anonymous player"
            score = (p.get("score") or "").strip()
            players.append(f"{name} ({score})" if score else name)
            if p.get("win") == "1":
                winners.append(name)
        length = play.get("length", "0")
        rows.append({
            "Play ID": play.get("id", ""),
            "Game ID": item.get("objectid", "") if item is not None else "",
            "Name": item.get("name", "") if item is not None else "",
            "Expansion": "Y" if "boardgameexpansion" in subtypes else "-",
            "Date": play.get("date", ""),
            "Plays": f"x{play.get('quantity', '1')}",
            "Location": play.get("location", "").strip() or "-",
            "Duration": length if length not in ("", "0") else "-",
            "Incomplete": "Y" if play.get("incomplete") == "1" else "-",
            "Report": _text(play, "comments").replace("\r", " ").replace("\n", " ") or "-",
            "Count": str(len(players)),
            "Players": ", ".join(players) or "-",
            "Winner": ", ".join(winners) or "-",
        })
    return total, rows


def write_csv(rows: list[dict], columns: list[str], path: Path, quote_all: bool) -> bool:
    """Escriu el CSV i retorna True si el contingut ha canviat."""
    df = pd.DataFrame(rows, columns=columns)
    if quote_all:  # com l'exportació de BGG: capçalera sense cometes i valors entre cometes
        new = ",".join(columns) + "\n" + df.to_csv(index=False, header=False, quoting=csv.QUOTE_ALL)
    else:
        new = df.to_csv(index=False)
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if new == old:
        return False
    path.write_text(new, encoding="utf-8")
    return True


# ------------------------------------------------------------------ xarxa

def get_xml(session: requests.Session, endpoint: str, params: dict, sleep=None) -> bytes:
    """GET amb reintents (202 = BGG prepara la resposta, 429/5xx = temporal)."""
    sleep = sleep or time.sleep
    wait = PAUSE_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        r = session.get(f"{API}/{endpoint}", params=params, timeout=60)
        if r.status_code == 200:
            sleep(PAUSE_SECONDS)  # BGG recomana espaiar les peticions
            return r.content
        if r.status_code in (401, 403):
            raise BGGAuthError(
                f"BGG ha rebutjat la petició ({r.status_code}). Cal un token d'aplicació de BGG "
                "a la variable d'entorn BGG_API_TOKEN (secret del repositori a GitHub)."
            )
        print(f"    {endpoint}: resposta {r.status_code}, reintent {attempt}/{MAX_RETRIES} d'aquí {wait} s")
        sleep(wait)
        wait = min(wait * 2, 120)
    raise RuntimeError(f"No s'ha pogut descarregar {endpoint} {params} després de {MAX_RETRIES} intents")


def download_collection(session, username: str, sleep=None) -> list[dict]:
    base = {"username": username, "own": 1, "stats": 1, "version": 1}
    # BGG barreja les expansions amb els jocs si no es demanen per separat
    rows = parse_collection(
        get_xml(session, "collection", {**base, "excludesubtype": "boardgameexpansion"}, sleep),
        "standalone",
    )
    rows += parse_collection(
        get_xml(session, "collection", {**base, "subtype": "boardgameexpansion"}, sleep),
        "expansion",
    )
    ids = list(dict.fromkeys(r["objectid"] for r in rows))
    weights: dict[str, str] = {}
    for batch in chunks(ids, THING_BATCH):
        weights.update(parse_weights(get_xml(session, "thing", {"id": ",".join(batch), "stats": 1}, sleep)))
    for r in rows:
        r["avgweight"] = weights.get(r["objectid"], "0")
    return sorted(rows, key=lambda r: r["objectname"].casefold())


def download_plays(session, username: str, sleep=None) -> list[dict]:
    rows, page = [], 1
    while True:
        total, page_rows = parse_plays(get_xml(session, "plays", {"username": username, "page": page}, sleep))
        rows += page_rows
        print(f"  Pàgina {page}: {len(rows)}/{total} partides")
        if not page_rows or len(rows) >= total:
            break
        page += 1
    # Més recents primer, com l'exportació
    return sorted(rows, key=lambda r: (r["Date"], int(r["Play ID"] or 0)), reverse=True)


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--username", default=os.environ.get("BGG_USERNAME") or DEFAULT_USERNAME)
    parser.add_argument("--only", choices=["collection", "plays"], help="descarrega només una de les dues")
    args = parser.parse_args(argv)

    session = make_session()
    try:
        if args.only in (None, "collection"):
            print(f"Descarregant la col·lecció de {args.username}...")
            rows = download_collection(session, args.username)
            if not rows:
                print("ERROR: BGG ha retornat una col·lecció buida; no es sobreescriu el CSV.", file=sys.stderr)
                return 1
            changed = write_csv(rows, COLLECTION_COLUMNS, COLLECTION_CSV, quote_all=True)
            print(f"  {len(rows)} jocs · {'actualitzat' if changed else 'sense canvis'}: {COLLECTION_CSV.name}")

        if args.only in (None, "plays"):
            print(f"Descarregant les partides de {args.username}...")
            rows = download_plays(session, args.username)
            if not rows:
                print("ERROR: BGG no ha retornat cap partida; no es sobreescriu el CSV.", file=sys.stderr)
                return 1
            changed = write_csv(rows, PLAYS_COLUMNS, PLAYS_CSV, quote_all=False)
            print(f"  {len(rows)} partides · {'actualitzat' if changed else 'sense canvis'}: {PLAYS_CSV.name}")
    except BGGAuthError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
