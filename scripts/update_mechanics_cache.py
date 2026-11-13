"""Actualitza pages/bgg_mechanics_cache.csv amb les mecàniques de BGG.

Fa servir l'API XML2 oficial de BGG (endpoint /thing, fins a 20 jocs per petició).
Només descarrega els jocs que falten a la caché o que hi són sense mecàniques
(amb --all es tornen a descarregar tots).

BGG demana registrar les aplicacions que fan servir l'API: si tens un token,
posa'l a la variable d'entorn BGG_API_TOKEN (a GitHub, com a secret del repositori).

Ús:
    python scripts/update_mechanics_cache.py            # actualitza la caché
    python scripts/update_mechanics_cache.py --dry-run  # només diu què faria
    python scripts/update_mechanics_cache.py --all      # ho torna a descarregar tot
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
COLLECTION_CSV = Path(os.environ.get("BGG_COLLECTION_CSV", ROOT / "pages" / "collection.csv"))
CACHE_CSV = Path(os.environ.get("BGG_CACHE_CSV", ROOT / "pages" / "bgg_mechanics_cache.csv"))

API_URL = "https://boardgamegeek.com/xmlapi2/thing"
BATCH_SIZE = 20          # màxim que accepta l'API per petició
PAUSE_SECONDS = 5        # BGG recomana no fer més d'una petició cada ~5 s
MAX_RETRIES = 6


class BGGAuthError(RuntimeError):
    pass


# ------------------------------------------------------------------ funcions pures

def load_cache(path: Path) -> dict[int, list[str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or not {"objectid", "mechanics"} <= set(df.columns):
        return {}
    return {int(oid): list(ast.literal_eval(m)) for oid, m in zip(df["objectid"], df["mechanics"])}


def save_cache(cache: dict[int, list[str]], path: Path) -> None:
    rows = [{"objectid": oid, "mechanics": str(mecs)} for oid, mecs in cache.items()]
    pd.DataFrame(rows, columns=["objectid", "mechanics"]).to_csv(path, index=False)


def load_collection_ids(path: Path) -> list[int]:
    df = pd.read_csv(path)
    ids = pd.to_numeric(df["objectid"], errors="coerce").dropna().astype(int)
    return list(dict.fromkeys(ids))  # sense duplicats, mantenint l'ordre


def ids_to_fetch(object_ids: list[int], cache: dict[int, list[str]], refresh_all: bool = False) -> list[int]:
    if refresh_all:
        return list(object_ids)
    # També reintentem els que són a la caché sense mecàniques (pot ser un error antic)
    return [oid for oid in object_ids if not cache.get(oid)]


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def parse_things(xml_content: bytes) -> dict[int, list[str]]:
    """Retorna {objectid: [mecàniques]} per a cada <item> de la resposta /thing."""
    root = ET.fromstring(xml_content)
    result = {}
    for item in root.findall("item"):
        oid = int(item.get("id"))
        result[oid] = [
            link.get("value")
            for link in item.findall("link")
            if link.get("type") == "boardgamemechanic" and link.get("value")
        ]
    return result


# ------------------------------------------------------------------ xarxa

def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "streamlit-bgg-mechanics-updater"
    token = os.environ.get("BGG_API_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def fetch_batch(session: requests.Session, ids: list[int], sleep=time.sleep) -> dict[int, list[str]]:
    params = {"id": ",".join(map(str, ids)), "type": "boardgame,boardgameexpansion"}
    wait = PAUSE_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        r = session.get(API_URL, params=params, timeout=30)
        if r.status_code == 200:
            return parse_things(r.content)
        if r.status_code in (401, 403):
            raise BGGAuthError(
                f"BGG ha rebutjat la petició ({r.status_code}). Cal un token d'aplicació de BGG "
                "a la variable d'entorn BGG_API_TOKEN (secret del repositori a GitHub)."
            )
        # 202 = petició en cua, 429 = massa peticions, 5xx = error temporal
        print(f"    resposta {r.status_code}, reintent {attempt}/{MAX_RETRIES} d'aquí {wait} s")
        sleep(wait)
        wait = min(wait * 2, 120)
    raise RuntimeError(f"No s'han pogut descarregar els jocs {ids} després de {MAX_RETRIES} intents")


def update_cache(object_ids, cache, session, refresh_all=False, sleep=time.sleep) -> tuple[int, int]:
    """Actualitza `cache` in situ. Retorna (jocs demanats, jocs amb mecàniques nous)."""
    pending = ids_to_fetch(object_ids, cache, refresh_all)
    amb_mecaniques = 0
    for n, batch in enumerate(chunks(pending, BATCH_SIZE), start=1):
        print(f"  Lot {n}: {len(batch)} jocs")
        found = fetch_batch(session, batch, sleep=sleep)
        for oid in batch:
            mecs = found.get(oid, [])
            if mecs and not cache.get(oid):
                amb_mecaniques += 1
            # No esborrem mecàniques bones si BGG no ha retornat el joc
            if mecs or oid not in cache:
                cache[oid] = mecs
        sleep(PAUSE_SECONDS)
    return len(pending), amb_mecaniques


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="torna a descarregar tots els jocs")
    parser.add_argument("--dry-run", action="store_true", help="no fa cap petició ni escriu res")
    args = parser.parse_args(argv)

    object_ids = load_collection_ids(COLLECTION_CSV)
    cache = load_cache(CACHE_CSV)
    pending = ids_to_fetch(object_ids, cache, args.all)
    print(f"Col·lecció: {len(object_ids)} jocs · caché: {len(cache)} · a descarregar: {len(pending)}")

    if args.dry_run or not pending:
        return 0

    try:
        demanats, nous = update_cache(object_ids, cache, make_session(), args.all)
    except BGGAuthError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    save_cache(cache, CACHE_CSV)
    print(f"Fet: {demanats} jocs consultats, {nous} amb mecàniques noves. Caché: {len(cache)} jocs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
