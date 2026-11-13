# Streamlit_bgg_test

App de Streamlit per explorar una ludoteca de BoardGameGeek: col·lecció, estadístiques de partides i un recomanador personalitzat.

## Executar en local

```bash
pip install -r requirements.txt
streamlit run streamlit_bgg.py
```

IMPORTANT: totes les llibreries de Python que es facin servir als `.py` han d'estar a `requirements.txt`, amb la versió fixada (`paquet==x.y.z`).

## Fonts de dades

- **Pàgina principal i recomanador**: CSV exportat manualment de BGG (*Collection → Export*), filtrant per *All* o *Owned* (els dos funcionen). Per defecte es fa servir `pages/collection.csv`; també es pot pujar un CSV des de la barra lateral de la pàgina principal i el recomanador el farà servir mentre duri la sessió.
- **Estadístiques de partides**: CSV exportat manualment de l'apartat *Plays*. Per defecte `pages/playsMrbrussels.csv`; es pot pujar un altre CSV des de la barra lateral de la pàgina.

### Descàrrega automàtica (`pages/collection.csv` i `pages/playsMrbrussels.csv`)

Ja no cal exportar-los a mà: la GitHub Action **Descarregar col·lecció i partides de BGG** (`.github/workflows/update-bgg-data.yml`) els baixa de l'API XML oficial de BGG cada dia (o manualment des de la pestanya *Actions*), actualitza les mecàniques dels jocs nous i fa commit només si alguna cosa ha canviat.

- L'usuari de BGG és `Mrbrussels` per defecte; es pot canviar amb la variable del repositori `BGG_USERNAME` (*Settings → Secrets and variables → Actions → Variables*).
- Fa servir el mateix secret `BGG_API_TOKEN` que l'actualització de mecàniques.
- La col·lecció inclou només els jocs en propietat, amb les mateixes columnes que l'exportació manual. Els camps privats (preu pagat, comentari privat...) i els recomanats per nombre de jugadors queden buits perquè l'API pública no els dona; l'app no els fa servir.

En local:

```bash
python scripts/download_bgg_data.py                    # col·lecció i partides
python scripts/download_bgg_data.py --only plays       # només partides
python scripts/download_bgg_data.py --username Altre   # un altre usuari
```

## Mecàniques de la col·lecció (`pages/bgg_mechanics_cache.csv`)

S'actualitzen automàticament amb la GitHub Action **Actualitzar mecàniques de BGG** (`.github/workflows/update-mechanics.yml`):

- cada dilluns,
- cada vegada que es puja un `pages/collection.csv` nou a `main`,
- o manualment des de la pestanya *Actions* (opció per tornar-ho a descarregar tot).

L'Action executa `scripts/update_mechanics_cache.py`, que consulta l'API XML oficial de BGG només per als jocs nous (o que no tenien mecàniques) i fa commit del CSV si ha canviat.

BGG demana registrar les aplicacions que fan servir la seva API. Si l'Action falla amb un error 401/403, cal crear un token d'aplicació a BGG i guardar-lo com a secret del repositori amb el nom `BGG_API_TOKEN` (*Settings → Secrets and variables → Actions*).

També es pot executar en local:

```bash
python scripts/update_mechanics_cache.py --dry-run   # quants jocs falten
python scripts/update_mechanics_cache.py             # actualitza la caché
```

Si les dades són en una altra carpeta, es poden indicar amb les variables d'entorn `BGG_COLLECTION_CSV` i `BGG_CACHE_CSV`. El notebook `Tests_Scrapping_bgg.ipynb` queda com a alternativa antiga.

## Proves

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

Les proves (`tests/`) obren cada pàgina amb `streamlit.testing` i comproven que carrega sense errors i que els filtres principals funcionen. S'executen automàticament a GitHub a cada push (`.github/workflows/tests.yml`).

## Tema

Els colors són a `.streamlit/config.toml` (versió clara i fosca). L'app segueix el mode del sistema i es pot canviar des del menú ⋮ → *Settings*.
