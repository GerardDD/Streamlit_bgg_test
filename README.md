# Streamlit_bgg_test


IMPORTANT: és necessari afegir totes les llibreries de python que es facin servir al .py dins de l'arxiu requirements.

Datsources:
  - pagina princpal: CSV exportat manualment de la bgg filtrant per All or Owned. Els dos funcionen
  - pagina 2: CSV exportat manualment de geekgroup, de l'apartat "Plays".


REFRESH de les mecaniques de la col·lecció:

cal executar el arxiu de jupyter adjunt aqui (des de dins del repositori). Llegeix `pages/collection.csv` i actualitza directament `pages/bgg_mechanics_cache.csv`; després només cal fer commit i push del csv.
Si les dades són en una altra carpeta, es poden indicar amb les variables d'entorn `BGG_COLLECTION_CSV` i `BGG_CACHE_CSV`.

