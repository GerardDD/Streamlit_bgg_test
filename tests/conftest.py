import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _repo_root_and_no_network(monkeypatch):
    """Les pàgines llegeixen rutes relatives (pages/...) i la resenya fa crides HTTP."""
    monkeypatch.chdir(ROOT)

    import requests

    def _sense_xarxa(*args, **kwargs):
        raise requests.ConnectionError("xarxa desactivada als tests")

    monkeypatch.setattr(requests, "get", _sense_xarxa)


@pytest.fixture
def root():
    return ROOT


def pytest_configure(config):
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
