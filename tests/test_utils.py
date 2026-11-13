import pandas as pd

from utils import clean_mechanics, split_players


def test_clean_mechanics():
    s = pd.Series(["Pseudo-wargames", "", None, "Car-driven", "  ", "Gestió de la mà"])
    assert clean_mechanics(s).tolist() == [
        "Pseudo-Wargame", "No informat", "No informat", "Card-driven", "No informat", "Gestió de mà"
    ]


def test_split_players():
    assert split_players("gerard, guillaum ,") == ["gerard", "guillaum"]
    assert split_players(float("nan")) == []
