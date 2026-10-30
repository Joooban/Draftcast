from pathlib import Path

from lolwp import champions

FIXTURE = str(Path(__file__).parent / "fixtures" / "ddragon_champions.json")


def test_load_returns_all_champions():
    champs = champions.load(FIXTURE)
    assert sorted(champs) == [86, 103, 222]
    assert champs[86].name == "Garen" and champs[86].attack == 7 and champs[86].magic == 1


def test_tags_preserved():
    assert champions.load(FIXTURE)[103].tags == ["Mage", "Assassin"]


def test_icon_url_has_version_and_key():
    url = champions.load(FIXTURE)[103].icon_url
    assert "15.21.1" in url and url.endswith("/Ahri.png")
