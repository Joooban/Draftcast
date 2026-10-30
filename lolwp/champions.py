"""Champion data from Data Dragon (public, no API key)."""
import json
from dataclasses import dataclass
from pathlib import Path

import requests

DDRAGON = "https://ddragon.leagueoflegends.com"
TAGS = ["Tank", "Fighter", "Mage", "Assassin", "Marksman", "Support"]


@dataclass
class Champion:
    id: int
    key: str
    name: str
    tags: list[str]
    attack: int
    magic: int
    icon_url: str


def download(data_dir: str) -> str:
    version = requests.get(f"{DDRAGON}/api/versions.json", timeout=30).json()[0]
    resp = requests.get(f"{DDRAGON}/cdn/{version}/data/en_US/champion.json", timeout=30)
    resp.raise_for_status()
    out = Path(data_dir) / "ddragon" / "champions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(resp.text, encoding="utf-8")
    return version


def load(path: str) -> dict[int, Champion]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    version = raw["version"]
    champs = {}
    for c in raw["data"].values():
        champ = Champion(
            id=int(c["key"]),
            key=c["id"],
            name=c["name"],
            tags=list(c["tags"]),
            attack=int(c["info"]["attack"]),
            magic=int(c["info"]["magic"]),
            icon_url=f"{DDRAGON}/cdn/{version}/img/champion/{c['id']}.png",
        )
        champs[champ.id] = champ
    return champs
