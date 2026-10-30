"""Settings and project-wide constants."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

ROLES: list[str] = ["top", "jungle", "mid", "bot", "support"]
RIOT_POSITION_TO_ROLE: dict[str, str] = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "BOTTOM": "bot",
    "UTILITY": "support",
}

STATE_FEATURES: list[str] = [
    "gold_diff",
    "xp_diff",
    "cs_diff",
    "level_diff",
    "kills_diff",
    "dragons_diff",
    "grubs_diff",
    "heralds_diff",
    "towers_diff",
    "plates_diff",
    "wards_placed_diff",
    "wards_killed_diff",
    "first_blood_blue",
    "first_dragon_blue",
    "first_tower_blue",
]

STATE_RANGES: dict[str, tuple[int, int]] = {
    "gold_diff": (-15000, 15000),
    "xp_diff": (-15000, 15000),
    "cs_diff": (-200, 200),
    "level_diff": (-20, 20),
    "kills_diff": (-30, 30),
    "dragons_diff": (-3, 3),
    "grubs_diff": (-6, 6),
    "heralds_diff": (-1, 1),
    "towers_diff": (-6, 6),
    "plates_diff": (-15, 15),
    "wards_placed_diff": (-60, 60),
    "wards_killed_diff": (-30, 30),
    "first_blood_blue": (-1, 1),
    "first_dragon_blue": (-1, 1),
    "first_tower_blue": (-1, 1),
}

ACCOUNT_ROUTING = "asia"
QUEUE_ID = 420
MATCH_WINDOW_DAYS = 42
MIN_DURATION_S = 960
STATE_MINUTE = 15

PRIOR_GAMES = 20
OOF_FOLDS = 5
SEED = 42
MIN_INTERVAL_S = 1.25


@dataclass
class Settings:
    api_key: str
    platform: str
    region: str
    db_path: str
    data_dir: str
    artifacts_dir: str


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    data_dir = os.environ.get("DATA_DIR", str(ROOT / "data"))
    return Settings(
        api_key=os.environ.get("RIOT_API_KEY", ""),
        platform=os.environ.get("PLATFORM", "sg2"),
        region=os.environ.get("REGION", "sea"),
        db_path=os.environ.get("DB_PATH", os.path.join(data_dir, "lol.db")),
        data_dir=data_dir,
        artifacts_dir=os.environ.get("ARTIFACTS_DIR", str(ROOT / "artifacts")),
    )
