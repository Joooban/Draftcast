import copy
import json
from pathlib import Path

import pytest

from lolwp.features.parse import parse_match, patch_key

FIXTURE = Path(__file__).parent / "fixtures" / "match_valid.json"


@pytest.fixture
def match():
    return json.loads(FIXTURE.read_text())


def test_valid_match_parsed(match):
    r = parse_match(match)
    assert r.match_id == "SG2_1000" and r.patch == "15.21" and r.blue_win == 1 and r.duration_s == 1800
    assert r.blue == {"top": 86, "jungle": 64, "mid": 103, "bot": 222, "support": 412}
    assert r.red["mid"] == 157


def test_red_win_parsed(match):
    for t in match["info"]["teams"]:
        t["win"] = not t["win"]
    assert parse_match(match).blue_win == 0


def test_wrong_queue_rejected(match):
    match["info"]["queueId"] = 440
    assert parse_match(match) is None


def test_early_surrender_rejected(match):
    match["info"]["participants"][7]["gameEndedInEarlySurrender"] = True
    assert parse_match(match) is None


def test_blank_team_position_rejected(match):
    match["info"]["participants"][3]["teamPosition"] = ""
    assert parse_match(match) is None


def test_missing_team_position_rejected(match):
    del match["info"]["participants"][3]["teamPosition"]
    assert parse_match(match) is None


def test_duplicate_role_rejected(match):
    match["info"]["participants"][1]["teamPosition"] = "TOP"
    assert parse_match(match) is None


def test_patch_key_orders_numerically():
    assert sorted(["15.9", "15.21", "15.10"], key=patch_key) == ["15.9", "15.10", "15.21"]
