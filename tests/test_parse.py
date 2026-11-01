import copy
import json
from pathlib import Path

import pytest

from lolwp.features.parse import parse_match, parse_match_checked, patch_key

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


def test_other_platform_rejected(match):
    match["metadata"]["matchId"] = "OC1_711130978"
    assert parse_match(match, platform="sg2") is None


def test_own_platform_kept(match):
    assert parse_match(match, platform="sg2") is not None


def _reason(match, **kw):
    record, reason = parse_match_checked(match, **kw)
    assert (record is None) == (reason is not None)
    return reason


def test_checked_valid(match):
    record, reason = parse_match_checked(match)
    assert record is not None and reason is None


@pytest.mark.parametrize("mutate, expected", [
    (lambda m: m["info"].update(queueId=440), "wrong_queue"),
    (lambda m: m["info"]["participants"][7].update(gameEndedInEarlySurrender=True), "early_surrender"),
    (lambda m: m["info"]["participants"][3].update(teamPosition=""), "role_missing"),
    (lambda m: m["info"]["participants"][3].pop("teamPosition"), "role_missing"),
    (lambda m: m["info"]["participants"][1].update(teamPosition="TOP"), "role_missing"),
    (lambda m: m["info"].update(participants=m["info"]["participants"][:9]), "not_10_players"),
])
def test_checked_reasons(match, mutate, expected):
    mutate(match)
    assert _reason(match) == expected


def test_checked_other_platform(match):
    match["metadata"]["matchId"] = "OC1_711130978"
    assert _reason(match, platform="sg2") == "other_platform"


def test_checked_malformed():
    assert _reason({}) == "malformed"
