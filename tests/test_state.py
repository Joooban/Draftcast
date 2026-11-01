import json
from pathlib import Path

import pytest

from lolwp.config import STATE_FEATURES
from lolwp.features.state import state_features

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def match():
    return json.loads((FIX / "match_valid.json").read_text())


@pytest.fixture
def timeline():
    return json.loads((FIX / "timeline_small.json").read_text())


def test_diffs_at_15(match, timeline):
    f = state_features(match, timeline)
    assert f["gold_diff"] == 1000 and "xp_diff" not in f and f["cs_diff"] == 25
    assert f["level_diff"] == 0
    assert f["kills_diff"] == 1  # late kill ignored
    assert f["grubs_diff"] == 1 and f["dragons_diff"] == -1
    assert f["towers_diff"] == 1 and f["plates_diff"] == 2
    assert f["wards_placed_diff"] == -1
    assert f["first_blood_blue"] == 1 and f["first_dragon_blue"] == -1 and f["first_tower_blue"] == 1
    assert "heralds_diff" not in f and f["wards_killed_diff"] == 0


def test_short_timeline_returns_none(match, timeline):
    timeline["info"]["frames"] = timeline["info"]["frames"][:10]
    assert state_features(match, timeline) is None


def test_keys_match_config(match, timeline):
    assert list(state_features(match, timeline)) == STATE_FEATURES


def test_team_comes_from_match_not_participant_id(match, timeline):
    for p in match["info"]["participants"]:
        p["teamId"] = 300 - p["teamId"]
    f = state_features(match, timeline)
    assert f["gold_diff"] == -1000 and f["kills_diff"] == -1 and f["wards_placed_diff"] == 1


def test_minute_10_reads_frame_10(match, timeline):
    f = state_features(match, timeline, minute=10)
    team = {p["participantId"]: p["teamId"] for p in match["info"]["participants"]}
    frame = timeline["info"]["frames"][10]["participantFrames"]
    gold = sum((1 if team[int(k)] == 100 else -1) * v["totalGold"] for k, v in frame.items())
    assert f["gold_diff"] == gold and "xp_diff" not in f
