import pytest

from lolwp.features.draft_stats import DraftStats, oof_stats, smooth
from lolwp.features.parse import MatchRecord

BLUE = {"top": 86, "jungle": 64, "mid": 103, "bot": 222, "support": 412}
RED = {"top": 122, "jungle": 121, "mid": 157, "bot": 51, "support": 89}


@pytest.fixture
def records():
    recs = []
    for i in range(10):
        blue, red = (BLUE, RED) if i % 3 else (RED, BLUE)
        recs.append(MatchRecord(f"SG2_{i}", "15.20", int(i % 5 != 0), 1800, dict(blue), dict(red)))
    return recs


def test_smooth_values():
    assert smooth(0, 0) == 0.5 and smooth(3, 3) == pytest.approx(13 / 23)


def test_unseen_champion_gets_half(records):  # Review Focus 1
    s = DraftStats.fit(records)
    assert s.champ_rate(99999) == 0.5
    assert s.lane_rate("mid", 99999, 103) == 0.5
    assert s.duo_rate(1, 99999) == 0.5


def test_seen_champion_rate_matches_counts(records):
    s = DraftStats.fit(records)
    games = sum(1 for r in records if 103 in r.blue.values() or 103 in r.red.values())
    wins = sum((r.blue_win if 103 in r.blue.values() else 1 - r.blue_win) for r in records)
    assert s.champ_rate(103) == pytest.approx(smooth(wins, games))
    assert s.champ_rate(103) != 0.5


def test_lane_rate_is_symmetric(records):
    s = DraftStats.fit(records)
    assert s.lane_rate("mid", 103, 157) != 0.5
    assert s.lane_rate("mid", 103, 157) == pytest.approx(1 - s.lane_rate("mid", 157, 103))


def test_oof_excludes_own_match(records):
    oof = oof_stats(records)
    assert set(oof) == {r.match_id for r in records}
    for mid, st in oof.items():
        assert mid not in st.match_ids
        assert len(st.match_ids) == 8


def test_json_roundtrip(tmp_path, records):
    s = DraftStats.fit(records)
    path = str(tmp_path / "stats.json")
    s.to_json(path)
    t = DraftStats.from_json(path)
    assert t.match_ids == s.match_ids
    assert t.champ_rate(103) == s.champ_rate(103)
    assert t.lane_rate("mid", 157, 103) == s.lane_rate("mid", 157, 103)
    assert t.duo_rate(222, 412) == s.duo_rate(222, 412)
