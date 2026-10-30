from lolwp.riot.store import Store


def details(duration=1800):
    return {"metadata": {"matchId": "SG2_1"}, "info": {"gameDuration": duration, "queueId": 420}}


def timeline(n_frames):
    return {"info": {"frames": [{"timestamp": i * 60000} for i in range(n_frames)]}}


def test_add_players_ignores_duplicates(tmp_path):
    s = Store(str(tmp_path / "db.sqlite"))
    assert s.add_players([("p1", "EMERALD", "I")]) == 1
    assert s.add_players([("p1", "EMERALD", "I")]) == 0
    assert s.counts()["players"] == 1


def test_players_todo_and_done(tmp_path):
    s = Store(str(tmp_path / "db.sqlite"))
    s.add_players([("p1", "EMERALD", "I"), ("p2", "DIAMOND", "II")])
    s.mark_player_done("p1")
    assert s.players_todo(10) == ["p2"]


def test_details_roundtrip(tmp_path):
    s = Store(str(tmp_path / "db.sqlite"))
    s.add_match_ids(["SG2_1"])
    s.save_details("SG2_1", details())
    assert list(s.iter_matches()) == [("SG2_1", details(), None)]


def test_details_todo_excludes_fetched(tmp_path):
    s = Store(str(tmp_path / "db.sqlite"))
    s.add_match_ids(["SG2_1", "SG2_2"])
    s.save_details("SG2_1", None)
    assert s.details_todo(10) == ["SG2_2"]
    assert list(s.iter_matches()) == []


def test_timeline_trimmed_to_16_frames(tmp_path):
    s = Store(str(tmp_path / "db.sqlite"))
    s.add_match_ids(["SG2_1"])
    s.save_details("SG2_1", details())
    s.save_timeline("SG2_1", timeline(30))
    _, _, tl = next(s.iter_matches())
    assert len(tl["info"]["frames"]) == 16


def test_timelines_todo_skips_short_games(tmp_path):
    s = Store(str(tmp_path / "db.sqlite"))
    s.add_match_ids(["SG2_short", "SG2_long"])
    s.save_details("SG2_short", details(900))
    s.save_details("SG2_long", details(1200))
    assert s.timelines_todo(10, seed=42) == ["SG2_long"]
    s.save_timeline("SG2_long", timeline(20))
    assert s.timelines_todo(10, seed=42) == []


def test_reopen_keeps_progress(tmp_path):
    path = str(tmp_path / "db.sqlite")
    s = Store(path)
    s.add_players([("p1", "EMERALD", "I")])
    s.add_match_ids(["SG2_1", "SG2_2"])
    s.save_details("SG2_1", details())
    before = s.counts()
    assert Store(path).counts() == before == {"players": 1, "match_ids": 2, "details": 1, "timelines": 0}
