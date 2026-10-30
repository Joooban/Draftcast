from lolwp.config import Settings
from lolwp.riot import collect
from lolwp.riot.client import KeyExpiredError
from lolwp.riot.store import Store


def settings(tmp_path):
    return Settings("k", "sg2", "sea", str(tmp_path / "db.sqlite"), str(tmp_path), str(tmp_path / "art"))


class FakeClient:
    """Answers by URL; `fail_on` makes the Nth call raise KeyExpiredError."""

    def __init__(self, route, fail_on=None):
        self.route = route
        self.fail_on = fail_on
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, params))
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise KeyExpiredError("expired")
        return self.route(url, params)


def test_seed_stores_puuids_from_entries(tmp_path):
    s = settings(tmp_path)
    store = Store(s.db_path)

    def route(url, params):
        if "/EMERALD/I" in url and params.get("page") == 1:
            return [{"puuid": "a"}, {"puuid": "b"}]
        return []

    client = FakeClient(route)
    assert collect.seed(client, store, s) == 2
    assert store.counts()["players"] == 2
    urls = {u for u, _ in client.calls}
    assert any(u.startswith("https://sg2.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/") for u in urls)
    assert any("/CHALLENGER/I" in u for u in urls)
    assert not any("/CHALLENGER/II" in u for u in urls)


def test_seed_skips_entries_without_puuid(tmp_path):
    s = settings(tmp_path)
    store = Store(s.db_path)
    client = FakeClient(lambda url, p: [{"summonerId": "x"}, {"puuid": "a"}] if "/DIAMOND/II" in url else [])
    assert collect.seed(client, store, s) == 1


def test_collect_ids_passes_start_time_and_queue(tmp_path):
    s = settings(tmp_path)
    store = Store(s.db_path)
    store.add_players([("p1", "EMERALD", "I")])
    client = FakeClient(lambda url, p: ["SG2_1", "SG2_2"])
    assert collect.collect_ids(client, store, s, limit=10, now=1_000_000_000) == 2
    url, params = client.calls[0]
    assert url == "https://sea.api.riotgames.com/lol/match/v5/matches/by-puuid/p1/ids"
    assert params == {"startTime": 1_000_000_000 - 42 * 86400, "queue": 420, "count": 50}
    assert store.players_todo(10) == []


def test_collect_details_resume_after_key_expiry(tmp_path):
    s = settings(tmp_path)
    store = Store(s.db_path)
    ids = [f"SG2_{i}" for i in range(5)]
    store.add_match_ids(ids)

    def route(url, params):
        return {"info": {"gameDuration": 1800}, "metadata": {"matchId": url.rsplit("/", 1)[1]}}

    first = FakeClient(route, fail_on=3)
    assert collect.collect_details(first, store, s, limit=100) == 2
    second = FakeClient(route)
    assert collect.collect_details(second, store, s, limit=100) == 3
    fetched = [u.rsplit("/", 1)[1] for u, _ in first.calls[:2] + second.calls]
    assert sorted(fetched) == sorted(ids)
    assert store.counts()["details"] == 5


def test_collect_timelines_uses_timeline_url(tmp_path):
    s = settings(tmp_path)
    store = Store(s.db_path)
    store.save_details("SG2_1", {"info": {"gameDuration": 1800}})
    client = FakeClient(lambda url, p: {"info": {"frames": [{}] * 20}})
    assert collect.collect_timelines(client, store, s, limit=10) == 1
    assert client.calls[0][0] == "https://sea.api.riotgames.com/lol/match/v5/matches/SG2_1/timeline"
    assert store.counts()["timelines"] == 1


def test_transient_failure_leaves_match_unfetched(tmp_path):
    from lolwp.riot.client import TransientError
    s = settings(tmp_path)
    store = Store(s.db_path)
    store.add_match_ids(["SG2_0", "SG2_1", "SG2_2"])

    def route(url, params):
        if url.endswith("SG2_1"):
            raise TransientError("503 x5")
        return {"info": {"gameDuration": 1800}}

    assert collect.collect_details(FakeClient(route), store, s, limit=100) == 2
    assert store.details_todo(10) == ["SG2_1"]  # retried on the next run, not lost


def test_transient_timeline_failure_leaves_unfetched(tmp_path):
    from lolwp.riot.client import TransientError
    s = settings(tmp_path)
    store = Store(s.db_path)
    store.save_details("SG2_1", {"info": {"gameDuration": 1800}})

    def route(url, params):
        raise TransientError("503 x5")

    assert collect.collect_timelines(FakeClient(route), store, s, limit=10) == 0
    assert store.timelines_todo(10, seed=42) == ["SG2_1"]


def test_transient_ids_failure_leaves_player_todo(tmp_path):
    from lolwp.riot.client import TransientError
    s = settings(tmp_path)
    store = Store(s.db_path)
    store.add_players([("p1", "EMERALD", "I"), ("p2", "EMERALD", "I")])

    def route(url, params):
        if "/p1/" in url:
            raise TransientError("503 x5")
        return ["SG2_9"]

    assert collect.collect_ids(FakeClient(route), store, s, limit=10) == 1
    assert store.players_todo(10) == ["p1"]


def test_transient_seed_page_skipped(tmp_path):
    from lolwp.riot.client import TransientError
    s = settings(tmp_path)
    store = Store(s.db_path)

    def route(url, params):
        if "/EMERALD/I" in url and params.get("page") == 1:
            raise TransientError("503 x5")
        return [{"puuid": "a"}] if "/CHALLENGER/" in url else []

    assert collect.seed(FakeClient(route), store, s) == 1
