import os

from lolwp.config import Settings
from lolwp.refresh import refresh
from lolwp.riot.client import KeyExpiredError
from lolwp.riot.store import Store
from synthetic import fill_store


class Expired:
    def get(self, url, params=None):
        raise KeyExpiredError("expired")


class NeverCalled:
    def get(self, url, params=None):
        raise AssertionError(f"unexpected call {url}")


def _settings(tmp_path, db):
    return Settings("k", "sg2", "sea", db, str(tmp_path / "data"), str(tmp_path / "art"))


def test_refresh_stops_on_expiry(tmp_path, capsys):
    s = _settings(tmp_path, str(tmp_path / "db.sqlite"))
    store = Store(s.db_path)
    store.add_match_ids(["SG2_1"])
    assert refresh(Expired(), store, s, ids=10, details=10, timelines=10,
                   data_dir=s.data_dir, artifacts_dir=s.artifacts_dir) is False
    assert "Key expired" in capsys.readouterr().out
    assert not os.path.exists(os.path.join(s.data_dir, "draft.parquet"))


def test_refresh_runs_pipeline(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    store = fill_store(str(tmp_path / "db.sqlite"), str(art / "champions.json"), per_patch=60)
    s = _settings(tmp_path, str(tmp_path / "db.sqlite"))
    totals = store.counts()
    assert refresh(NeverCalled(), store, s, ids=totals["match_ids"], details=totals["details"],
                   timelines=totals["timelines"], data_dir=s.data_dir, artifacts_dir=s.artifacts_dir) is True
    assert os.path.exists(os.path.join(s.artifacts_dir, "metrics.json"))
