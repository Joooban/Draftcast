import pandas as pd

from lolwp.config import ROLES, STATE_FEATURES
from lolwp.features.build import build, split_by_patch
from lolwp.features.draft import DRAFT_FEATURES
from lolwp.features.draft_stats import DraftStats
from lolwp.features.parse import MatchRecord
from synthetic import fill_store


def rec(mid, patch):
    team = {r: i for i, r in enumerate(ROLES)}
    return MatchRecord(mid, patch, 1, 1800, team, team)


def test_split_newest_patch_is_test():
    recs = [rec("a", "15.19"), rec("b", "15.21"), rec("c", "15.20"), rec("d", "15.9")]
    train, test, test_patch = split_by_patch(recs)
    assert test_patch == "15.21"
    assert [r.match_id for r in test] == ["b"]
    assert {r.match_id for r in train} == {"a", "c", "d"}


def test_no_test_match_in_saved_stats(tmp_path):
    champs = str(tmp_path / "champions.json")
    store = fill_store(str(tmp_path / "lol.db"), champs, per_patch=40)
    summary = build(store, champs, str(tmp_path / "data"), str(tmp_path / "art"))
    draft = pd.read_parquet(tmp_path / "data" / "draft.parquet")
    stats = DraftStats.from_json(str(tmp_path / "art" / "draft_stats.json"))
    test_ids = set(draft.loc[draft.split == "test", "match_id"])
    assert summary["test_patch"] == "15.21" and test_ids
    assert not stats.match_ids & test_ids
    assert stats.match_ids == set(draft.loc[draft.split == "train", "match_id"])


def test_dataset_columns_and_filters(tmp_path):
    champs = str(tmp_path / "champions.json")
    store = fill_store(str(tmp_path / "lol.db"), champs, per_patch=40)
    build(store, champs, str(tmp_path / "data"), str(tmp_path / "art"))
    draft = pd.read_parquet(tmp_path / "data" / "draft.parquet")
    state = pd.read_parquet(tmp_path / "data" / "state.parquet")
    role_cols = [f"{s}_{r}" for s in ("blue", "red") for r in ROLES]
    assert list(draft.columns) == ["match_id", "patch", "split", "blue_win", *DRAFT_FEATURES, *role_cols]
    assert list(state.columns) == ["match_id", "patch", "split", "blue_win", "duration_s", *STATE_FEATURES]
    assert len(draft) == 120 - 3          # every 40th match has a blank position
    assert state.duration_s.min() >= 960  # every 25th match is short
    assert set(state.match_id) <= set(draft.match_id)
    assert len(state) < len(draft)        # every 10th match has no timeline
