import pandas as pd
import pytest

from lolwp.config import STATE_FEATURES
from lolwp.models.common import COMBINED_FEATURES
from lolwp.models.refit import ModelBank


@pytest.fixture
def bank(synthetic_dirs):
    return ModelBank(synthetic_dirs.artifacts_dir)


def test_full_set_uses_saved_model(bank):
    fm = bank.get("state", 15, tuple(STATE_FEATURES))
    assert bank.fits == 0 and fm.features == STATE_FEATURES and 0.5 <= fm.auc <= 1


def test_refit_is_cached(bank):
    first = bank.get("state", 15, ("gold_diff", "kills_diff"))
    second = bank.get("state", 15, ("gold_diff", "kills_diff"))
    assert bank.fits == 1 and first is second
    assert list(first.model.feature_names_in_) == ["gold_diff", "kills_diff"]


def test_refit_scored_on_test_rows(bank, synthetic_dirs):
    fm = bank.get("combined", 15, ("draft_logit", "gold_diff"))
    rows = pd.read_parquet(f"{synthetic_dirs.artifacts_dir}/state_rows_m15.parquet")
    assert 0.5 <= fm.auc <= 1 and 0 <= fm.accuracy <= 1
    assert sum(b["n"] for b in fm.calibration) == (rows.split == "test").sum()


def test_minute_10_uses_m10_rows(bank, monkeypatch):
    import lolwp.models.refit as refit

    loaded = []
    real = refit.joblib.load
    monkeypatch.setattr(refit.joblib, "load", lambda p: loaded.append(p) or real(p))
    bank.get("combined", 10, tuple(COMBINED_FEATURES))
    assert loaded and loaded[0].replace("\\", "/").endswith("combined_m10_model.joblib")


def test_cache_survives_eviction_between_check_and_use(bank):
    # Under FastAPI's threadpool another request can evict a key right after this one sees it.
    from collections import OrderedDict

    class EvictedAfterCheck(OrderedDict):
        def __contains__(self, key):
            found = super().__contains__(key)
            if found:
                self.pop(key)
            return found

    first = bank.get("state", 15, ("gold_diff",))
    bank._cache = EvictedAfterCheck(bank._cache)
    again = bank.get("state", 15, ("gold_diff",))
    assert again.features == first.features
