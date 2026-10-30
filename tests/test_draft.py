from pathlib import Path

import pytest

from lolwp import champions
from lolwp.features.draft import DRAFT_FEATURES, draft_features
from lolwp.features.draft_stats import DraftStats
from lolwp.features.parse import MatchRecord

CHAMPS = champions.load(str(Path(__file__).parent / "fixtures" / "ddragon_champions.json"))
BLUE = {"top": 86, "jungle": 64, "mid": 103, "bot": 222, "support": 412}
RED = {"top": 122, "jungle": 121, "mid": 157, "bot": 51, "support": 89}


@pytest.fixture
def stats():
    recs = [MatchRecord(f"SG2_{i}", "15.20", int(i % 3 != 0), 1800, dict(BLUE), dict(RED)) for i in range(9)]
    return DraftStats.fit(recs)


def test_keys_and_order(stats):
    assert list(draft_features(BLUE, RED, stats, CHAMPS)) == DRAFT_FEATURES


def test_mirror_draft_is_zero(stats):
    f = draft_features(BLUE, dict(BLUE), stats, CHAMPS)
    assert all(v == pytest.approx(0) for v in f.values())


def test_swap_negates(stats):
    f = draft_features(BLUE, RED, stats, CHAMPS)
    g = draft_features(RED, BLUE, stats, CHAMPS)
    assert f["champ_wr_diff"] != 0 and f["lane_mid"] != 0
    for k in DRAFT_FEATURES:
        assert g[k] == pytest.approx(-f[k])


def test_tags_counted(stats):
    # blue has Garen (Fighter, Tank) and Jinx; red has Ahri; all other ids unknown to CHAMPS
    f = draft_features({**BLUE, "mid": 157}, {**RED, "mid": 103}, stats, CHAMPS)
    assert f["tag_Tank_diff"] == 1 and f["tag_Fighter_diff"] == 1 and f["tag_Mage_diff"] == -1
    assert f["tag_Marksman_diff"] == 1 and f["tag_Support_diff"] == 0
    assert f["attack_diff"] == (7 + 9) - 3 and f["magic_diff"] == (1 + 2) - 8
