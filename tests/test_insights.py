import json
import os

import pytest

from lolwp.models.insights import wilson


def test_wilson_half():
    lo, hi = wilson(50, 100)
    assert (round(lo, 3), round(hi, 3)) == (0.404, 0.596)


def test_wilson_zero_wins():
    lo, hi = wilson(0, 10)
    assert lo == 0 and 0 < hi < 0.35


def test_wilson_all_wins():
    lo, hi = wilson(10, 10)
    assert hi == 1 and lo > 0.65


@pytest.fixture
def insights(pipeline_dirs):
    path = os.path.join(pipeline_dirs.artifacts_dir, "insights.json")
    if not os.path.exists(path):
        pytest.skip("insights.json not built yet")
    return json.load(open(path))


def test_champion_intervals_contain_rate(insights):
    assert insights["champions"]
    for c in insights["champions"]:
        assert c["games"] >= 10 and c["name"]
        assert c["ci_low"] <= c["win_rate"] <= c["ci_high"]


def test_gold_curve_bins(insights):
    centers = [b["gold_bin_center"] for b in insights["gold_curve"]]
    assert centers == sorted(centers) and set(centers) <= set(range(-8000, 8001, 1000))
    assert all(b["n"] > 0 and 0 <= b["win_rate"] <= 1 for b in insights["gold_curve"])


def test_objectives_and_blue_side(insights):
    terms = {o["term"] for o in insights["objectives"]["terms"]}
    assert terms == {"gold_k", "dragons_diff", "towers_diff", "grubs_diff", "kills_diff"}
    for o in insights["objectives"]["terms"]:
        assert o["ci_low"] <= o["odds_ratio"] <= o["ci_high"] and 0 <= o["p_value"] <= 1
    assert set(insights["objectives"]["dragon_vs_tower"]) == {"estimate_diff", "p_value"}
    assert set(insights["blue_side"]) == {"win_rate", "n", "p_value"}
    assert "draft_lr" in insights["model_comparison"]


def test_synthetic_gold_lead_raises_win_rate(synthetic_dirs):
    ins = json.load(open(os.path.join(synthetic_dirs.artifacts_dir, "insights.json")))
    gold = next(o for o in ins["objectives"]["terms"] if o["term"] == "gold_k")
    assert gold["odds_ratio"] > 1


def test_objectives_skips_constant_terms():
    # A term that never varies in the data (here grubs) must be left out, not crash the fit.
    import numpy as np
    import pandas as pd

    from lolwp.models.insights import objectives

    rng = np.random.default_rng(0)
    n = 400
    gold = rng.normal(0, 3000, n)
    df = pd.DataFrame({
        "gold_diff": gold,
        "dragons_diff": rng.integers(-2, 3, n),
        "towers_diff": rng.integers(-2, 3, n),
        "grubs_diff": np.zeros(n, dtype=int),
        "kills_diff": rng.integers(-5, 6, n),
        "blue_win": (gold + rng.normal(0, 3000, n) > 0).astype(int),
    })
    terms = {o["term"] for o in objectives(df)["terms"]}
    assert terms == {"gold_k", "dragons_diff", "towers_diff", "kills_diff"}


def test_insights_carry_calibration_and_data(pipeline_dirs):
    from lolwp.models import insights as ins_mod

    assert ins_mod.MIN_CHAMP_GAMES == 10
    ins = json.load(open(os.path.join(pipeline_dirs.artifacts_dir, "insights.json"), encoding="utf-8"))
    assert set(ins["calibration"]) == {"draft_lr", "state_lr", "combined_lr", "state_lr_m10", "combined_lr_m10"}
    assert ins["data"]["test_patch"] and ins["data"]["n_games"] > 0
