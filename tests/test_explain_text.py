from lolwp.api.explain_text import reason_text, strength, verdict


def test_strength_boundaries():
    assert strength(0.549) == "toss-up"
    assert strength(0.55) == "slight edge"
    assert strength(0.30) == "favored"
    assert strength(0.81) == "strong favorite"


def test_state_reasons_name_side_and_value():
    assert reason_text("gold_diff", -2300) == "Red leads by 2,300 gold"
    assert reason_text("dragons_diff", 1) == "Blue has 1 more dragon"
    assert reason_text("dragons_diff", -2) == "Red has 2 more dragons"
    assert reason_text("level_diff", 3) == "Blue is 3 levels ahead in total"
    assert reason_text("level_diff", -1) == "Red is 1 level ahead in total"
    assert reason_text("first_blood_blue", -1) == "Red took first blood"
    assert reason_text("towers_diff", 1) == "Blue has taken 1 more tower"


def test_zero_value_has_no_reason():
    assert reason_text("kills_diff", 0) is None
    assert reason_text("first_dragon_blue", 0) is None
    assert reason_text("champ_wr_diff", 0.0) is None


def test_draft_reasons():
    assert reason_text("lane_mid", 0.04) == "Blue's mid matchup has won 54% of games"
    assert reason_text("lane_top", -0.05) == "Red's top matchup has won 55% of games"
    assert reason_text("draft_logit", 0.2) == "The draft leans Blue"


def test_verdict():
    assert verdict(0.68, "Blue leads by 2,300 gold") == "Blue is favored: 68%, mostly because Blue leads by 2,300 gold"
    assert verdict(0.52, "Blue leads by 100 gold") == "Toss-up: Blue 52%"
    assert verdict(0.40, None) == "Red has a slight edge: 60%"
    assert verdict(0.10, "The draft leans Red") == "Red is a strong favorite: 90%, mostly because the draft leans Red"
