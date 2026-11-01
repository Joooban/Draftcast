from lolwp import config


def test_roles_and_mapping_align():
    assert set(config.RIOT_POSITION_TO_ROLE.values()) == set(config.ROLES)


def test_every_state_feature_has_range():
    assert set(config.STATE_FEATURES) == set(config.STATE_RANGES)
    assert len(config.STATE_FEATURES) == 13


def test_xp_not_a_state_feature():
    # Players can't see XP on the scoreboard; total level carries the same signal.
    assert "xp_diff" not in config.STATE_FEATURES


def test_state_minutes():
    assert config.STATE_MINUTES == (10, 15)


def test_herald_not_a_state_feature():
    # Rift Herald spawns after 15:00 in current patches, so it is always 0 at the cutoff.
    assert "heralds_diff" not in config.STATE_FEATURES


def test_plates_range_covers_inner_turret_plates():
    # Inner turrets carry plates too; real SG2 games reach about ±35 at 15:00.
    assert config.STATE_RANGES["plates_diff"] == (-30, 30)


def test_load_settings_defaults(monkeypatch):
    monkeypatch.setenv("RIOT_API_KEY", "RGAPI-test")
    monkeypatch.delenv("PLATFORM", raising=False)
    s = config.load_settings()
    assert s.api_key == "RGAPI-test" and s.platform == "sg2" and s.region == "sea"
