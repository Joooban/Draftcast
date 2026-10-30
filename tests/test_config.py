from lolwp import config


def test_roles_and_mapping_align():
    assert set(config.RIOT_POSITION_TO_ROLE.values()) == set(config.ROLES)


def test_every_state_feature_has_range():
    assert set(config.STATE_FEATURES) == set(config.STATE_RANGES)
    assert len(config.STATE_FEATURES) == 15


def test_load_settings_defaults(monkeypatch):
    monkeypatch.setenv("RIOT_API_KEY", "RGAPI-test")
    monkeypatch.delenv("PLATFORM", raising=False)
    s = config.load_settings()
    assert s.api_key == "RGAPI-test" and s.platform == "sg2" and s.region == "sea"
