import pytest
from fastapi.testclient import TestClient

from lolwp.api.main import create_app
from lolwp.config import ROLES, STATE_FEATURES


@pytest.fixture(scope="module")
def client(pipeline_dirs):
    return TestClient(create_app(pipeline_dirs.artifacts_dir))


@pytest.fixture(scope="module")
def ids(client):
    return [c["id"] for c in client.get("/api/champions").json()][:10]


def teams(ids):
    return dict(zip(ROLES, ids[:5])), dict(zip(ROLES, ids[5:10]))


STATE = {k: 0 for k in STATE_FEATURES} | {"gold_diff": 2500, "dragons_diff": 1, "first_blood_blue": 1}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["models_loaded"] is True


def test_champions_list(client):
    r = client.get("/api/champions")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) >= 10
    assert all({"id", "name", "icon_url", "tags"} <= set(c) for c in body)


def test_predict_draft_ok(client, ids):
    blue, red = teams(ids)
    r = client.post("/api/predict", json={"mode": "draft", "blue": blue, "red": red})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability"] <= 1 and len(body["reasons"]) <= 5 and body["split"] is None
    assert all(set(r) == {"text", "value_pct"} for r in body["reasons"])
    assert "coin flip" in body["reliability"]["text"]
    total = body["baseline"] + sum(c["value_pct"] for c in body["contributions"]) / 100
    assert total == pytest.approx(body["probability"], abs=1e-9)
    assert all(c["label"] for c in body["contributions"])


def test_predict_state_ok(client):
    r = client.post("/api/predict", json={"mode": "state", "state": STATE})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["probability"] > body["baseline"]  # big gold lead favours Blue
    assert {c["feature"] for c in body["contributions"]} == set(STATE_FEATURES)


def test_predict_both_has_split(client, ids):
    blue, red = teams(ids)
    r = client.post("/api/predict", json={"mode": "both", "blue": blue, "red": red, "state": STATE})
    assert r.status_code == 200, r.text
    body = r.json()
    split = body["split"]
    assert split["draft_pct"] + split["state_pct"] == pytest.approx((body["probability"] - body["baseline"]) * 100)
    assert "draft_logit" in {c["feature"] for c in body["contributions"]}


def test_draft_missing_team_422(client, ids):
    blue, _ = teams(ids)
    r = client.post("/api/predict", json={"mode": "draft", "blue": blue})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)


def test_both_with_partial_draft_422(client, ids):  # Review Focus 4
    blue, _ = teams(ids)
    r = client.post("/api/predict", json={"mode": "both", "blue": blue, "state": STATE})
    assert r.status_code == 422 and "draft" in r.json()["detail"].lower()


def test_both_without_state_422(client, ids):
    blue, red = teams(ids)
    r = client.post("/api/predict", json={"mode": "both", "blue": blue, "red": red})
    assert r.status_code == 422


def test_duplicate_champion_422(client, ids):
    blue, red = teams(ids)
    red["mid"] = blue["mid"]
    r = client.post("/api/predict", json={"mode": "draft", "blue": blue, "red": red})
    assert r.status_code == 422 and "twice" in r.json()["detail"]


def test_unknown_champion_422(client, ids):
    blue, red = teams(ids)
    blue["top"] = 999999
    r = client.post("/api/predict", json={"mode": "draft", "blue": blue, "red": red})
    assert r.status_code == 422 and "999999" in r.json()["detail"]


def test_state_out_of_range_422(client):  # Review Focus 5
    r = client.post("/api/predict", json={"mode": "state", "state": STATE | {"gold_diff": 999999}})
    assert r.status_code == 422 and "gold_diff" in r.json()["detail"]


def test_state_partial_ok(client):
    state = {"gold_diff": 2500, "kills_diff": 3, "cs_diff": None}
    r = client.post("/api/predict", json={"mode": "state", "state": state})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["omitted"] == sorted(set(STATE_FEATURES) - {"gold_diff", "kills_diff"})
    assert {c["feature"] for c in body["contributions"]} == {"gold_diff", "kills_diff"}
    assert body["probability"] > 0.5 and "leaves out 11" in body["reliability"]["text"]
    total = body["baseline"] + sum(c["value_pct"] for c in body["contributions"]) / 100
    assert total == pytest.approx(body["probability"], abs=1e-9)


def test_state_all_blank_422(client):  # Review Focus 1
    r = client.post("/api/predict", json={"mode": "state", "state": {k: None for k in STATE_FEATURES}})
    assert r.status_code == 422 and "at least one" in r.json()["detail"]


def test_both_partial_state_split_exact(client, ids):  # Review Focus 2
    blue, red = teams(ids)
    r = client.post("/api/predict", json={"mode": "both", "blue": blue, "red": red, "state": {"gold_diff": 1500}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {c["feature"] for c in body["contributions"]} == {"draft_logit", "gold_diff"}
    total = body["baseline"] + sum(c["value_pct"] for c in body["contributions"]) / 100
    assert total == pytest.approx(body["probability"], abs=1e-9)
    split = body["split"]
    assert split["draft_pct"] + split["state_pct"] == pytest.approx((body["probability"] - body["baseline"]) * 100)


def test_bool_rejected_422(client):
    r = client.post("/api/predict", json={"mode": "state", "state": {"gold_diff": True}})
    assert r.status_code == 422


def test_reasons_skip_zero_values(client):
    state = {"gold_diff": 2500, "kills_diff": 0, "towers_diff": 0}
    texts = [x["text"].lower() for x in client.post("/api/predict", json={"mode": "state", "state": state}).json()["reasons"]]
    assert texts and not any("kill" in t or "tower" in t for t in texts)


def test_response_has_layers(client):
    body = client.post("/api/predict", json={"mode": "state", "state": STATE}).json()
    assert body["verdict"] and body["strength"] in {"toss-up", "slight edge", "favored", "strong favorite"}
    cal = body["calibration"]
    assert cal["lo"] <= body["probability"] <= cal["hi"] and cal["n"] > 0
    assert body["reliability"]["text"] and body["omitted"] == [] and body["minute"] == 15
    gold = next(c for c in body["contributions"] if c["feature"] == "gold_diff")
    assert gold["value"] == 2500 and isinstance(gold["logit"], float)


def test_minute_10_request(client):
    body = client.post("/api/predict", json={"mode": "state", "minute": 10, "state": {"gold_diff": 1000}}).json()
    assert body["minute"] == 10 and 0 <= body["probability"] <= 1


def test_bad_minute_422(client):
    assert client.post("/api/predict", json={"mode": "state", "minute": 20, "state": STATE}).status_code == 422


def test_meta(client):
    body = client.get("/api/meta").json()
    assert body["test_patch"] and body["n_games"] > 0 and body["minutes"] == [10, 15]


def test_unseen_champion_ok(synthetic_dirs, tmp_path):
    import json
    import shutil

    from lolwp.features.draft_stats import DraftStats

    art = tmp_path / "art"
    shutil.copytree(synthetic_dirs.artifacts_dir, art)
    data = json.loads((art / "champions.json").read_text(encoding="utf-8"))
    template = next(iter(data["data"].values()))
    data["data"]["Newcomer"] = template | {"id": "Newcomer", "key": "9999", "name": "Newcomer"}
    (art / "champions.json").write_text(json.dumps(data), encoding="utf-8")
    assert DraftStats.from_json(str(art / "draft_stats.json")).champ_rate(9999) == 0.5

    c = TestClient(create_app(str(art)))
    ids = [ch["id"] for ch in c.get("/api/champions").json() if ch["id"] != 9999][:9]
    blue, red = dict(zip(ROLES, [9999, *ids[:4]])), dict(zip(ROLES, ids[4:9]))
    r = c.post("/api/predict", json={"mode": "draft", "blue": blue, "red": red})
    assert r.status_code == 200, r.text
    assert 0 <= r.json()["probability"] <= 1


def test_state_unknown_feature_422(client):
    r = client.post("/api/predict", json={"mode": "state", "state": STATE | {"barons_diff": 1}})
    assert r.status_code == 422


def test_bad_mode_has_readable_detail(client):
    r = client.post("/api/predict", json={"mode": "lol"})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)


def test_insights(client):
    r = client.get("/api/insights")
    assert r.status_code == 200 and "champions" in r.json()


def test_missing_artifacts_returns_503(tmp_path):
    c = TestClient(create_app(str(tmp_path)))
    assert c.get("/api/health").json()["models_loaded"] is False
    assert c.get("/api/champions").status_code == 503


@pytest.fixture
def web_client(tmp_path, synthetic_dirs):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id=root></div>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (dist / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("nope")
    return TestClient(create_app(synthetic_dirs.artifacts_dir, web_dir=str(dist)))


def test_root_serves_index(web_client):
    r = web_client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "id=root" in r.text


def test_spa_fallback_for_client_routes(web_client):
    r = web_client.get("/insights")
    assert r.status_code == 200 and "id=root" in r.text


def test_static_files_served(web_client):
    assert web_client.get("/assets/app.js").text == "console.log(1)"
    assert web_client.get("/favicon.svg").text == "<svg/>"


def test_api_still_json_with_frontend(web_client):
    r = web_client.get("/api/health")
    assert r.headers["content-type"].startswith("application/json")
    assert web_client.get("/api/nope").status_code == 404


def test_no_path_traversal(web_client):
    r = web_client.get("/..%2Fsecret.txt")
    assert "nope" not in r.text


def test_api_does_not_import_training(synthetic_dirs):
    import subprocess
    import sys

    code = ("import sys; from lolwp.api.main import create_app; create_app(sys.argv[1]); "
            "print(any(m in sys.modules for m in ('lightgbm', 'statsmodels', 'lolwp.models.train')))")
    out = subprocess.run([sys.executable, "-c", code, synthetic_dirs.artifacts_dir],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
