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
    assert 0 <= body["probability"] <= 1 and len(body["reasons"]) == 3 and body["split"] is None
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


def test_state_missing_feature_422(client):
    state = dict(STATE)
    del state["towers_diff"]
    r = client.post("/api/predict", json={"mode": "state", "state": state})
    assert r.status_code == 422 and "towers_diff" in r.json()["detail"]


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
