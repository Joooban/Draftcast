import pytest

from lolwp.riot.client import KeyExpiredError, RiotClient


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


class Recorder:
    def __init__(self):
        self.sleeps = []

    def __call__(self, s):
        self.sleeps.append(s)


def make(responses):
    session, sleep = FakeSession(responses), Recorder()
    client = RiotClient("RGAPI-x", session=session, sleep=sleep, clock=lambda: 0.0)
    return client, session, sleep


def test_returns_json_on_200():
    client, _, _ = make([FakeResponse(200, {"ok": 1})])
    assert client.get("u") == {"ok": 1}


def test_404_returns_none():
    client, _, _ = make([FakeResponse(404)])
    assert client.get("u") is None


def test_403_raises_key_expired():
    client, _, _ = make([FakeResponse(403)])
    with pytest.raises(KeyExpiredError):
        client.get("u")


def test_429_sleeps_retry_after_then_succeeds():
    client, _, sleep = make([FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {"ok": 1})])
    assert client.get("u") == {"ok": 1}
    assert 7 in sleep.sleeps


def test_500_retries_then_gives_up_after_5():
    client, session, _ = make([FakeResponse(500) for _ in range(5)])
    assert client.get("u") is None
    assert len(session.calls) == 5


def test_spacing_between_calls():
    client, _, sleep = make([FakeResponse(200, {}), FakeResponse(200, {})])
    client.get("a")
    assert sleep.sleeps == []
    client.get("b")
    assert sleep.sleeps == [pytest.approx(1.25)]


def test_sends_token_header():
    _, session, _ = make([])
    assert session.headers["X-Riot-Token"] == "RGAPI-x"
