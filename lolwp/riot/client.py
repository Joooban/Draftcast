"""Rate-limited GET client for the Riot API."""
import time

import requests

from lolwp.config import MIN_INTERVAL_S

MAX_ATTEMPTS = 5


class KeyExpiredError(Exception):
    """Raised on 401/403: the dev key is missing, wrong, or expired."""


class RiotClient:
    def __init__(self, api_key: str, session=None, sleep=time.sleep, clock=time.monotonic,
                 min_interval: float = MIN_INTERVAL_S):
        self.session = session if session is not None else requests.Session()
        self.session.headers["X-Riot-Token"] = api_key
        self.sleep = sleep
        self.clock = clock
        self.min_interval = min_interval
        self._last: float | None = None

    def _wait_turn(self) -> None:
        if self._last is not None:
            wait = self.min_interval - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
        self._last = self.clock()

    def get(self, url: str, params: dict | None = None) -> dict | list | None:
        attempt = 0
        while attempt < MAX_ATTEMPTS:
            self._wait_turn()
            resp = self.session.get(url, params=params, timeout=30)
            code = resp.status_code
            if code == 200:
                return resp.json()
            if code == 404:
                return None
            if code in (401, 403):
                raise KeyExpiredError(f"HTTP {code} for {url}")
            if code == 429:
                self.sleep(int(resp.headers.get("Retry-After", 10)))
                continue
            attempt += 1
            if code >= 500 and attempt < MAX_ATTEMPTS:
                self.sleep(5 * attempt)
                continue
            if code < 500:
                return None
        return None
