"""SQLite storage for raw match JSON, with progress flags so collection can resume."""
import json
import random
import sqlite3
import zlib
from collections.abc import Iterator
from pathlib import Path

from lolwp.config import MIN_DURATION_S, STATE_MINUTE

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    puuid TEXT PRIMARY KEY, tier TEXT, division TEXT, done INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS match_ids (
    match_id TEXT PRIMARY KEY, details_fetched INTEGER DEFAULT 0, timeline_fetched INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY, details BLOB, timeline BLOB, duration INTEGER);
"""


def _pack(obj: dict) -> bytes:
    return zlib.compress(json.dumps(obj, separators=(",", ":")).encode())


def _unpack(blob: bytes | None) -> dict | None:
    return None if blob is None else json.loads(zlib.decompress(blob))


class Store:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)

    def _insert_ignore(self, sql: str, rows: list[tuple]) -> int:
        before = self.conn.total_changes
        with self.conn:
            self.conn.executemany(sql, rows)
        return self.conn.total_changes - before

    def add_players(self, rows: list[tuple[str, str, str]]) -> int:
        return self._insert_ignore("INSERT OR IGNORE INTO players (puuid, tier, division) VALUES (?, ?, ?)", rows)

    def players_todo(self, limit: int) -> list[str]:
        cur = self.conn.execute("SELECT puuid FROM players WHERE done = 0 ORDER BY rowid LIMIT ?", (limit,))
        return [r[0] for r in cur]

    def mark_player_done(self, puuid: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE players SET done = 1 WHERE puuid = ?", (puuid,))

    def add_match_ids(self, ids: list[str]) -> int:
        return self._insert_ignore("INSERT OR IGNORE INTO match_ids (match_id) VALUES (?)", [(i,) for i in ids])

    def details_todo(self, limit: int) -> list[str]:
        cur = self.conn.execute(
            "SELECT match_id FROM match_ids WHERE details_fetched = 0 ORDER BY rowid LIMIT ?", (limit,))
        return [r[0] for r in cur]

    def save_details(self, match_id: str, match: dict | None) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO match_ids (match_id) VALUES (?)", (match_id,))
            self.conn.execute("UPDATE match_ids SET details_fetched = 1 WHERE match_id = ?", (match_id,))
            if match is not None:
                duration = int(match.get("info", {}).get("gameDuration", 0))
                self.conn.execute(
                    "INSERT OR REPLACE INTO matches (match_id, details, timeline, duration) VALUES (?, ?, NULL, ?)",
                    (match_id, _pack(match), duration))

    def timelines_todo(self, limit: int, seed: int) -> list[str]:
        cur = self.conn.execute(
            "SELECT m.match_id FROM matches m JOIN match_ids i USING (match_id) "
            "WHERE m.duration >= ? AND i.timeline_fetched = 0 ORDER BY m.match_id",
            (MIN_DURATION_S,))
        ids = [r[0] for r in cur]
        if len(ids) <= limit:
            return ids
        return random.Random(seed).sample(ids, limit)

    def save_timeline(self, match_id: str, timeline: dict | None) -> None:
        with self.conn:
            self.conn.execute("UPDATE match_ids SET timeline_fetched = 1 WHERE match_id = ?", (match_id,))
            if timeline is not None:
                info = timeline.setdefault("info", {})
                info["frames"] = info.get("frames", [])[: STATE_MINUTE + 1]
                self.conn.execute("UPDATE matches SET timeline = ? WHERE match_id = ?", (_pack(timeline), match_id))

    def iter_matches(self) -> Iterator[tuple[str, dict, dict | None]]:
        cur = self.conn.execute("SELECT match_id, details, timeline FROM matches ORDER BY match_id")
        for match_id, details, timeline in cur:
            yield match_id, _unpack(details), _unpack(timeline)

    def counts(self) -> dict[str, int]:
        q = {
            "players": "SELECT COUNT(*) FROM players",
            "match_ids": "SELECT COUNT(*) FROM match_ids",
            "details": "SELECT COUNT(*) FROM matches",
            "timelines": "SELECT COUNT(*) FROM matches WHERE timeline IS NOT NULL",
        }
        return {k: self.conn.execute(sql).fetchone()[0] for k, sql in q.items()}
