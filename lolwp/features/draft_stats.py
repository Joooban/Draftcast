"""Smoothed champion, lane-matchup and bot-duo win rates, with out-of-fold fitting.

A match's own result must never feed the stats that describe it, so training rows get
stats fitted on the other folds (target encoding without leakage).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from lolwp.config import OOF_FOLDS, PRIOR_GAMES, ROLES, SEED
from lolwp.features.parse import MatchRecord


def smooth(wins: float, games: float, prior: int = PRIOR_GAMES) -> float:
    return (wins + prior * 0.5) / (games + prior)


def _key(*parts) -> str:
    return "|".join(str(p) for p in parts)


class DraftStats:
    def __init__(self, champ: dict, lane: dict, duo: dict, match_ids: set[str]):
        # each table maps a string key -> [wins, games]
        self.champ = champ
        self.lane = lane
        self.duo = duo
        self.match_ids = match_ids

    @classmethod
    def fit(cls, records: list[MatchRecord]) -> "DraftStats":
        champ = defaultdict(lambda: [0, 0])
        lane = defaultdict(lambda: [0, 0])
        duo = defaultdict(lambda: [0, 0])
        for r in records:
            for team, enemy, won in ((r.blue, r.red, r.blue_win), (r.red, r.blue, 1 - r.blue_win)):
                for role in ROLES:
                    for table, key in ((champ, _key(team[role])), (lane, _key(role, team[role], enemy[role]))):
                        table[key][0] += won
                        table[key][1] += 1
                d = duo[_key(team["bot"], team["support"])]
                d[0] += won
                d[1] += 1
        return cls(dict(champ), dict(lane), dict(duo), {r.match_id for r in records})

    @staticmethod
    def _rate(table: dict, key: str) -> float:
        wins, games = table.get(key, (0, 0))
        return smooth(wins, games)

    def champ_rate(self, champ: int) -> float:
        return self._rate(self.champ, _key(champ))

    def lane_rate(self, role: str, champ: int, enemy: int) -> float:
        return self._rate(self.lane, _key(role, champ, enemy))

    def duo_rate(self, bot: int, support: int) -> float:
        return self._rate(self.duo, _key(bot, support))

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {"champ": self.champ, "lane": self.lane, "duo": self.duo, "match_ids": sorted(self.match_ids)}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str) -> "DraftStats":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["champ"], d["lane"], d["duo"], set(d["match_ids"]))


def oof_stats(records: list[MatchRecord], k: int = OOF_FOLDS, seed: int = SEED) -> dict[str, DraftStats]:
    out: dict[str, DraftStats] = {}
    idx = np.arange(len(records))
    for train_idx, fold_idx in KFold(n_splits=k, shuffle=True, random_state=seed).split(idx):
        stats = DraftStats.fit([records[i] for i in train_idx])
        for i in fold_idx:
            out[records[i].match_id] = stats
    return out
