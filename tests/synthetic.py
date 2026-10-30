"""Synthetic Riot-shaped match and timeline JSON, for exercising the pipeline without real data.

Wins depend on a hidden champion strength and on the 15-minute gold lead, so trained
models have real signal to find. Nothing here is ever written to artifacts/.
"""
import json
import math
import random
from pathlib import Path

from lolwp.riot.store import Store

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
TAG_POOL = [["Fighter", "Tank"], ["Mage"], ["Mage", "Assassin"], ["Marksman"], ["Support", "Tank"],
            ["Assassin"], ["Fighter"], ["Support", "Mage"]]
N_CHAMPS = 40


def write_champions(path: str, version: str = "15.21.1") -> dict[int, float]:
    """Write a Data Dragon-shaped champion.json; return hidden strength per champion id."""
    rng = random.Random(7)
    data, strength = {}, {}
    for i in range(N_CHAMPS):
        cid = 1000 + i
        name = f"Champ{i}"
        data[name] = {"id": name, "key": str(cid), "name": name, "tags": TAG_POOL[i % len(TAG_POOL)],
                      "info": {"attack": rng.randint(1, 10), "defense": 5, "magic": rng.randint(1, 10),
                               "difficulty": 5}}
        strength[cid] = rng.gauss(0, 0.35)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"type": "champion", "version": version, "data": data}))
    return strength


def _match_and_timeline(rng: random.Random, match_id: str, patch: str, strength: dict[int, float],
                        duration: int, blank_position: bool = False):
    picks = rng.sample(sorted(strength), 10)
    draft = sum(strength[c] for c in picks[:5]) - sum(strength[c] for c in picks[5:])
    gold_diff = int(rng.gauss(1500 * draft, 2500))
    logit = 0.00035 * gold_diff + 0.6 * draft
    blue_win = rng.random() < 1 / (1 + math.exp(-logit))

    participants = []
    for i, cid in enumerate(picks):
        team = 100 if i < 5 else 200
        participants.append({"participantId": i + 1, "teamId": team, "teamPosition": POSITIONS[i % 5],
                             "championId": cid, "win": blue_win == (team == 100),
                             "gameEndedInEarlySurrender": False})
    if blank_position:
        participants[rng.randrange(10)]["teamPosition"] = ""
    match = {"metadata": {"matchId": match_id},
             "info": {"queueId": 420, "gameDuration": duration, "gameVersion": f"{patch}.500.1234",
                      "participants": participants,
                      "teams": [{"teamId": 100, "win": blue_win}, {"teamId": 200, "win": not blue_win}]}}

    lead = gold_diff / 10000  # tilts objective odds toward the team ahead
    frames = []
    for f in range(16):
        pf = {}
        for pid in range(1, 11):
            s = 1 if pid <= 5 else -1
            share = s * gold_diff / 10 * f / 15
            pf[str(pid)] = {"participantId": pid, "totalGold": int(500 + 330 * f + share),
                            "xp": int(400 * f + share * 0.8), "minionsKilled": int(6.5 * f + share / 60),
                            "jungleMinionsKilled": f, "level": 1 + f * 8 // 15}
        frames.append({"timestamp": f * 60000, "participantFrames": pf, "events": []})

    def add(kind, frame, **kw):
        frames[frame]["events"].append({"type": kind, "timestamp": frame * 60000 - rng.randint(1, 59000), **kw})

    def blue_gets(p):
        return rng.random() < min(0.95, max(0.05, p + lead))

    for _ in range(rng.randint(4, 16)):
        victim = rng.randint(6, 10) if blue_gets(0.5) else rng.randint(1, 5)
        add("CHAMPION_KILL", rng.randint(2, 15), killerId=0, victimId=victim)
    for frame in (6, 11):
        add("ELITE_MONSTER_KILL", frame, killerTeamId=100 if blue_gets(0.5) else 200, monsterType="DRAGON")
    for frame in (7, 8):
        add("ELITE_MONSTER_KILL", frame, killerTeamId=100 if blue_gets(0.5) else 200, monsterType="HORDE")
    add("ELITE_MONSTER_KILL", 15, killerTeamId=100 if blue_gets(0.5) else 200, monsterType="RIFTHERALD")
    for _ in range(rng.randint(0, 3)):
        add("BUILDING_KILL", rng.randint(10, 15), buildingType="TOWER_BUILDING",
            teamId=200 if blue_gets(0.5) else 100)
    for _ in range(rng.randint(2, 8)):
        add("TURRET_PLATE_DESTROYED", rng.randint(5, 14), teamId=200 if blue_gets(0.5) else 100)
    for _ in range(rng.randint(20, 40)):
        add("WARD_PLACED", rng.randint(1, 15), creatorId=rng.randint(1, 10))
    for _ in range(rng.randint(2, 10)):
        add("WARD_KILL", rng.randint(3, 15), killerId=rng.randint(1, 10))
    timeline = {"metadata": {"matchId": match_id}, "info": {"frames": frames}}
    return match, timeline


def fill_store(db_path: str, champions_path: str, patches=("15.19", "15.20", "15.21"),
               per_patch: int = 200, seed: int = 0) -> Store:
    """Fill a store with synthetic matches; a few are short, invalid, or lack timelines."""
    rng = random.Random(seed)
    strength = write_champions(champions_path)
    store = Store(db_path)
    n = 0
    for patch in patches:
        for _ in range(per_patch):
            n += 1
            mid = f"SG2_{n:06d}"
            duration = 900 if n % 25 == 0 else rng.randint(1300, 2400)
            match, timeline = _match_and_timeline(rng, mid, patch, strength, duration, blank_position=n % 40 == 0)
            store.add_match_ids([mid])
            store.save_details(mid, match)
            if n % 10 != 0:
                store.save_timeline(mid, timeline)
    return store
