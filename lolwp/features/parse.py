"""Match details JSON → one clean MatchRecord, or None when the match is unusable."""
from dataclasses import dataclass

from lolwp.config import QUEUE_ID, RIOT_POSITION_TO_ROLE, ROLES

BLUE, RED = 100, 200


@dataclass
class MatchRecord:
    match_id: str
    patch: str
    blue_win: int
    duration_s: int
    blue: dict[str, int]
    red: dict[str, int]


def patch_key(patch: str) -> tuple[int, int]:
    major, minor = patch.split(".")[:2]
    return int(major), int(minor)


def parse_match(match: dict, platform: str | None = None) -> MatchRecord | None:
    """`platform` (e.g. "sg2") drops matches played on other servers by the same players."""
    try:
        if platform and not match["metadata"]["matchId"].upper().startswith(platform.upper() + "_"):
            return None
        info = match["info"]
        if info.get("queueId") != QUEUE_ID:
            return None
        participants = info["participants"]
        if len(participants) != 10:
            return None
        if any(p.get("gameEndedInEarlySurrender") for p in participants):
            return None

        teams: dict[int, dict[str, int]] = {BLUE: {}, RED: {}}
        for p in participants:
            role = RIOT_POSITION_TO_ROLE.get(p.get("teamPosition") or "")
            side = teams.get(p.get("teamId"))
            if role is None or side is None or role in side:
                return None
            side[role] = int(p["championId"])
        if any(set(side) != set(ROLES) for side in teams.values()):
            return None

        blue_team = next(t for t in info["teams"] if t["teamId"] == BLUE)
        major, minor = info["gameVersion"].split(".")[:2]
        return MatchRecord(
            match_id=match["metadata"]["matchId"],
            patch=f"{major}.{minor}",
            blue_win=int(bool(blue_team["win"])),
            duration_s=int(info["gameDuration"]),
            blue={r: teams[BLUE][r] for r in ROLES},
            red={r: teams[RED][r] for r in ROLES},
        )
    except (KeyError, TypeError, ValueError, StopIteration):
        return None
