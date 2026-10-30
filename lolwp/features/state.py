"""Timeline → game-state features at a given minute, all Blue minus Red."""
from lolwp.config import STATE_FEATURES, STATE_MINUTE

BLUE, RED = 100, 200


def _sign(team: int | None) -> int:
    return 1 if team == BLUE else -1 if team == RED else 0


def state_features(match: dict, timeline: dict, minute: int = STATE_MINUTE) -> dict[str, int] | None:
    frames = timeline.get("info", {}).get("frames", [])
    if len(frames) < minute + 1:
        return None
    team_of = {p["participantId"]: p["teamId"] for p in match["info"]["participants"]}
    cutoff = minute * 60000

    f = {k: 0 for k in STATE_FEATURES}
    for pid_str, pf in frames[minute]["participantFrames"].items():
        s = _sign(team_of.get(int(pid_str)))
        f["gold_diff"] += s * pf["totalGold"]
        f["xp_diff"] += s * pf["xp"]
        f["cs_diff"] += s * (pf["minionsKilled"] + pf["jungleMinionsKilled"])
        f["level_diff"] += s * pf["level"]

    for frame in frames[: minute + 1]:
        for e in frame.get("events", []):
            if e.get("timestamp", 0) > cutoff:
                continue
            kind = e.get("type")
            if kind == "CHAMPION_KILL":
                # Credit the victim's opponents, so executions (killerId 0) still count.
                s = -_sign(team_of.get(e.get("victimId")))
                f["kills_diff"] += s
                if not f["first_blood_blue"]:
                    f["first_blood_blue"] = s
            elif kind == "ELITE_MONSTER_KILL":
                s = _sign(e.get("killerTeamId"))
                monster = e.get("monsterType")
                if monster == "DRAGON":
                    f["dragons_diff"] += s
                    if not f["first_dragon_blue"]:
                        f["first_dragon_blue"] = s
                elif monster == "HORDE":
                    f["grubs_diff"] += s
                elif monster == "RIFTHERALD":
                    f["heralds_diff"] += s
            elif kind == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
                s = -_sign(e.get("teamId"))  # teamId is the team that lost the tower
                f["towers_diff"] += s
                if not f["first_tower_blue"]:
                    f["first_tower_blue"] = s
            elif kind == "TURRET_PLATE_DESTROYED":
                f["plates_diff"] -= _sign(e.get("teamId"))
            elif kind == "WARD_PLACED":
                f["wards_placed_diff"] += _sign(team_of.get(e.get("creatorId")))
            elif kind == "WARD_KILL":
                f["wards_killed_diff"] += _sign(team_of.get(e.get("killerId")))
    return f
