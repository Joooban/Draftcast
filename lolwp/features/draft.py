"""Picks → draft features, all Blue minus Red."""
from lolwp.champions import TAGS, Champion
from lolwp.config import ROLES
from lolwp.features.draft_stats import DraftStats

DRAFT_FEATURES: list[str] = (
    ["champ_wr_diff"]
    + [f"lane_{r}" for r in ROLES]
    + ["duo_diff"]
    + [f"tag_{t}_diff" for t in TAGS]
    + ["attack_diff", "magic_diff"]
)


def _comp(team: dict[str, int], champs: dict[int, Champion]) -> dict[str, float]:
    out = {f"tag_{t}_diff": 0.0 for t in TAGS} | {"attack_diff": 0.0, "magic_diff": 0.0}
    for cid in team.values():
        c = champs.get(cid)
        if c is None:
            continue
        for t in c.tags:
            if t in TAGS:
                out[f"tag_{t}_diff"] += 1
        out["attack_diff"] += c.attack
        out["magic_diff"] += c.magic
    return out


def draft_features(blue: dict[str, int], red: dict[str, int], stats: DraftStats,
                   champs: dict[int, Champion]) -> dict[str, float]:
    f: dict[str, float] = {}
    mean = lambda team: sum(stats.champ_rate(team[r]) for r in ROLES) / len(ROLES)
    f["champ_wr_diff"] = mean(blue) - mean(red)
    for r in ROLES:
        f[f"lane_{r}"] = stats.lane_rate(r, blue[r], red[r]) - 0.5
    f["duo_diff"] = stats.duo_rate(blue["bot"], blue["support"]) - stats.duo_rate(red["bot"], red["support"])
    b, rd = _comp(blue, champs), _comp(red, champs)
    for k in b:
        f[k] = b[k] - rd[k]
    return {k: f[k] for k in DRAFT_FEATURES}
