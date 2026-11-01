"""Plain-language wording for predictions: strength words, one sentence per factor, the verdict line.

Every sentence names the side a factor favours and the real number behind it. A factor whose
value is 0 (even, or no difference) gets no sentence, so nothing is credited for being level.
"""
from lolwp.api.schemas import Reason
from lolwp.config import ROLES

MAX_REASONS = 5

# feature -> (verb phrase, singular noun, plural noun); "{S} {verb} {n} {noun}"
_COUNTS = {
    "gold_diff": ("leads by", "gold", "gold"),
    "cs_diff": ("leads by", "CS", "CS"),
    "kills_diff": ("has", "more kill", "more kills"),
    "dragons_diff": ("has", "more dragon", "more dragons"),
    "grubs_diff": ("has", "more voidgrub", "more voidgrubs"),
    "towers_diff": ("has taken", "more tower", "more towers"),
    "plates_diff": ("has taken", "more turret plate", "more turret plates"),
    "wards_placed_diff": ("has placed", "more ward", "more wards"),
    "wards_killed_diff": ("has cleared", "more ward", "more wards"),
}
_FIRSTS = {"first_blood_blue": "first blood", "first_dragon_blue": "first dragon", "first_tower_blue": "first tower"}
_TAGS = {"Tank": ("tank", "tanks"), "Fighter": ("fighter", "fighters"), "Mage": ("mage", "mages"),
         "Assassin": ("assassin", "assassins"), "Marksman": ("marksman", "marksmen"),
         "Support": ("support", "supports")}
_ROLE_WORDS = {"top": "top", "jungle": "jungle", "mid": "mid", "bot": "bot", "support": "support"}


def strength(p: float, draft_only: bool = False) -> str:
    """Draft-only predictions are near a coin flip, so they never get stronger than "slight edge"."""
    lead = max(p, 1 - p)
    if lead < 0.55:
        return "toss-up"
    if lead < 0.65 or draft_only:
        return "slight edge"
    if lead < 0.80:
        return "favored"
    return "strong favorite"


def _side(value: float) -> str:
    return "Blue" if value > 0 else "Red"


def reason_text(feature: str, value: float) -> str | None:
    """One sentence for a factor, or None when its value gives neither side anything."""
    if abs(value) < 1e-9:
        return None
    s = _side(value)
    if feature in _COUNTS:
        verb, one, many = _COUNTS[feature]
        n = round(abs(value))
        return f"{s} {verb} {n:,} {one if n == 1 else many}"
    if feature == "level_diff":
        n = round(abs(value))
        return f"{s} is {n} level{'' if n == 1 else 's'} ahead in total"
    if feature in _FIRSTS:
        return f"{s} took {_FIRSTS[feature]}"
    if feature.startswith("lane_") and feature.removeprefix("lane_") in ROLES:
        role = _ROLE_WORDS[feature.removeprefix("lane_")]
        return f"{s}'s {role} matchup has won {0.5 + abs(value):.0%} of games"
    if feature == "champ_wr_diff":
        return f"{s}'s champions win {abs(value) * 100:.1f} pts more often on average"
    if feature == "duo_diff":
        return f"{s}'s bot duo has the better record together"
    if feature.startswith("tag_") and feature.endswith("_diff"):
        one, many = _TAGS[feature[4:-5]]
        n = round(abs(value))
        return f"{s} has {n} more {one if n == 1 else many}" if n else None
    if feature == "attack_diff":
        return f"{s}'s team leans more on physical damage"
    if feature == "magic_diff":
        return f"{s}'s team leans more on magic damage"
    if feature == "draft_logit":
        return f"The draft leans {s}"
    return None


_PHRASES = {"slight edge": "has a slight edge", "favored": "is favored", "strong favorite": "is a strong favorite"}


def pick_reasons(factors: list[tuple[str, float, float]], limit: int = MAX_REASONS) -> list[Reason]:
    """Sentences for (feature, input value, effect in pts), largest effect first.

    Effects are measured against the average training game, so a small input can push the other
    way from what its sign says (CS -1 is above the average CS diff and helps Blue). Such a factor
    gets no sentence: "Red leads by 1 CS" next to a Blue effect would contradict itself.
    """
    out = []
    for feature, value, pts in factors:
        text = reason_text(feature, value)
        if text and (value > 0) == (pts > 0):
            out.append(Reason(text=text, value_pct=pts))
        if len(out) == limit:
            break
    return out


def verdict(p: float, top_reason: str | None, draft_only: bool = False) -> str:
    """'Blue is favored: 68%, mostly because …'. `top_reason` should favour the leading side."""
    side, pct = ("Blue", p) if p >= 0.5 else ("Red", 1 - p)
    word = strength(p, draft_only)
    if word == "toss-up":
        return f"Toss-up: {side} {pct:.0%}"
    line = f"{side} {_PHRASES[word]}: {pct:.0%}"
    if top_reason:
        reason = top_reason[0].lower() + top_reason[1:] if top_reason.startswith("The ") else top_reason
        line += f", mostly because {reason}"
    return line
