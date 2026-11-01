"""Collector CLI: seed players from ranked ladders, then fetch match IDs, details and timelines.

Every step reads its to-do list from the store, so rerunning after a key expiry resumes.
`--limit N` is a total: a step stops once the store holds N of its item, so reruns never overshoot.
"""
import argparse
import time
from urllib.parse import quote

from lolwp.config import ACCOUNT_ROUTING, MATCH_WINDOW_DAYS, QUEUE_ID, SEED, Settings, load_settings
from lolwp.riot.client import KeyExpiredError, RiotClient, TransientError
from lolwp.riot.store import Store

DIVISIONS = ["I", "II", "III", "IV"]
# (tier, divisions, pages)
SEED_TIERS = [
    ("EMERALD", DIVISIONS, [1, 2]),
    ("DIAMOND", DIVISIONS, [1, 2]),
    ("MASTER", ["I"], [1]),
    ("GRANDMASTER", ["I"], [1]),
    ("CHALLENGER", ["I"], [1]),
]
EXPIRED_MSG = "Key expired — update .env and rerun"


def _platform(settings: Settings) -> str:
    return f"https://{settings.platform}.api.riotgames.com"


def _regional(settings: Settings) -> str:
    return f"https://{settings.region}.api.riotgames.com"


def check(client, settings: Settings, riot_id: str) -> None:
    name, _, tag = riot_id.partition("#")
    account = client.get(
        f"https://{ACCOUNT_ROUTING}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/"
        f"{quote(name)}/{quote(tag)}")
    if not account:
        print(f"Riot ID {riot_id} not found")
        return
    puuid = account["puuid"]
    ids = client.get(f"{_regional(settings)}/lol/match/v5/matches/by-puuid/{puuid}/ids",
                     {"queue": QUEUE_ID, "count": 5}) or []
    print(f"PUUID: {puuid}")
    for match_id in ids:
        print(match_id)


def seed(client, store: Store, settings: Settings) -> int:
    added = 0
    try:
        for tier, divisions, pages in SEED_TIERS:
            for division in divisions:
                for page in pages:
                    try:
                        entries = client.get(
                            f"{_platform(settings)}/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}",
                            {"page": page}) or []
                    except TransientError as e:
                        print(f"Skipped {tier} {division} p{page} ({e}); rerun seed to retry")
                        continue
                    rows = [(e["puuid"], tier, division) for e in entries if e.get("puuid")]
                    n = store.add_players(rows)
                    added += n
                    print(f"{tier} {division} p{page}: {len(rows)} entries, {n} new")
    except KeyExpiredError:
        print(EXPIRED_MSG)
    return added


def _remaining(store: Store, item: str, limit: int) -> int:
    return max(0, limit - store.counts()[item])


def collect_ids(client, store: Store, settings: Settings, limit: int, per_player: int = 50,
                now: float | None = None, raise_expired: bool = False) -> int:
    """Scan unscanned players until the store holds `limit` match IDs."""
    now = time.time() if now is None else now
    params = {"startTime": int(now - MATCH_WINDOW_DAYS * 86400), "queue": QUEUE_ID, "count": per_player}
    added = 0
    try:
        for i, puuid in enumerate(store.players_todo(_remaining(store, "match_ids", limit)), 1):
            if not _remaining(store, "match_ids", limit):
                break
            try:
                ids = client.get(f"{_regional(settings)}/lol/match/v5/matches/by-puuid/{puuid}/ids", params) or []
            except TransientError as e:
                print(f"Skipped player {puuid[:8]}… for now ({e}); a rerun will retry it")
                continue
            added += store.add_match_ids(ids)
            store.mark_player_done(puuid)
            if i % 50 == 0:
                print(f"{i} players, {added} new match ids")
    except KeyExpiredError:
        if raise_expired:
            raise
        print(EXPIRED_MSG)
    return added


def collect_details(client, store: Store, settings: Settings, limit: int, raise_expired: bool = False) -> int:
    saved = 0
    try:
        for i, match_id in enumerate(store.details_todo(_remaining(store, "details", limit)), 1):
            try:
                match = client.get(f"{_regional(settings)}/lol/match/v5/matches/{match_id}")
            except TransientError as e:
                print(f"Skipped {match_id} for now ({e}); a rerun will retry it")
                continue
            store.save_details(match_id, match)
            saved += match is not None
            if i % 100 == 0:
                print(f"{i} fetched, {saved} saved")
    except KeyExpiredError:
        if raise_expired:
            raise
        print(EXPIRED_MSG)
    return saved


def collect_timelines(client, store: Store, settings: Settings, limit: int, raise_expired: bool = False) -> int:
    saved = 0
    try:
        for i, match_id in enumerate(store.timelines_todo(_remaining(store, "timelines", limit), seed=SEED), 1):
            try:
                timeline = client.get(f"{_regional(settings)}/lol/match/v5/matches/{match_id}/timeline")
            except TransientError as e:
                print(f"Skipped {match_id} for now ({e}); a rerun will retry it")
                continue
            store.save_timeline(match_id, timeline)
            saved += timeline is not None
            if i % 100 == 0:
                print(f"{i} fetched, {saved} saved")
    except KeyExpiredError:
        if raise_expired:
            raise
        print(EXPIRED_MSG)
    return saved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m lolwp.riot.collect")
    parser.add_argument("command", choices=["check", "seed", "ids", "details", "timelines", "status"])
    parser.add_argument("--limit", type=int, default=1000,
                        help="stop once the store holds this many of the step's item (ids, details, timelines)")
    parser.add_argument("--riot-id", help="Name#TAG, for check")
    args = parser.parse_args(argv)

    settings = load_settings()
    store = Store(settings.db_path)
    if args.command == "status":
        print(store.counts())
        return
    if not settings.api_key:
        parser.error("RIOT_API_KEY is not set in .env")
    client = RiotClient(settings.api_key)
    if args.command == "check":
        if not args.riot_id or "#" not in args.riot_id:
            parser.error("check needs --riot-id Name#TAG")
        check(client, settings, args.riot_id)
    elif args.command == "seed":
        print(f"{seed(client, store, settings)} new players")
    elif args.command == "ids":
        print(f"{collect_ids(client, store, settings, args.limit)} new match ids")
    elif args.command == "details":
        print(f"{collect_details(client, store, settings, args.limit)} details saved")
    elif args.command == "timelines":
        print(f"{collect_timelines(client, store, settings, args.limit)} timelines saved")
    print(store.counts())


if __name__ == "__main__":
    main()
