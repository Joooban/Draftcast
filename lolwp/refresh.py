"""CLI: one command to refresh everything for a new patch.

Collects up to the given store totals (resumable), then rebuilds features, retrains and
recomputes insights. Stops before building if the Riot key expired, so a half-collected
store never produces new artifacts.
"""
import argparse
import os
import sys

from lolwp.config import Settings, load_settings
from lolwp.riot.client import KeyExpiredError, RiotClient
from lolwp.riot.collect import EXPIRED_MSG, collect_details, collect_ids, collect_timelines
from lolwp.riot.store import Store


def refresh(client, store: Store, settings: Settings, ids: int, details: int, timelines: int,
            data_dir: str, artifacts_dir: str) -> bool:
    try:
        print(f"{collect_ids(client, store, settings, ids, raise_expired=True)} new match ids")
        print(f"{collect_details(client, store, settings, details, raise_expired=True)} details saved")
        print(f"{collect_timelines(client, store, settings, timelines, raise_expired=True)} timelines saved")
    except KeyExpiredError:
        print(EXPIRED_MSG)
        return False
    print(store.counts())

    # Heavy imports only once collection is done.
    from lolwp.features.build import build
    from lolwp.models import insights, train

    summary = build(store, os.path.join(artifacts_dir, "champions.json"), data_dir, artifacts_dir, settings.platform)
    print(f"Built: test patch {summary['test_patch']}, draft rows {summary['draft']}, state rows {summary['state']}")
    metrics = train.train(data_dir, artifacts_dir)
    print(f"Trained: combined AUC {metrics['models']['combined_lr']['auc']:.3f}")
    insights.make_insights(data_dir, artifacts_dir)
    print("Insights written")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m lolwp.refresh")
    parser.add_argument("--ids", type=int, default=20000, help="target total match IDs")
    parser.add_argument("--details", type=int, default=20000, help="target total match details")
    parser.add_argument("--timelines", type=int, default=8000, help="target total timelines")
    args = parser.parse_args(argv)

    settings = load_settings()
    if not settings.api_key:
        parser.error("RIOT_API_KEY is not set in .env")
    ok = refresh(RiotClient(settings.api_key), Store(settings.db_path), settings, args.ids, args.details,
                 args.timelines, settings.data_dir, settings.artifacts_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
