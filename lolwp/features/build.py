"""CLI: raw store → data/draft.parquet, data/state.parquet and artifacts/draft_stats.json.

The newest patch is the test set. Training rows get out-of-fold draft stats; test rows
(and the live API) use stats fitted on the whole training set.
"""
import os
from pathlib import Path

import pandas as pd

from lolwp import champions
from lolwp.config import MIN_DURATION_S, ROLES, STATE_FEATURES, load_settings
from lolwp.features.draft import DRAFT_FEATURES, draft_features
from lolwp.features.draft_stats import DraftStats, oof_stats
from lolwp.features.parse import MatchRecord, parse_match, patch_key
from lolwp.features.state import state_features
from lolwp.riot.store import Store


def split_by_patch(records: list[MatchRecord]) -> tuple[list[MatchRecord], list[MatchRecord], str]:
    test_patch = max((r.patch for r in records), key=patch_key)
    train = [r for r in records if r.patch != test_patch]
    test = [r for r in records if r.patch == test_patch]
    return train, test, test_patch


def build(store: Store, champions_path: str, data_dir: str, artifacts_dir: str,
          platform: str | None = None) -> dict:
    champs = champions.load(champions_path)
    records: list[MatchRecord] = []
    # State features don't depend on draft stats, so compute them while streaming and keep
    # only the small feature dicts; raw JSON for ~20k matches would not fit in memory.
    states: dict[str, dict[str, int]] = {}
    dropped = 0
    for match_id, details, timeline in store.iter_matches():
        r = parse_match(details, platform)
        if r is None:
            dropped += 1
            continue
        records.append(r)
        if timeline is not None and r.duration_s >= MIN_DURATION_S:
            feats = state_features(details, timeline)
            if feats is not None:
                states[r.match_id] = feats
    if not records:
        raise SystemExit("No valid matches in the store; run the collector first.")

    train, test, test_patch = split_by_patch(records)
    if not train:
        raise SystemExit(f"Only one patch ({test_patch}) present; need older patches to train on.")
    full = DraftStats.fit(train)
    oof = oof_stats(train)

    draft_rows, state_rows = [], []
    for split, recs in (("train", train), ("test", test)):
        for r in recs:
            stats = oof[r.match_id] if split == "train" else full
            row = {"match_id": r.match_id, "patch": r.patch, "split": split, "blue_win": r.blue_win}
            row |= draft_features(r.blue, r.red, stats, champs)
            row |= {f"blue_{role}": r.blue[role] for role in ROLES}
            row |= {f"red_{role}": r.red[role] for role in ROLES}
            draft_rows.append(row)

            feats = states.get(r.match_id)
            if feats is None:
                continue
            state_rows.append({"match_id": r.match_id, "patch": r.patch, "split": split,
                               "blue_win": r.blue_win, "duration_s": r.duration_s} | feats)

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    draft_cols = ["match_id", "patch", "split", "blue_win", *DRAFT_FEATURES,
                  *[f"{s}_{role}" for s in ("blue", "red") for role in ROLES]]
    state_cols = ["match_id", "patch", "split", "blue_win", "duration_s", *STATE_FEATURES]
    draft = pd.DataFrame(draft_rows, columns=draft_cols)
    state = pd.DataFrame(state_rows, columns=state_cols)
    draft.to_parquet(os.path.join(data_dir, "draft.parquet"), index=False)
    state.to_parquet(os.path.join(data_dir, "state.parquet"), index=False)
    full.to_json(os.path.join(artifacts_dir, "draft_stats.json"))

    return {
        "test_patch": test_patch,
        "dropped": dropped,
        "draft": draft.split.value_counts().to_dict(),
        "state": state.split.value_counts().to_dict(),
    }


def main() -> None:
    s = load_settings()
    summary = build(Store(s.db_path), os.path.join(s.artifacts_dir, "champions.json"), s.data_dir, s.artifacts_dir,
                    s.platform)
    print(f"Test patch: {summary['test_patch']}  (invalid matches dropped: {summary['dropped']})")
    print(f"Draft rows: {summary['draft']}")
    print(f"State rows: {summary['state']}")


if __name__ == "__main__":
    main()
