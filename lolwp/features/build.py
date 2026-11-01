"""CLI: raw store → data/draft.parquet, data/state{,_m10}.parquet, artifacts/draft_stats.json
and artifacts/build_summary.json.

The newest patch is the test set. Training rows get out-of-fold draft stats; test rows
(and the live API) use stats fitted on the whole training set.
"""
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

from lolwp import champions
from lolwp.config import MIN_DURATION_S, ROLES, STATE_FEATURES, STATE_MINUTES, load_settings
from lolwp.features.draft import DRAFT_FEATURES, draft_features
from lolwp.features.draft_stats import DraftStats, oof_stats
from lolwp.features.parse import MatchRecord, parse_match_checked, patch_key
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
    states: dict[str, dict[int, dict[str, int]]] = {}
    dropped: Counter[str] = Counter()
    for match_id, details, timeline in store.iter_matches():
        r, reason = parse_match_checked(details, platform)
        if r is None:
            dropped[reason] += 1
            continue
        records.append(r)
        if timeline is not None and r.duration_s >= MIN_DURATION_S:
            per_minute = {m: state_features(details, timeline, m) for m in STATE_MINUTES}
            if all(f is not None for f in per_minute.values()):
                states[r.match_id] = per_minute
    if not records:
        raise SystemExit("No valid matches in the store; run the collector first.")

    train, test, test_patch = split_by_patch(records)
    if not train:
        raise SystemExit(f"Only one patch ({test_patch}) present; need older patches to train on.")
    full = DraftStats.fit(train)
    oof = oof_stats(train)

    draft_rows: list[dict] = []
    state_rows: dict[int, list[dict]] = {m: [] for m in STATE_MINUTES}
    for split, recs in (("train", train), ("test", test)):
        for r in recs:
            stats = oof[r.match_id] if split == "train" else full
            row = {"match_id": r.match_id, "patch": r.patch, "split": split, "blue_win": r.blue_win}
            row |= draft_features(r.blue, r.red, stats, champs)
            row |= {f"blue_{role}": r.blue[role] for role in ROLES}
            row |= {f"red_{role}": r.red[role] for role in ROLES}
            draft_rows.append(row)

            per_minute = states.get(r.match_id)
            if per_minute is None:
                continue
            for m, feats in per_minute.items():
                state_rows[m].append({"match_id": r.match_id, "patch": r.patch, "split": split,
                                      "blue_win": r.blue_win, "duration_s": r.duration_s} | feats)

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    draft_cols = ["match_id", "patch", "split", "blue_win", *DRAFT_FEATURES,
                  *[f"{s}_{role}" for s in ("blue", "red") for role in ROLES]]
    state_cols = ["match_id", "patch", "split", "blue_win", "duration_s", *STATE_FEATURES]
    draft = pd.DataFrame(draft_rows, columns=draft_cols)
    draft.to_parquet(os.path.join(data_dir, "draft.parquet"), index=False)
    for m in STATE_MINUTES:
        name = "state.parquet" if m == 15 else f"state_m{m}.parquet"
        frame = pd.DataFrame(state_rows[m], columns=state_cols)
        frame.to_parquet(os.path.join(data_dir, name), index=False)
        if m == 15:
            state = frame
    full.to_json(os.path.join(artifacts_dir, "draft_stats.json"))
    patches = sorted({r.patch for r in records}, key=patch_key)
    with open(os.path.join(artifacts_dir, "build_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"dropped": dict(dropped), "patches": patches}, f, indent=1)

    return {
        "test_patch": test_patch,
        "dropped": sum(dropped.values()),
        "dropped_by_reason": dict(dropped),
        "draft": draft.split.value_counts().to_dict(),
        "state": state.split.value_counts().to_dict(),
    }


def main() -> None:
    s = load_settings()
    summary = build(Store(s.db_path), os.path.join(s.artifacts_dir, "champions.json"), s.data_dir, s.artifacts_dir,
                    s.platform)
    print(f"Test patch: {summary['test_patch']}  (invalid matches dropped: {summary['dropped']})")
    for reason, n in sorted(summary["dropped_by_reason"].items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {n}")
    print(f"Draft rows: {summary['draft']}")
    print(f"State rows: {summary['state']}")


if __name__ == "__main__":
    main()
