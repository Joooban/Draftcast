"""CLI: train draft, game-state and combined models, evaluate on the newest patch, save artifacts.

Logistic regression runs the app (exact explanations); LightGBM is a benchmark only.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from lolwp import champions
from lolwp.champions import Champion
from lolwp.config import OOF_FOLDS, ROLES, SEED, STATE_FEATURES, STATE_MINUTES, load_settings
from lolwp.features.draft import DRAFT_FEATURES, draft_features
from lolwp.features.draft_stats import DraftStats, oof_stats
from lolwp.features.parse import MatchRecord, patch_key
from lolwp.models.common import COMBINED_FEATURES, SHIPPED_COLUMNS, calibration_bins, make_lr, scores

__all__ = ["COMBINED_FEATURES", "make_lr", "scores", "train"]
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=SEED, verbose=-1)


def make_lgbm():
    return LGBMClassifier(**LGBM_PARAMS)


def _cv():
    return StratifiedKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=SEED)


def baseline_prob(champ_wr_diff) -> np.ndarray:
    return 1 / (1 + np.exp(-10 * np.asarray(champ_wr_diff)))


def evaluate(model, X_train, y_train, X_test, y_test) -> tuple[object, dict]:
    cv = cross_val_score(model, X_train, y_train, cv=_cv(), scoring="roc_auc")
    model.fit(X_train, y_train)
    m = scores(y_test, model.predict_proba(X_test)[:, 1])
    m |= {"cv_auc_mean": float(cv.mean()), "cv_auc_std": float(cv.std())}
    return model, m


def records_from_frame(frame: pd.DataFrame) -> list[MatchRecord]:
    return [MatchRecord(row.match_id, row.patch, int(row.blue_win), 0,
                        {r: int(getattr(row, f"blue_{r}")) for r in ROLES},
                        {r: int(getattr(row, f"red_{r}")) for r in ROLES})
            for row in frame.itertuples(index=False)]


def drift_frames(draft: pd.DataFrame, state: pd.DataFrame,
                 champs: dict[int, Champion]) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Rebuild draft features as if the newest two patches were both unseen.

    The saved features give second-newest-patch rows stats that include their own patch, which
    would flatter that patch and overstate drift. Here older rows get out-of-fold stats from the
    older patches only, and both evaluation patches get stats fitted on the older patches.
    """
    patches = sorted(draft.patch.unique(), key=patch_key)
    if len(patches) < 3:
        return None
    evaluated = draft.patch.isin(patches[-2:])
    old_recs = records_from_frame(draft[~evaluated])
    oof, full = oof_stats(old_recs), DraftStats.fit(old_recs)
    rows = []
    for rec, is_eval in zip(records_from_frame(draft), evaluated):
        rows.append(draft_features(rec.blue, rec.red, full if is_eval else oof[rec.match_id], champs))
    d = draft.copy()
    d[DRAFT_FEATURES] = pd.DataFrame(rows, index=draft.index)[DRAFT_FEATURES]

    old = d[~evaluated]
    oof_logit = cross_val_predict(make_lr(), old[DRAFT_FEATURES], old.blue_win, cv=_cv(),
                                  method="decision_function")
    model = make_lr().fit(old[DRAFT_FEATURES], old.blue_win)
    new = d[evaluated]
    logit = pd.concat([pd.Series(oof_logit, index=old.match_id),
                       pd.Series(model.decision_function(new[DRAFT_FEATURES]), index=new.match_id)])
    return d, state.assign(draft_logit=state.match_id.map(logit).to_numpy())


def drift_check(frame: pd.DataFrame, features: list[str]) -> dict | None:
    """Train on all but the newest two patches; score on the second-newest and the newest."""
    patches = sorted(frame.patch.unique(), key=patch_key)
    if len(patches) < 3:
        return None
    prev, newest = patches[-2], patches[-1]
    old = frame[~frame.patch.isin([prev, newest])]
    model = make_lr().fit(old[features], old.blue_win)
    auc = lambda d: float(roc_auc_score(d.blue_win, model.predict_proba(d[features])[:, 1]))
    return {"prev_patch": prev, "test_patch": newest,
            "auc_prev_patch": auc(frame[frame.patch == prev]), "auc_test_patch": auc(frame[frame.patch == newest])}


def _state_path(data_dir: str, minute: int) -> str:
    return os.path.join(data_dir, "state.parquet" if minute == 15 else f"state_m{minute}.parquet")


def train(data_dir: str, artifacts_dir: str) -> dict:
    draft = pd.read_parquet(os.path.join(data_dir, "draft.parquet"))
    state = pd.read_parquet(_state_path(data_dir, 15))
    d_tr, d_te = draft[draft.split == "train"], draft[draft.split == "test"]
    models: dict[str, dict] = {}

    # Baseline: no fitting, just champion win-rate difference mapped to a probability.
    folds = [roc_auc_score(d_tr.blue_win.iloc[i], baseline_prob(d_tr.champ_wr_diff.iloc[i]))
             for _, i in _cv().split(d_tr, d_tr.blue_win)]
    models["baseline"] = scores(d_te.blue_win, baseline_prob(d_te.champ_wr_diff)) | {
        "cv_auc_mean": float(np.mean(folds)), "cv_auc_std": float(np.std(folds))}

    draft_lr, models["draft_lr"] = evaluate(make_lr(), d_tr[DRAFT_FEATURES], d_tr.blue_win,
                                            d_te[DRAFT_FEATURES], d_te.blue_win)
    _, models["draft_lgbm"] = evaluate(make_lgbm(), d_tr[DRAFT_FEATURES], d_tr.blue_win,
                                       d_te[DRAFT_FEATURES], d_te.blue_win)

    s_tr, s_te = state[state.split == "train"], state[state.split == "test"]
    state_lr, models["state_lr"] = evaluate(make_lr(), s_tr[STATE_FEATURES], s_tr.blue_win,
                                            s_te[STATE_FEATURES], s_te.blue_win)
    _, models["state_lgbm"] = evaluate(make_lgbm(), s_tr[STATE_FEATURES], s_tr.blue_win,
                                       s_te[STATE_FEATURES], s_te.blue_win)

    # Combined: train rows get out-of-fold draft logits so the draft input never saw its own label.
    oof_logit = cross_val_predict(make_lr(), d_tr[DRAFT_FEATURES], d_tr.blue_win, cv=_cv(),
                                  method="decision_function")
    logit = pd.concat([pd.Series(oof_logit, index=d_tr.match_id),
                       pd.Series(draft_lr.decision_function(d_te[DRAFT_FEATURES]), index=d_te.match_id)])
    combined = state.assign(draft_logit=state.match_id.map(logit).to_numpy())
    c_tr, c_te = combined[combined.split == "train"], combined[combined.split == "test"]
    combined_lr, models["combined_lr"] = evaluate(make_lr(), c_tr[COMBINED_FEATURES], c_tr.blue_win,
                                                  c_te[COMBINED_FEATURES], c_te.blue_win)
    _, models["combined_lgbm"] = evaluate(make_lgbm(), c_tr[COMBINED_FEATURES], c_tr.blue_win,
                                          c_te[COMBINED_FEATURES], c_te.blue_win)

    os.makedirs(artifacts_dir, exist_ok=True)
    saved = {"draft": draft_lr, "state": state_lr, "combined": combined_lr}
    calibration = {"draft_lr": calibration_bins(d_te.blue_win, draft_lr.predict_proba(d_te[DRAFT_FEATURES])[:, 1])}

    # The app's state and combined LRs at each minute; minute 15 keeps the original names.
    for minute in STATE_MINUTES:
        frame = combined if minute == 15 else pd.read_parquet(_state_path(data_dir, minute)).assign(
            draft_logit=lambda f: f.match_id.map(logit).to_numpy())
        tr, te = frame[frame.split == "train"], frame[frame.split == "test"]
        suffix = "" if minute == 15 else f"_m{minute}"
        if minute == 15:
            s_model, c_model = state_lr, combined_lr
        else:
            s_model, models[f"state_lr{suffix}"] = evaluate(make_lr(), tr[STATE_FEATURES], tr.blue_win,
                                                            te[STATE_FEATURES], te.blue_win)
            c_model, models[f"combined_lr{suffix}"] = evaluate(make_lr(), tr[COMBINED_FEATURES], tr.blue_win,
                                                               te[COMBINED_FEATURES], te.blue_win)
            saved[f"state{suffix}"], saved[f"combined{suffix}"] = s_model, c_model
        calibration[f"state_lr{suffix}"] = calibration_bins(te.blue_win, s_model.predict_proba(te[STATE_FEATURES])[:, 1])
        calibration[f"combined_lr{suffix}"] = calibration_bins(
            te.blue_win, c_model.predict_proba(te[COMBINED_FEATURES])[:, 1])
        frame[SHIPPED_COLUMNS].to_parquet(os.path.join(artifacts_dir, f"state_rows_m{minute}.parquet"), index=False)

    for name, model in saved.items():
        joblib.dump(model, os.path.join(artifacts_dir, f"{name}_model.joblib"))

    drift = drift_frames(draft, state, champions.load(os.path.join(artifacts_dir, "champions.json")))
    drift_draft, drift_combined = drift if drift else (draft, combined)
    summary_path = os.path.join(artifacts_dir, "build_summary.json")
    dropped = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            dropped = json.load(f).get("dropped", {})
    metrics = {
        "models": models,
        "calibration": calibration,
        "data": {
            "train_patches": sorted(d_tr.patch.unique().tolist(), key=patch_key),
            "test_patch": str(d_te.patch.iloc[0]) if len(d_te) else None,
            "n_games": len(draft),
            "dropped": dropped,
        },
        "test_patch": str(d_te.patch.iloc[0]) if len(d_te) else None,
        "n_train": {"draft": len(d_tr), "state": len(s_tr)},
        "n_test": {"draft": len(d_te), "state": len(s_te)},
        "prev_patch_auc": {
            "draft_lr": drift_check(drift_draft, DRAFT_FEATURES),
            "state_lr": drift_check(state, STATE_FEATURES),
            "combined_lr": drift_check(drift_combined, COMBINED_FEATURES),
        },
    }
    with open(os.path.join(artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    s = load_settings()
    m = train(s.data_dir, s.artifacts_dir)
    print(f"Test patch {m['test_patch']}  train {m['n_train']}  test {m['n_test']}")
    print(f"{'model':<15}{'auc':>8}{'logloss':>9}{'acc':>7}{'brier':>8}{'cv_auc':>9}")
    for name, r in m["models"].items():
        print(f"{name:<15}{r['auc']:>8.4f}{r['log_loss']:>9.4f}{r['accuracy']:>7.3f}{r['brier']:>8.4f}"
              f"{r['cv_auc_mean']:>9.4f}")
    print("Drift:", json.dumps(m["prev_patch_auc"], indent=1))


if __name__ == "__main__":
    main()
