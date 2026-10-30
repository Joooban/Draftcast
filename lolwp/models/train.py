"""CLI: train draft, game-state and combined models, evaluate on the newest patch, save artifacts.

Logistic regression runs the app (exact explanations); LightGBM is a benchmark only.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lolwp.config import OOF_FOLDS, SEED, STATE_FEATURES, load_settings
from lolwp.features.draft import DRAFT_FEATURES
from lolwp.features.parse import patch_key

COMBINED_FEATURES = ["draft_logit", *STATE_FEATURES]
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=SEED, verbose=-1)


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def make_lgbm():
    return LGBMClassifier(**LGBM_PARAMS)


def _cv():
    return StratifiedKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=SEED)


def scores(y, p) -> dict[str, float]:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, p > 0.5)),
        "brier": float(brier_score_loss(y, p)),
    }


def baseline_prob(champ_wr_diff) -> np.ndarray:
    return 1 / (1 + np.exp(-10 * np.asarray(champ_wr_diff)))


def evaluate(model, X_train, y_train, X_test, y_test) -> tuple[object, dict]:
    cv = cross_val_score(model, X_train, y_train, cv=_cv(), scoring="roc_auc")
    model.fit(X_train, y_train)
    m = scores(y_test, model.predict_proba(X_test)[:, 1])
    m |= {"cv_auc_mean": float(cv.mean()), "cv_auc_std": float(cv.std())}
    return model, m


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


def train(data_dir: str, artifacts_dir: str) -> dict:
    draft = pd.read_parquet(os.path.join(data_dir, "draft.parquet"))
    state = pd.read_parquet(os.path.join(data_dir, "state.parquet"))
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
    for name, model in (("draft", draft_lr), ("state", state_lr), ("combined", combined_lr)):
        joblib.dump(model, os.path.join(artifacts_dir, f"{name}_model.joblib"))

    metrics = {
        "models": models,
        "test_patch": str(d_te.patch.iloc[0]) if len(d_te) else None,
        "n_train": {"draft": len(d_tr), "state": len(s_tr)},
        "n_test": {"draft": len(d_te), "state": len(s_te)},
        "prev_patch_auc": {
            "draft_lr": drift_check(draft, DRAFT_FEATURES),
            "state_lr": drift_check(state, STATE_FEATURES),
            "combined_lr": drift_check(combined, COMBINED_FEATURES),
        },
    }
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
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
