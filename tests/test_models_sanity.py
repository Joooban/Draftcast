import json
import os

import joblib
import numpy as np
import pytest
import pandas as pd

from lolwp import champions
from lolwp.config import ROLES, STATE_FEATURES
from lolwp.features.draft import DRAFT_FEATURES, draft_features
from lolwp.features.draft_stats import DraftStats

MODELS = ["baseline", "draft_lr", "draft_lgbm", "state_lr", "state_lgbm", "combined_lr", "combined_lgbm",
          "state_lr_m10", "combined_lr_m10"]
APP_MODELS = ["draft_lr", "state_lr", "combined_lr", "state_lr_m10", "combined_lr_m10"]
METRICS = {"auc", "log_loss", "accuracy", "brier", "cv_auc_mean", "cv_auc_std"}


def load(dirs, name):
    return joblib.load(os.path.join(dirs.artifacts_dir, f"{name}_model.joblib"))


def test_probabilities_in_range(pipeline_dirs):
    draft = pd.read_parquet(os.path.join(pipeline_dirs.data_dir, "draft.parquet")).query("split == 'test'")
    state = pd.read_parquet(os.path.join(pipeline_dirs.data_dir, "state.parquet")).query("split == 'test'")
    for p in (load(pipeline_dirs, "draft").predict_proba(draft[DRAFT_FEATURES])[:, 1],
              load(pipeline_dirs, "state").predict_proba(state[STATE_FEATURES])[:, 1]):
        assert len(p) and np.all((p >= 0) & (p <= 1))


def test_draft_swap_symmetry(pipeline_dirs):
    art = pipeline_dirs.artifacts_dir
    draft = pd.read_parquet(os.path.join(pipeline_dirs.data_dir, "draft.parquet")).query("split == 'test'")
    stats = DraftStats.from_json(os.path.join(art, "draft_stats.json"))
    champs = champions.load(os.path.join(art, "champions.json"))
    swapped = pd.DataFrame([
        draft_features({r: int(row[f"red_{r}"]) for r in ROLES}, {r: int(row[f"blue_{r}"]) for r in ROLES},
                       stats, champs)
        for _, row in draft.iterrows()])
    model = load(pipeline_dirs, "draft")
    p = model.predict_proba(draft[DRAFT_FEATURES])[:, 1]
    p_swapped = model.predict_proba(swapped[DRAFT_FEATURES])[:, 1]
    assert np.mean(np.abs(p + p_swapped - 1)) <= 0.05


def test_state_swap_symmetry(pipeline_dirs):
    state = pd.read_parquet(os.path.join(pipeline_dirs.data_dir, "state.parquet")).query("split == 'test'")
    model = load(pipeline_dirs, "state")
    p = model.predict_proba(state[STATE_FEATURES])[:, 1]
    p_swapped = model.predict_proba(-state[STATE_FEATURES])[:, 1]
    assert np.mean(np.abs(p + p_swapped - 1)) <= 0.05


def test_combined_model_uses_draft_logit(pipeline_dirs):
    model = load(pipeline_dirs, "combined")
    assert list(model.feature_names_in_) == ["draft_logit", *STATE_FEATURES]


def test_metrics_file_complete(pipeline_dirs):
    m = json.load(open(os.path.join(pipeline_dirs.artifacts_dir, "metrics.json")))
    assert set(m["models"]) == set(MODELS)
    for name in MODELS:
        assert set(m["models"][name]) == METRICS
        assert 0 <= m["models"][name]["auc"] <= 1
    assert {"test_patch", "n_train", "n_test", "prev_patch_auc"} <= set(m)
    assert set(m["prev_patch_auc"]) == {"draft_lr", "state_lr", "combined_lr"}
    assert set(m["calibration"]) == set(APP_MODELS)
    for name in APP_MODELS:
        n_test = m["n_test"]["draft" if name == "draft_lr" else "state"]
        assert sum(b["n"] for b in m["calibration"][name]) == n_test
    assert m["data"]["n_games"] == m["n_train"]["draft"] + m["n_test"]["draft"]
    assert m["data"]["test_patch"] == m["test_patch"] and m["data"]["train_patches"]
    assert isinstance(m["data"]["dropped"], dict)


def test_shipped_rows_have_no_ids(pipeline_dirs):
    from lolwp.models.common import SHIPPED_COLUMNS

    for minute in (10, 15):
        rows = pd.read_parquet(os.path.join(pipeline_dirs.artifacts_dir, f"state_rows_m{minute}.parquet"))
        assert list(rows.columns) == SHIPPED_COLUMNS
        assert set(rows.split) == {"train", "test"} and rows.draft_logit.notna().all()


def test_m10_models_saved(pipeline_dirs):
    for name in ("state_m10", "combined_m10"):
        assert load(pipeline_dirs, name).predict_proba is not None


def test_calibration_bins_basic():
    from lolwp.models.common import calibration_bins

    bins = calibration_bins([0, 1, 1, 0], [0.05, 0.95, 0.92, 0.15])
    assert [(b["lo"], b["hi"], b["n"]) for b in bins] == [(0.0, 0.1, 1), (0.1, 0.2, 1), (0.9, 1.0, 2)]
    assert bins[-1]["win_rate"] == 1.0 and bins[0]["win_rate"] == 0.0
    assert bins[-1]["mean_pred"] == pytest.approx(0.935)


def test_synthetic_signal_is_learned(synthetic_dirs):
    m = json.load(open(os.path.join(synthetic_dirs.artifacts_dir, "metrics.json")))["models"]
    assert m["state_lr"]["auc"] > 0.7
    assert m["combined_lr"]["auc"] >= m["state_lr"]["auc"] - 0.02


def test_drift_features_use_only_older_patches(synthetic_dirs):
    from lolwp.features.parse import patch_key
    from lolwp.models.train import COMBINED_FEATURES, drift_frames, records_from_frame

    draft = pd.read_parquet(os.path.join(synthetic_dirs.data_dir, "draft.parquet"))
    state = pd.read_parquet(os.path.join(synthetic_dirs.data_dir, "state.parquet"))
    champs = champions.load(os.path.join(synthetic_dirs.artifacts_dir, "champions.json"))
    d, c = drift_frames(draft, state, champs)

    patches = sorted(draft.patch.unique(), key=patch_key)
    old = draft[~draft.patch.isin(patches[-2:])]
    stats = DraftStats.fit(records_from_frame(old))
    for patch in patches[-2:]:
        row = d[d.patch == patch].iloc[0]
        blue = {r: int(row[f"blue_{r}"]) for r in ROLES}
        red = {r: int(row[f"red_{r}"]) for r in ROLES}
        expected = draft_features(blue, red, stats, champs)
        assert [row[k] for k in DRAFT_FEATURES] == pytest.approx([expected[k] for k in DRAFT_FEATURES])
    assert len(c) == len(state) and set(COMBINED_FEATURES) <= set(c.columns)
    assert not c.draft_logit.isna().any()
