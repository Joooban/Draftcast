"""Models for whichever game-state fields the user actually knows.

A blank field means "unknown", so instead of guessing it as 0 we use a logistic regression
fitted on just the known fields. The full field set uses the saved model; any other set is
fitted on the shipped training rows, scored on the shipped test-patch rows (so every number
quoted to the user belongs to the model that predicted), and cached.
"""
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import joblib
import pandas as pd

from lolwp.config import STATE_FEATURES
from lolwp.models.common import COMBINED_FEATURES, calibration_bins, make_lr, scores

Kind = Literal["state", "combined"]
CACHE_SIZE = 256


@dataclass
class FittedModel:
    model: object
    features: list[str]
    auc: float
    accuracy: float
    calibration: list[dict]


def model_name(kind: Kind, minute: int) -> str:
    return f"{kind}_lr" + ("" if minute == 15 else f"_m{minute}")


class ModelBank:
    def __init__(self, artifacts_dir: str):
        self.dir = artifacts_dir
        with open(os.path.join(artifacts_dir, "metrics.json"), encoding="utf-8") as f:
            self.metrics = json.load(f)
        self._rows: dict[int, pd.DataFrame] = {}
        self._cache: OrderedDict[tuple, FittedModel] = OrderedDict()
        self.fits = 0

    def rows(self, minute: int) -> pd.DataFrame:
        if minute not in self._rows:
            self._rows[minute] = pd.read_parquet(os.path.join(self.dir, f"state_rows_m{minute}.parquet"))
        return self._rows[minute]

    def get(self, kind: Kind, minute: int, features: tuple[str, ...]) -> FittedModel:
        key = (kind, minute, features)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        full = STATE_FEATURES if kind == "state" else COMBINED_FEATURES
        name = model_name(kind, minute)
        if list(features) == full:
            suffix = "" if minute == 15 else f"_m{minute}"
            model = joblib.load(os.path.join(self.dir, f"{kind}{suffix}_model.joblib"))
            m = self.metrics["models"][name]
            fitted = FittedModel(model, list(full), m["auc"], m["accuracy"], self.metrics["calibration"][name])
        else:
            rows = self.rows(minute)
            tr, te = rows[rows.split == "train"], rows[rows.split == "test"]
            cols = list(features)
            model = make_lr().fit(tr[cols], tr.blue_win)
            p = model.predict_proba(te[cols])[:, 1]
            s = scores(te.blue_win, p)
            fitted = FittedModel(model, cols, s["auc"], s["accuracy"], calibration_bins(te.blue_win, p))
            self.fits += 1
        self._cache[key] = fitted
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return fitted
