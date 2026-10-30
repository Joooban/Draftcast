"""Exact waterfall explanations for a StandardScaler + LogisticRegression pipeline.

In log-odds, each feature adds coef * scaled value. Adding features one at a time
(largest effect first) and converting to probability after each step gives bars that
sum exactly to the final probability minus the baseline.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Contribution:
    feature: str
    logit: float
    value_pct: float


def sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


def explain(pipeline, x: pd.DataFrame) -> tuple[float, float, list[Contribution]]:
    scaler, model = pipeline[0], pipeline[-1]
    scaled = scaler.transform(x)[0]
    logits = model.coef_[0] * scaled
    intercept = float(model.intercept_[0])

    order = sorted(range(len(logits)), key=lambda i: abs(logits[i]), reverse=True)
    baseline = sigmoid(intercept)
    z, prev, contributions = intercept, baseline, []
    for i in order:
        z += logits[i]
        p = sigmoid(z)
        contributions.append(Contribution(str(x.columns[i]), float(logits[i]), (p - prev) * 100))
        prev = p
    return baseline, prev, contributions
