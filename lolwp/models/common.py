"""Model helpers shared by training and the API.

Kept free of LightGBM and statsmodels so the API process stays small.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lolwp.config import STATE_FEATURES

COMBINED_FEATURES = ["draft_logit", *STATE_FEATURES]
# Rows shipped in artifacts/ so the API can refit on any subset of known fields. No match IDs.
SHIPPED_COLUMNS = ["split", "blue_win", "draft_logit", *STATE_FEATURES]


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def scores(y, p) -> dict[str, float]:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, p > 0.5)),
        "brier": float(brier_score_loss(y, p)),
    }


def calibration_bins(y, p, n_bins: int = 10) -> list[dict]:
    """Equal-width bins of predicted P(Blue): how often Blue actually won in each. Empty bins dropped."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins = []
    for i in range(n_bins):
        mask = idx == i
        if mask.any():
            bins.append({"lo": round(i / n_bins, 10), "hi": round((i + 1) / n_bins, 10),
                         "mean_pred": float(p[mask].mean()), "win_rate": float(y[mask].mean()),
                         "n": int(mask.sum())})
    return bins
