import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lolwp.models.explain import explain


@pytest.fixture
def pipe_and_x():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=["a", "b", "c", "d"])
    y = (X.a * 2 - X.b + 0.3 * X.c + rng.normal(size=400) > 0).astype(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    return pipe, pd.DataFrame([[1.5, -0.4, 2.0, 0.1]], columns=X.columns)


def test_contributions_sum_to_final(pipe_and_x):
    base, p, cs = explain(*pipe_and_x)
    assert base + sum(c.value_pct for c in cs) / 100 == pytest.approx(p, abs=1e-9)


def test_final_matches_predict_proba(pipe_and_x):
    pipe, x = pipe_and_x
    _, p, _ = explain(pipe, x)
    assert p == pytest.approx(pipe.predict_proba(x)[0, 1], abs=1e-12)


def test_baseline_is_sigmoid_of_intercept(pipe_and_x):
    pipe, x = pipe_and_x
    base, _, _ = explain(pipe, x)
    b = pipe[-1].intercept_[0]
    assert base == pytest.approx(1 / (1 + np.exp(-b)))


def test_sorted_by_abs_logit(pipe_and_x):
    _, _, cs = explain(*pipe_and_x)
    mags = [abs(c.logit) for c in cs]
    assert mags == sorted(mags, reverse=True)
    assert [c.feature for c in cs][0] == "a"
    assert {c.feature for c in cs} == {"a", "b", "c", "d"}
