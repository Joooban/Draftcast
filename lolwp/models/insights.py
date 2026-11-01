"""CLI: statistics for the Insights page → artifacts/insights.json."""
import json
import math
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import binomtest

from lolwp import champions
from lolwp.config import ROLES, load_settings

MIN_CHAMP_GAMES = 10  # the app filters further (default 30); this keeps rarer picks searchable
OBJECTIVE_TERMS = ["gold_k", "dragons_diff", "towers_diff", "grubs_diff", "kills_diff"]


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # At 0 or n wins the bound is exactly 0 or 1; avoid float noise there.
    lo = 0.0 if wins == 0 else center - margin
    hi = 1.0 if wins == n else center + margin
    return lo, hi


def champion_rates(draft: pd.DataFrame, names: dict[int, str]) -> list[dict]:
    train = draft[draft.split == "train"]
    picks = pd.concat(
        [pd.DataFrame({"champ": train[f"blue_{r}"], "win": train.blue_win}) for r in ROLES]
        + [pd.DataFrame({"champ": train[f"red_{r}"], "win": 1 - train.blue_win}) for r in ROLES])
    agg = picks.groupby("champ").win.agg(["sum", "count"])
    out = []
    for champ, row in agg[agg["count"] >= MIN_CHAMP_GAMES].iterrows():
        wins, n = int(row["sum"]), int(row["count"])
        lo, hi = wilson(wins, n)
        out.append({"id": int(champ), "name": names.get(int(champ), str(champ)), "games": n,
                    "win_rate": wins / n, "ci_low": lo, "ci_high": hi})
    return sorted(out, key=lambda c: c["win_rate"], reverse=True)


def gold_curve(state: pd.DataFrame) -> list[dict]:
    out = []
    for center in range(-8000, 8001, 1000):
        rows = state[(state.gold_diff >= center - 500) & (state.gold_diff < center + 500)]
        if len(rows):
            out.append({"gold_bin_center": center, "win_rate": float(rows.blue_win.mean()), "n": len(rows)})
    return out


def objectives(state: pd.DataFrame) -> dict:
    df = state.assign(gold_k=state.gold_diff / 1000)
    # A term with no variation (e.g. no grubs taken by 15:00) makes the fit singular.
    used = [t for t in OBJECTIVE_TERMS if df[t].nunique() > 1]
    fit = smf.logit("blue_win ~ " + " + ".join(used), data=df).fit(disp=0)
    ci = fit.conf_int()
    terms = [{"term": t, "odds_ratio": float(np.exp(fit.params[t])), "ci_low": float(np.exp(ci.loc[t, 0])),
              "ci_high": float(np.exp(ci.loc[t, 1])), "p_value": float(fit.pvalues[t])} for t in used]
    test = fit.t_test("dragons_diff = towers_diff")
    return {"terms": terms,
            "dragon_vs_tower": {"estimate_diff": float(np.squeeze(test.effect)),
                                "p_value": float(np.squeeze(test.pvalue))}}


def make_insights(data_dir: str, artifacts_dir: str) -> dict:
    draft = pd.read_parquet(os.path.join(data_dir, "draft.parquet"))
    state = pd.read_parquet(os.path.join(data_dir, "state.parquet"))
    names = {cid: c.name for cid, c in champions.load(os.path.join(artifacts_dir, "champions.json")).items()}
    with open(os.path.join(artifacts_dir, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)

    wins, n = int(draft.blue_win.sum()), len(draft)
    insights = {
        "champions": champion_rates(draft, names),
        "gold_curve": gold_curve(state),
        "objectives": objectives(state),
        "blue_side": {"win_rate": wins / n, "n": n, "p_value": float(binomtest(wins, n, 0.5).pvalue)},
        "model_comparison": metrics["models"],
        "calibration": metrics["calibration"],
        "data": metrics["data"],
    }
    with open(os.path.join(artifacts_dir, "insights.json"), "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=1)
    return insights


def main() -> None:
    s = load_settings()
    ins = make_insights(s.data_dir, s.artifacts_dir)
    print(f"{len(ins['champions'])} champions with >= {MIN_CHAMP_GAMES} games")
    print(f"Blue side: {ins['blue_side']}")
    for o in ins["objectives"]["terms"]:
        print(f"{o['term']:<14} OR {o['odds_ratio']:.3f} [{o['ci_low']:.3f}, {o['ci_high']:.3f}] p={o['p_value']:.3g}")
    print(f"Dragon vs tower: {ins['objectives']['dragon_vs_tower']}")


if __name__ == "__main__":
    main()
