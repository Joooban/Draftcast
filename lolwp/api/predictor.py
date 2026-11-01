"""Loads the trained artifacts and turns validated requests into explained predictions."""
import json
import os

import joblib
import pandas as pd

from lolwp import champions as champ_data
from lolwp.api.schemas import ContributionOut, PredictRequest, PredictResponse, Split
from lolwp.config import ROLES, STATE_FEATURES, STATE_RANGES
from lolwp.features.draft import DRAFT_FEATURES, draft_features
from lolwp.features.draft_stats import DraftStats
from lolwp.models.explain import explain, sigmoid
from lolwp.models.common import COMBINED_FEATURES

ROLE_NAMES = {"top": "Top", "jungle": "Jungle", "mid": "Mid", "bot": "Bot", "support": "Support"}
LABELS = {
    "champ_wr_diff": "Champion strength",
    **{f"lane_{r}": f"{ROLE_NAMES[r]} lane matchup" for r in ROLES},
    "duo_diff": "Bot duo synergy",
    "tag_Tank_diff": "Tanks", "tag_Fighter_diff": "Fighters", "tag_Mage_diff": "Mages",
    "tag_Assassin_diff": "Assassins", "tag_Marksman_diff": "Marksmen", "tag_Support_diff": "Supports",
    "attack_diff": "Physical damage", "magic_diff": "Magic damage",
    "gold_diff": "Gold lead at 15:00", "cs_diff": "CS lead at 15:00",
    "level_diff": "Level lead at 15:00", "kills_diff": "Kill lead", "dragons_diff": "Dragons",
    "grubs_diff": "Voidgrubs", "towers_diff": "Towers",
    "plates_diff": "Turret plates", "wards_placed_diff": "Wards placed", "wards_killed_diff": "Wards cleared",
    "first_blood_blue": "First blood", "first_dragon_blue": "First dragon", "first_tower_blue": "First tower",
    "draft_logit": "Draft",
}


class ValidationError(ValueError):
    """A request that is well-formed JSON but not a valid prediction input."""


class Predictor:
    def __init__(self, artifacts_dir: str):
        path = lambda name: os.path.join(artifacts_dir, name)
        self.models = {n: joblib.load(path(f"{n}_model.joblib")) for n in ("draft", "state", "combined")}
        self.stats = DraftStats.from_json(path("draft_stats.json"))
        self.champs = champ_data.load(path("champions.json"))
        self._insights_path = path("insights.json")

    def champions(self) -> list[dict]:
        return [{"id": c.id, "name": c.name, "icon_url": c.icon_url, "tags": c.tags}
                for c in sorted(self.champs.values(), key=lambda c: c.name)]

    def insights(self) -> dict:
        with open(self._insights_path, encoding="utf-8") as f:
            return json.load(f)

    # --- validation ---

    def _draft_input(self, req: PredictRequest) -> tuple[dict[str, int], dict[str, int]]:
        if req.blue is None or req.red is None:
            raise ValidationError(f"Mode '{req.mode}' needs a full draft: all 10 picks for Blue and Red.")
        blue, red = req.blue.model_dump(), req.red.model_dump()
        picks = list(blue.values()) + list(red.values())
        unknown = sorted({p for p in picks if p not in self.champs})
        if unknown:
            raise ValidationError(f"Unknown champion id(s): {', '.join(map(str, unknown))}.")
        repeated = sorted({self.champs[p].name for p in picks if picks.count(p) > 1})
        if repeated:
            raise ValidationError(f"A champion can't be picked twice: {', '.join(repeated)}.")
        return blue, red

    def _state_input(self, req: PredictRequest) -> dict[str, int]:
        if req.state is None:
            raise ValidationError(f"Mode '{req.mode}' needs the game state at 15:00.")
        missing = [k for k in STATE_FEATURES if k not in req.state]
        if missing:
            raise ValidationError(f"Missing game-state value(s): {', '.join(missing)}.")
        extra = sorted(set(req.state) - set(STATE_FEATURES))
        if extra:
            raise ValidationError(f"Unknown game-state field(s): {', '.join(extra)}.")
        for k in STATE_FEATURES:
            lo, hi = STATE_RANGES[k]
            if not lo <= req.state[k] <= hi:
                raise ValidationError(f"{k} must be between {lo} and {hi} (got {req.state[k]}).")
        return {k: req.state[k] for k in STATE_FEATURES}

    # --- prediction ---

    def predict(self, req: PredictRequest) -> PredictResponse:
        # Validate everything the mode needs before predicting anything.
        draft = self._draft_input(req) if req.mode in ("draft", "both") else None
        state = self._state_input(req) if req.mode in ("state", "both") else None

        draft_x = None
        if draft is not None:
            draft_x = pd.DataFrame([draft_features(*draft, self.stats, self.champs)])[DRAFT_FEATURES]

        if req.mode == "draft":
            model, x = self.models["draft"], draft_x
        elif req.mode == "state":
            model, x = self.models["state"], pd.DataFrame([state])[STATE_FEATURES]
        else:
            logit = float(self.models["draft"].decision_function(draft_x)[0])
            model, x = self.models["combined"], pd.DataFrame([{"draft_logit": logit, **state}])[COMBINED_FEATURES]

        baseline, prob, contribs = explain(model, x)
        split = self._split(model, x, baseline, prob) if req.mode == "both" else None
        return PredictResponse(
            probability=prob,
            baseline=baseline,
            contributions=[ContributionOut(feature=c.feature, label=LABELS.get(c.feature, c.feature),
                                           value_pct=c.value_pct) for c in contribs],
            reasons=[self._reason(c, x) for c in contribs[:3]],
            split=split,
        )

    @staticmethod
    def _split(model, x: pd.DataFrame, baseline: float, prob: float) -> Split:
        """Draft share = change from adding the draft input to the baseline; game state gets the rest."""
        scaler, lr = model[0], model[-1]
        i = list(x.columns).index("draft_logit")
        draft_term = lr.coef_[0][i] * scaler.transform(x)[0][i]
        draft_pct = (sigmoid(lr.intercept_[0] + draft_term) - baseline) * 100
        return Split(draft_pct=draft_pct, state_pct=(prob - baseline) * 100 - draft_pct)

    @staticmethod
    def _reason(c, x: pd.DataFrame) -> str:
        side = "Blue" if c.value_pct >= 0 else "Red"
        if c.feature.startswith("lane_"):
            role = ROLE_NAMES[c.feature.removeprefix("lane_")].lower()
            rate = float(x[c.feature].iloc[0]) + 0.5
            return f"Blue's {role} lane matchup historically wins {rate:.0%} of games ({c.value_pct:+.1f}%)"
        return f"{LABELS.get(c.feature, c.feature)} favors {side} ({c.value_pct:+.1f}%)"
