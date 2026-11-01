"""Loads the trained artifacts and turns validated requests into explained predictions."""
import json
import os

import joblib
import pandas as pd

from lolwp import champions as champ_data
from lolwp.api.explain_text import pick_reasons, strength, verdict
from lolwp.api.schemas import (CalibrationBin, ContributionOut, PredictRequest, PredictResponse, Reliability,
                               Split)
from lolwp.config import ROLES, STATE_FEATURES, STATE_MINUTES, STATE_RANGES
from lolwp.features.draft import DRAFT_FEATURES, draft_features
from lolwp.features.draft_stats import DraftStats
from lolwp.models.explain import explain, sigmoid
from lolwp.models.refit import FittedModel, ModelBank, model_name

ROLE_NAMES = {"top": "Top", "jungle": "Jungle", "mid": "Mid", "bot": "Bot", "support": "Support"}
LABELS = {
    "champ_wr_diff": "Champion strength",
    **{f"lane_{r}": f"{ROLE_NAMES[r]} lane matchup" for r in ROLES},
    "duo_diff": "Bot duo synergy",
    "tag_Tank_diff": "Tanks", "tag_Fighter_diff": "Fighters", "tag_Mage_diff": "Mages",
    "tag_Assassin_diff": "Assassins", "tag_Marksman_diff": "Marksmen", "tag_Support_diff": "Supports",
    "attack_diff": "Physical damage", "magic_diff": "Magic damage",
    "gold_diff": "Gold lead", "cs_diff": "CS lead",
    "level_diff": "Level lead", "kills_diff": "Kill lead", "dragons_diff": "Dragons",
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
        self.draft_model = joblib.load(path("draft_model.joblib"))
        self.stats = DraftStats.from_json(path("draft_stats.json"))
        self.champs = champ_data.load(path("champions.json"))
        self.bank = ModelBank(artifacts_dir)
        self.metrics = self.bank.metrics
        self._insights_path = path("insights.json")

    def champions(self) -> list[dict]:
        return [{"id": c.id, "name": c.name, "icon_url": c.icon_url, "tags": c.tags}
                for c in sorted(self.champs.values(), key=lambda c: c.name)]

    def insights(self) -> dict:
        with open(self._insights_path, encoding="utf-8") as f:
            return json.load(f)

    def meta(self) -> dict:
        return {**self.metrics.get("data", {}), "minutes": list(STATE_MINUTES)}

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
        """Known game-state fields only; missing or null fields are unknown."""
        if req.state is None:
            raise ValidationError(f"Mode '{req.mode}' needs the game state.")
        extra = sorted(set(req.state) - set(STATE_FEATURES))
        if extra:
            raise ValidationError(f"Unknown game-state field(s): {', '.join(extra)}.")
        known = {k: req.state[k] for k in STATE_FEATURES if req.state.get(k) is not None}
        if not known:
            raise ValidationError("Enter at least one game-state value.")
        for k, v in known.items():
            lo, hi = STATE_RANGES[k]
            if not lo <= v <= hi:
                raise ValidationError(f"{k} must be between {lo} and {hi} (got {v}).")
        return known

    # --- prediction ---

    def predict(self, req: PredictRequest) -> PredictResponse:
        # Validate everything the mode needs before predicting anything.
        draft = self._draft_input(req) if req.mode in ("draft", "both") else None
        state = self._state_input(req) if req.mode in ("state", "both") else None

        draft_x = None
        if draft is not None:
            draft_x = pd.DataFrame([draft_features(*draft, self.stats, self.champs)])[DRAFT_FEATURES]

        known = tuple(f for f in STATE_FEATURES if state is not None and f in state)
        if req.mode == "draft":
            m = self.metrics["models"]["draft_lr"]
            fitted = FittedModel(self.draft_model, DRAFT_FEATURES, m["auc"], m["accuracy"],
                                 self.metrics["calibration"]["draft_lr"])
            x = draft_x
        elif req.mode == "state":
            fitted = self.bank.get("state", req.minute, known)
            x = pd.DataFrame([state])[list(known)]
        else:
            logit = float(self.draft_model.decision_function(draft_x)[0])
            features = ("draft_logit", *known)
            fitted = self.bank.get("combined", req.minute, features)
            x = pd.DataFrame([{"draft_logit": logit, **state}])[list(features)]

        baseline, prob, contribs = explain(fitted.model, x)
        omitted = sorted(set(STATE_FEATURES) - set(known)) if state is not None else []
        reasons = pick_reasons([(c.feature, float(x[c.feature].iloc[0]), c.value_pct) for c in contribs])
        draft_only = req.mode == "draft"
        # The verdict's "mostly because" must favour the side that's ahead.
        top = next((r.text for r in reasons if (r.value_pct > 0) == (prob >= 0.5)), None)

        return PredictResponse(
            probability=prob,
            baseline=baseline,
            verdict=verdict(prob, top, draft_only),
            strength=strength(prob, draft_only),
            contributions=[ContributionOut(feature=c.feature, label=LABELS.get(c.feature, c.feature),
                                           value_pct=c.value_pct, logit=c.logit,
                                           value=float(state[c.feature]) if state and c.feature in state else None)
                           for c in contribs],
            reasons=reasons,
            split=self._split(fitted.model, x, baseline, prob) if req.mode == "both" else None,
            omitted=omitted,
            calibration=self._bin(fitted.calibration, prob),
            reliability=self._reliability(req, fitted, omitted),
            minute=req.minute,
        )

    @staticmethod
    def _bin(bins: list[dict], prob: float) -> CalibrationBin | None:
        for b in bins:
            if b["lo"] <= prob < b["hi"] or (prob == 1.0 and b["hi"] == 1.0):
                return CalibrationBin(lo=b["lo"], hi=b["hi"], win_rate=b["win_rate"], n=b["n"])
        return None

    def _reliability(self, req: PredictRequest, fitted: FittedModel, omitted: list[str]) -> Reliability:
        if req.mode == "draft":
            text = (f"Draft-only predictions are close to a coin flip (test AUC {fitted.auc:.2f}); "
                    "read this as a lean, not a call.")
        else:
            patch = self.metrics.get("test_patch")
            text = f"This model was right {fitted.accuracy:.0%} of the time on patch {patch} (AUC {fitted.auc:.2f})."
            if omitted:
                kind = "state" if req.mode == "state" else "combined"
                full_acc = self.metrics["models"][model_name(kind, req.minute)]["accuracy"]
                text += (f" It leaves out {len(omitted)} unknown field{'' if len(omitted) == 1 else 's'};"
                         f" the full model is right {full_acc:.0%}.")
        return Reliability(auc=fitted.auc, accuracy=fitted.accuracy, text=text)

    @staticmethod
    def _split(model, x: pd.DataFrame, baseline: float, prob: float) -> Split:
        """Draft share = change from adding the draft input to the baseline; game state gets the rest."""
        scaler, lr = model[0], model[-1]
        i = list(x.columns).index("draft_logit")
        draft_term = lr.coef_[0][i] * scaler.transform(x)[0][i]
        draft_pct = (sigmoid(lr.intercept_[0] + draft_term) - baseline) * 100
        return Split(draft_pct=draft_pct, state_pct=(prob - baseline) * 100 - draft_pct)
