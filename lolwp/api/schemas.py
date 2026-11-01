"""Request and response models for the prediction API."""
from typing import Literal

from pydantic import BaseModel, StrictInt


class Team(BaseModel):
    top: int
    jungle: int
    mid: int
    bot: int
    support: int


class PredictRequest(BaseModel):
    mode: Literal["draft", "state", "both"]
    minute: Literal[10, 15] = 15
    blue: Team | None = None
    red: Team | None = None
    # A missing key or null means "unknown"; StrictInt rejects true/false.
    state: dict[str, StrictInt | None] | None = None


class ContributionOut(BaseModel):
    feature: str
    label: str
    value_pct: float
    value: float | None  # the raw input, for game-state fields; None for derived draft features
    logit: float


class Reason(BaseModel):
    text: str
    value_pct: float


class Split(BaseModel):
    draft_pct: float
    state_pct: float


class CalibrationBin(BaseModel):
    lo: float
    hi: float
    win_rate: float
    n: int


class Reliability(BaseModel):
    auc: float
    accuracy: float
    text: str


class PredictResponse(BaseModel):
    probability: float
    baseline: float
    verdict: str
    strength: Literal["toss-up", "slight edge", "favored", "strong favorite"]
    contributions: list[ContributionOut]
    reasons: list[Reason]
    split: Split | None = None
    omitted: list[str]
    calibration: CalibrationBin | None
    reliability: Reliability
    minute: int
