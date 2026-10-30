"""Request and response models for the prediction API."""
from typing import Literal

from pydantic import BaseModel


class Team(BaseModel):
    top: int
    jungle: int
    mid: int
    bot: int
    support: int


class PredictRequest(BaseModel):
    mode: Literal["draft", "state", "both"]
    blue: Team | None = None
    red: Team | None = None
    state: dict[str, int] | None = None


class ContributionOut(BaseModel):
    feature: str
    label: str
    value_pct: float


class Split(BaseModel):
    draft_pct: float
    state_pct: float


class PredictResponse(BaseModel):
    probability: float
    baseline: float
    contributions: list[ContributionOut]
    reasons: list[str]
    split: Split | None = None
