"""Reading thresholds.toml, and refusing the ones that cannot mean anything."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from llm_evals.errors import ConfigError

DEFAULT_PATH = Path("thresholds.toml")
Metric = Literal["mean", "min", "pass_rate"]


class ScorerThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: Metric
    floor: float
    case_floor: float = 1.0

    @field_validator("floor", "case_floor")
    @classmethod
    def _in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be in [0, 1], got {v}")
        return v


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scorers: dict[str, ScorerThreshold] = {}
    rates: dict[str, float] = {}

    @model_validator(mode="after")
    def _not_empty(self) -> Thresholds:
        if not self.scorers:
            raise ValueError("no scorer thresholds; a run with no floor cannot fail")
        return self

    @field_validator("rates")
    @classmethod
    def _rates_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for name, ceiling in v.items():
            if not 0.0 <= ceiling <= 1.0:
                raise ValueError(f"rate {name} must be in [0, 1], got {ceiling}")
        return v

    def for_scorer(self, name: str) -> ScorerThreshold | None:
        return self.scorers.get(name)


def load_thresholds(path: Path = DEFAULT_PATH) -> Thresholds:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"no threshold file at {path}")
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    try:
        return Thresholds(**payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "<root>"
        raise ConfigError(f"{path}: {where}: {first['msg']}") from exc
