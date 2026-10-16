"""What every scorer returns, and the one convention they all follow.

**Higher is always better, and the range is always [0, 1].** Toxicity is the
reason this has to be written down: the natural reading of a toxicity model is
that a big number is bad, so the scorer reports `1 - toxic` and is named for
the property it wants rather than the one it detects. A suite where one scorer
runs the other way is a suite where one threshold is silently inverted, and
nothing about the output looks wrong.

`detail` is where the sub-signals go. A single value is what the threshold
reads; the breakdown is what tells you which half of the scorer moved.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Score:
    value: float
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.value}")


@dataclass(frozen=True)
class ScoredCase:
    """One scorer's verdict on one case, with the response that produced it."""

    case_id: str
    scorer: str
    value: float
    response: str
    detail: Mapping[str, Any] = field(default_factory=dict)
    guarded: bool = False


class Scorer(Protocol):
    """A pure function over (prompt, response, context) plus the case's spec.

    The toxicity scorer is the one that is not pure in the strict sense: it is
    pure given its classifier. That is exactly why the classifier is a
    constructor argument and not a module-level import.
    """

    name: str

    def __call__(
        self,
        prompt: str,
        response: str,
        context: Sequence[str],
        spec: Mapping[str, Any],
    ) -> Score: ...


def clamp(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value
