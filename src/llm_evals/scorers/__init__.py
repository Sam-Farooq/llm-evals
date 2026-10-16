"""The scorer registry.

Toxicity is always registered, even with no classifier and no extra
installed. It fails at call time with `MissingBackend`, the runner records it
as skipped, and the report prints the skip. Leaving it out of the registry
when the extra is absent would mean a suite quietly scoring three properties
where the case file asks for four.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from llm_evals.errors import ConfigError
from llm_evals.scorers.base import Score, ScoredCase, Scorer
from llm_evals.scorers.format_adherence import NAME as FORMAT
from llm_evals.scorers.format_adherence import FormatAdherence
from llm_evals.scorers.groundedness import NAME as GROUNDEDNESS
from llm_evals.scorers.groundedness import Groundedness
from llm_evals.scorers.refusal import NAME as REFUSAL
from llm_evals.scorers.refusal import RefusalCorrectness
from llm_evals.scorers.toxicity import NAME as TOXICITY
from llm_evals.scorers.toxicity import Classifier, Toxicity

ALL = (GROUNDEDNESS, REFUSAL, FORMAT, TOXICITY)

__all__ = [
    "ALL",
    "FORMAT",
    "GROUNDEDNESS",
    "REFUSAL",
    "TOXICITY",
    "Score",
    "ScoredCase",
    "Scorer",
    "build_scorers",
]


def build_scorers(
    names: Iterable[str] | None = None,
    *,
    toxicity_classifier: Classifier | None = None,
) -> Mapping[str, Scorer]:
    wanted = tuple(names) if names is not None else ALL
    unknown = [n for n in wanted if n not in ALL]
    if unknown:
        raise ConfigError(f"unknown scorer(s): {', '.join(unknown)}; known: {', '.join(ALL)}")
    built: dict[str, Scorer] = {}
    for name in wanted:
        if name == GROUNDEDNESS:
            built[name] = Groundedness()
        elif name == REFUSAL:
            built[name] = RefusalCorrectness()
        elif name == FORMAT:
            built[name] = FormatAdherence()
        elif name == TOXICITY:
            built[name] = Toxicity(toxicity_classifier)
    return built
