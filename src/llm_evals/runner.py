"""Walk the cases, get a completion, run the guardrails, score what comes out.

The order matters and is the whole reason this is one function rather than
two. The scorers see what the caller would have seen: the guarded text when a
chain is configured, the raw text when it is not. Scoring the model's output
and then reporting the guarded text is how a suite ends up measuring
something nobody ships.

A run therefore produces one of two things for each case, never both, and the
report says which. To compare them, run twice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_evals.cases import EvalCase
from llm_evals.clients import DEFAULT_MAX_TOKENS, ModelClient
from llm_evals.errors import MissingBackend
from llm_evals.guardrails import GuardrailChain, as_row
from llm_evals.scorers.base import ScoredCase, Scorer
from llm_evals.tracing import NullTracer


@dataclass(frozen=True)
class RunResult:
    scored: tuple[ScoredCase, ...]
    skipped: tuple[str, ...] = field(default_factory=tuple)
    decisions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    completions: int = 0
    guarded: bool = False

    def for_scorer(self, name: str) -> tuple[ScoredCase, ...]:
        return tuple(s for s in self.scored if s.scorer == name)


def run_suite(
    cases: Sequence[EvalCase],
    scorers: Mapping[str, Scorer],
    client: ModelClient,
    *,
    chain: GuardrailChain | None = None,
    tracer: Any = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> RunResult:
    tracer = tracer or NullTracer()
    scored: list[ScoredCase] = []
    skipped: set[str] = set()
    decisions: dict[str, Mapping[str, Any]] = {}
    completions = 0

    for case in cases:
        completion = client.complete(case.prompt, system=case.system, max_tokens=max_tokens)
        completions += 1
        response = completion.text

        if chain is not None:
            decision = chain.apply(case.prompt, response, case.context)
            decisions[case.case_id] = as_row(decision)
            response = decision.text

        with tracer.span(case.case_id, model=completion.model, guarded=chain is not None) as span:
            span.update(input=case.prompt, output=response)
            for name in case.scorers:
                scorer = scorers.get(name)
                if scorer is None:
                    skipped.add(name)
                    continue
                try:
                    score = scorer(case.prompt, response, case.context, case.spec_for(name))
                except MissingBackend:
                    # The extra is not installed. Record it and keep going, so
                    # the other three scorers still produce a report.
                    skipped.add(name)
                    continue
                span.score(name, score.value)
                scored.append(
                    ScoredCase(
                        case_id=case.case_id,
                        scorer=name,
                        value=score.value,
                        response=response,
                        detail=score.detail,
                        guarded=chain is not None,
                    )
                )

    tracer.flush()
    return RunResult(
        scored=tuple(scored),
        skipped=tuple(sorted(skipped)),
        decisions=decisions,
        completions=completions,
        guarded=chain is not None,
    )
