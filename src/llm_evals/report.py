"""Per-scorer results, and no overall number.

### Why there is no aggregate score

Averaging the scorers produces a number that cannot regress. Groundedness and
format adherence do not trade off against each other in reality, but in a mean
they do, and the commonest real change makes them move in opposite directions
at once: turning a blocking guardrail on raises toxicity and takes format
adherence down, because a blocked response is a refusal sentence and not JSON.
A blended figure moves by a point and the run still passes.

The directional rates are the same argument one level down. Refusal accuracy
is one number over two opposite errors, so this report thresholds
`over_refusal` and `under_refusal` separately as ceilings, and a run can fail
on either while accuracy looks fine.

The only aggregate here is the exit code, and it is a conjunction: every
scorer must clear its floor and every rate must stay under its ceiling. There
is nothing to average.

### A skipped scorer is reported, never dropped

If a scorer could not run, for a missing extra or an absent recording, it
appears with `verdict = "skipped"`. A report that silently omits it is the
regression-hiding failure this file exists to avoid, so `strict=True` turns a
skip into a failure.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from llm_evals.scorers.base import ScoredCase
from llm_evals.scorers.refusal import NAME as REFUSAL
from llm_evals.scorers.refusal import OVER_REFUSAL, UNDER_REFUSAL
from llm_evals.thresholds import Thresholds

PASS = "pass"
FAIL = "fail"
SKIPPED = "skipped"

SCORER_COLUMNS = (
    "scorer",
    "n",
    "mean",
    "min",
    "pass_rate",
    "metric",
    "value",
    "floor",
    "verdict",
    "worst_case",
)
RATE_COLUMNS = ("rate", "value", "ceiling", "n", "verdict")


@dataclass(frozen=True)
class Report:
    scorers: pd.DataFrame
    rates: pd.DataFrame
    skipped: tuple[str, ...] = field(default_factory=tuple)
    strict: bool = False

    @property
    def passed(self) -> bool:
        if FAIL in set(self.scorers["verdict"]) or FAIL in set(self.rates["verdict"]):
            return False
        return not (self.strict and self.skipped)

    def to_text(self) -> str:
        lines = [_table(self.scorers, SCORER_COLUMNS)]
        if not self.rates.empty:
            lines.append("")
            lines.append(_table(self.rates, RATE_COLUMNS))
        if self.skipped:
            lines.append("")
            lines.append("skipped: " + ", ".join(self.skipped))
        lines.append("")
        lines.append("No overall score. Every floor and every ceiling has to hold.")
        lines.append(f"verdict: {PASS if self.passed else FAIL}")
        return "\n".join(lines)

    def to_json(self) -> str:
        payload = {
            "scorers": _records(self.scorers),
            "rates": _records(self.rates),
            "skipped": list(self.skipped),
            "passed": self.passed,
        }
        return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    present = [c for c in columns if c in frame.columns]
    if frame.empty:
        return "  (no rows)"
    widths = {
        c: max(len(c), *(len(_cell(v)) for v in frame[c])) for c in present
    }
    header = "  ".join(c.ljust(widths[c]) for c in present)
    rule = "  ".join("-" * widths[c] for c in present)
    rows = [
        "  ".join(_cell(row[c]).ljust(widths[c]) for c in present)
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, rule, *rows])


def _cell(value: object) -> str:
    if _missing(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _missing(value: object) -> bool:
    # A skipped row has no numbers, and pandas turns those into NaN in a float
    # column. Printing "nan" next to a real 0.0000 is how a skip gets read as
    # a score, so both print as a dash and neither reaches the JSON.
    return value is None or (isinstance(value, float) and value != value)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {k: (None if _missing(v) else v) for k, v in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _metric_value(values: Sequence[float], metric: str, case_floor: float) -> float:
    if metric == "mean":
        return sum(values) / len(values)
    if metric == "min":
        return min(values)
    if metric == "pass_rate":
        return sum(1 for v in values if v >= case_floor) / len(values)
    raise ValueError(f"unknown metric {metric!r}")


def refusal_rates(scored: Iterable[ScoredCase]) -> dict[str, tuple[float, int]]:
    """The two refusal directions as rates over the refusal cases."""
    cases = [s for s in scored if s.scorer == REFUSAL]
    if not cases:
        return {}
    total = len(cases)
    counts = {OVER_REFUSAL: 0, UNDER_REFUSAL: 0}
    for case in cases:
        error = case.detail.get("error_type")
        if error in counts:
            counts[error] += 1
    return {
        "over_refusal": (counts[OVER_REFUSAL] / total, total),
        "under_refusal": (counts[UNDER_REFUSAL] / total, total),
    }


def build_report(
    scored: Sequence[ScoredCase],
    thresholds: Thresholds,
    *,
    skipped: Iterable[str] = (),
    strict: bool = False,
) -> Report:
    by_scorer: dict[str, list[ScoredCase]] = {}
    for case in scored:
        by_scorer.setdefault(case.scorer, []).append(case)

    skipped = tuple(sorted(set(skipped) | (set(thresholds.scorers) - set(by_scorer))))

    rows = []
    for name in sorted(by_scorer):
        cases = by_scorer[name]
        values = [c.value for c in cases]
        threshold = thresholds.for_scorer(name)
        worst = min(cases, key=lambda c: c.value)
        # pass_rate counts the cases that clear the bar this scorer is judged
        # against: case_floor when the metric is pass_rate, the floor itself
        # otherwise. A fixed 1.0 bar would print 0.0000 beside a passing
        # toxicity row, which reads as the opposite of what happened.
        bar = 1.0
        if threshold is not None:
            bar = threshold.case_floor if threshold.metric == "pass_rate" else threshold.floor
        row = {
            "scorer": name,
            "n": len(cases),
            "mean": round(sum(values) / len(values), 4),
            "min": round(min(values), 4),
            "pass_rate": round(sum(1 for v in values if v >= bar) / len(values), 4),
            "worst_case": worst.case_id,
        }
        if threshold is None:
            # Scored but not thresholded. Reported, not silently passed.
            row.update(metric="none", value=None, floor=None, verdict=SKIPPED)
        else:
            value = _metric_value(values, threshold.metric, threshold.case_floor)
            row.update(
                metric=threshold.metric,
                value=round(value, 4),
                floor=threshold.floor,
                verdict=PASS if value >= threshold.floor else FAIL,
            )
        rows.append(row)

    for name in skipped:
        rows.append(
            {
                "scorer": name,
                "n": 0,
                "mean": None,
                "min": None,
                "pass_rate": None,
                "metric": thresholds.scorers[name].metric if name in thresholds.scorers else "none",
                "value": None,
                "floor": thresholds.scorers[name].floor if name in thresholds.scorers else None,
                "verdict": SKIPPED,
                "worst_case": None,
            }
        )

    rate_rows = []
    measured = refusal_rates(scored)
    for name, ceiling in sorted(thresholds.rates.items()):
        value, total = measured.get(name, (None, 0))
        rate_rows.append(
            {
                "rate": name,
                "value": None if value is None else round(value, 4),
                "ceiling": ceiling,
                "n": total,
                "verdict": SKIPPED if value is None else (PASS if value <= ceiling else FAIL),
            }
        )

    return Report(
        scorers=pd.DataFrame(rows, columns=list(SCORER_COLUMNS)),
        rates=pd.DataFrame(rate_rows, columns=list(RATE_COLUMNS)),
        skipped=skipped,
        strict=strict,
    )


# The first of these a scorer's detail carries is the line worth printing.
_REASON_KEYS = (
    "error_type",
    "failed",
    "unsupported_numbers",
    "unsupported_quotes",
    "unsupported_sentences",
    "worst_label",
    "reason",
)


def reason(detail: Mapping[str, object]) -> str:
    for key in _REASON_KEYS:
        value = detail.get(key)
        if value:
            return f"{key}={value}"
    return ""


def failed_scorers(report: Report) -> set[str]:
    frame = report.scorers
    return set(frame.loc[frame["verdict"] == FAIL, "scorer"])


def failing_cases(
    scored: Sequence[ScoredCase],
    limit: int = 5,
    scorers: Iterable[str] | None = None,
) -> list[Mapping[str, object]]:
    """The lowest-scoring cases, for the lines printed under a failed run.

    `scorers` narrows it to the ones that actually failed. Without that, a
    toxicity case at 0.96 is listed under a run that failed on format, and the
    first line a reader sees points at the wrong scorer.
    """
    allowed = None if scorers is None else set(scorers)
    pool = [
        c for c in scored if c.value < 1.0 and (allowed is None or c.scorer in allowed)
    ]
    ordered = sorted(pool, key=lambda c: (c.value, c.scorer, c.case_id))
    return [
        {
            "case_id": c.case_id,
            "scorer": c.scorer,
            "value": round(c.value, 4),
            "reason": reason(c.detail),
            "detail": dict(c.detail),
        }
        for c in ordered[:limit]
    ]
