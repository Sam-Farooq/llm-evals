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

### The lowest case is named on a passing run

A mean clears its floor with a zero under it: five cases at 1.0 and one at
0.0 average to 0.8333 and pass a floor of 0.8. The per-case lines in the CLI
only print when the run has already failed, so on the passing run the zero
reached nobody. `lowest_cases` is printed either way, with the reason beside
it, and `distribution` puts the per-case counts in bands so a per-case floor
can be set from what the split actually scores rather than chosen.

A tie is not a worst case. `worst_case` is empty when no case scored below
another, because a column that points at a case on every row, including the
rows where nothing went wrong, is one a reader learns to skip.

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
LOWEST_COLUMNS = ("scorer", "case_id", "value", "reason")

# Left-closed bands, with an exact 1.0 of its own. Most cases in a healthy
# split sit at exactly 1.0, and folding them into 0.8-1.0 hides whether the
# tail is one case or forty.
BANDS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))
BAND_COLUMNS = (*(f"{lo:.1f}-{hi:.1f}" for lo, hi in BANDS), "1.0")
DISTRIBUTION_COLUMNS = ("scorer", "n", *BAND_COLUMNS)


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([], columns=list(columns))


@dataclass(frozen=True)
class Report:
    scorers: pd.DataFrame
    rates: pd.DataFrame
    skipped: tuple[str, ...] = field(default_factory=tuple)
    strict: bool = False
    lowest: pd.DataFrame = field(default_factory=lambda: _empty(LOWEST_COLUMNS))
    distribution: pd.DataFrame = field(default_factory=lambda: _empty(DISTRIBUTION_COLUMNS))

    @property
    def passed(self) -> bool:
        if FAIL in set(self.scorers["verdict"]) or FAIL in set(self.rates["verdict"]):
            return False
        return not (self.strict and self.skipped)

    def to_text(self, *, show_distribution: bool = False) -> str:
        lines = [_table(self.scorers, SCORER_COLUMNS)]
        if not self.rates.empty:
            lines.append("")
            lines.append(_table(self.rates, RATE_COLUMNS))
        lines.append("")
        if self.lowest.empty:
            lines.append("lowest: nothing scored below another case in its scorer.")
        else:
            lines.append("lowest case per scorer, reported whether or not the run failed:")
            lines.append(_table(self.lowest, LOWEST_COLUMNS))
        if show_distribution:
            lines.append("")
            lines.append("cases per band, left-closed, with an exact 1.0 of its own:")
            lines.append(_table(self.distribution, DISTRIBUTION_COLUMNS))
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
            "lowest": _records(self.lowest),
            # Always in the JSON, flag or not: a machine reading the report
            # has no way to ask for it again.
            "distribution": _records(self.distribution),
            "skipped": list(self.skipped),
            "passed": self.passed,
        }
        return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    present = [c for c in columns if c in frame.columns]
    if frame.empty:
        return "  (no rows)"
    widths = {c: max(len(c), *(len(_cell(v)) for v in frame[c])) for c in present}
    header = "  ".join(c.ljust(widths[c]) for c in present)
    rule = "  ".join("-" * widths[c] for c in present)
    rows = [
        "  ".join(_cell(row[c]).ljust(widths[c]) for c in present) for _, row in frame.iterrows()
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


def _worst(cases: Sequence[ScoredCase]) -> ScoredCase | None:
    """The case that scored below the others, or None when none of them did."""
    worst = min(cases, key=lambda c: c.value)
    return None if worst.value >= max(c.value for c in cases) else worst


def lowest_cases(scored: Iterable[ScoredCase]) -> list[dict[str, object]]:
    """The worst case per scorer, with the reason, for the scorers with a spread.

    `failing_cases` answers "what broke this run" and only runs once the run
    has broken. This answers "what is the worst thing in here", which is the
    question a passing mean suppresses.
    """
    by_scorer: dict[str, list[ScoredCase]] = {}
    for case in scored:
        by_scorer.setdefault(case.scorer, []).append(case)
    rows = []
    for name in sorted(by_scorer):
        worst = _worst(by_scorer[name])
        if worst is None:
            continue
        rows.append(
            {
                "scorer": name,
                "case_id": worst.case_id,
                "value": round(worst.value, 4),
                "reason": reason(worst.detail),
            }
        )
    return rows


def _band(value: float) -> str:
    for (lo, hi), label in zip(BANDS, BAND_COLUMNS[:-1], strict=True):
        if lo <= value < hi:
            return label
    return BAND_COLUMNS[-1]


def distribution(scored: Iterable[ScoredCase]) -> pd.DataFrame:
    """Cases per score band, per scorer.

    A per-case floor has to come from somewhere. Picked without this it is the
    same guess as the mean floor it is meant to back up; with it, the counts
    say how many cases a candidate floor would cut before anyone gates on it.
    """
    counts: dict[str, dict[str, int]] = {}
    for case in scored:
        if not 0.0 <= case.value <= 1.0:
            # A value outside the range belongs in no band, and a row whose
            # bands sum to less than its own n is a table nobody re-adds.
            raise ValueError(
                f"{case.scorer} case {case.case_id}: score must be in [0, 1], got {case.value}"
            )
        row = counts.setdefault(case.scorer, dict.fromkeys(BAND_COLUMNS, 0))
        row[_band(case.value)] += 1
    rows = [
        {"scorer": name, "n": sum(counts[name].values()), **counts[name]} for name in sorted(counts)
    ]
    return pd.DataFrame(rows, columns=list(DISTRIBUTION_COLUMNS))


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
        worst = _worst(cases)
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
            "worst_case": None if worst is None else worst.case_id,
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
        lowest=pd.DataFrame(lowest_cases(scored), columns=list(LOWEST_COLUMNS)),
        distribution=distribution(scored),
    )


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
    pool = [c for c in scored if c.value < 1.0 and (allowed is None or c.scorer in allowed)]
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
