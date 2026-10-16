"""Evaluation cases, and the two splits they are read from.

A case is one (prompt, context, expectation) triple plus the names of the
scorers that apply to it. The expectation lives in `spec`, keyed by scorer,
because a format rule and a refusal label have nothing in common and forcing
them into one flat schema produces a model with twelve optional fields where
eleven are null on every row.

`extra="forbid"` is load-bearing. A misspelled key in a JSONL line is the
quietest possible way to disable an expectation: the file still parses, the
scorer still runs, and it scores a case that is no longer asking anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEV = "dev"
HOLDOUT = "holdout"
SPLITS = (DEV, HOLDOUT)


class SuiteError(ValueError):
    """A suite file that cannot be read, reported with the line number."""


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    prompt: str
    system: str | None = None
    context: tuple[str, ...] = ()
    scorers: tuple[str, ...]
    spec: dict[str, Any] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()
    note: str | None = None

    @field_validator("scorers")
    @classmethod
    def _at_least_one_scorer(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("a case with no scorers is never evaluated; delete it or name one")
        return v

    def spec_for(self, scorer: str) -> dict[str, Any]:
        """The expectation this case states for one scorer, empty if it states none."""
        value = self.spec.get(scorer, {})
        if not isinstance(value, dict):
            kind = type(value).__name__
            raise SuiteError(f"{self.case_id}: spec.{scorer} must be an object, got {kind}")
        return value


def load_jsonl(path: Path) -> list[EvalCase]:
    """Read one suite file. Blank lines are skipped, `#` comments are not allowed."""
    if not path.is_file():
        raise SuiteError(f"no suite file at {path}")
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SuiteError(f"{path}:{lineno}: {exc.msg}") from exc
            try:
                cases.append(EvalCase(**payload))
            except ValidationError as exc:
                first = exc.errors()[0]
                where = ".".join(str(p) for p in first["loc"]) or "<root>"
                raise SuiteError(f"{path}:{lineno}: {where}: {first['msg']}") from exc
    if not cases:
        raise SuiteError(f"{path} has no cases in it")
    return cases


def load_suite(root: Path, split: str, scorer: str | None = None) -> list[EvalCase]:
    """Every case in one split, optionally narrowed to the cases a scorer applies to.

    Case ids must be unique inside a split and must not be reused across
    splits. A duplicated id makes two different cases indistinguishable in a
    report, and an id shared between dev and holdout makes a leak look like a
    coincidence.
    """
    if split not in SPLITS:
        raise SuiteError(f"unknown split {split!r}; expected one of {', '.join(SPLITS)}")
    directory = root / split
    if not directory.is_dir():
        raise SuiteError(f"no {split} directory under {root}")

    cases: list[EvalCase] = []
    seen: dict[str, Path] = {}
    for path in sorted(directory.glob("*.jsonl")):
        for case in load_jsonl(path):
            if case.case_id in seen:
                raise SuiteError(
                    f"case id {case.case_id!r} appears in both "
                    f"{seen[case.case_id].name} and {path.name}"
                )
            seen[case.case_id] = path
            cases.append(case)
    if not cases:
        raise SuiteError(f"{directory} contains no .jsonl suite files")

    other = HOLDOUT if split == DEV else DEV
    other_dir = root / other
    if other_dir.is_dir():
        overlap = sorted(
            case.case_id
            for path in other_dir.glob("*.jsonl")
            for case in load_jsonl(path)
            if case.case_id in seen
        )
        if overlap:
            raise SuiteError(
                f"{len(overlap)} case id(s) shared between {split} and {other}: "
                + ", ".join(overlap[:3])
            )

    if scorer is not None:
        cases = [c for c in cases if scorer in c.scorers]
    return cases
