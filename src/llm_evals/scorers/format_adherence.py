"""Does the response obey the shape that was asked for.

All or nothing. A response that satisfies three of four format rules scores
0.0, not 0.75, because the consumer on the other end is a parser and it reads
the one rule that broke. Partial credit here produces a number that climbs
while every downstream call still fails, which is the exact thing the suite
exists to catch.

The per-rule outcome is in `detail["checks"]`, and `detail["partial"]` carries
the fraction for the person reading a failure. It is not the score, and
nothing thresholds it.

Rules a case can state under `spec.format`:

    json                must parse as JSON
    json_only           must be JSON with no prose wrapped around it
    required_keys       dot paths that must exist, e.g. "result.status"
    forbidden_keys      dot paths that must not exist
    max_words           an upper word count
    min_words           a lower word count
    max_lines           an upper newline count
    pattern             a regex that must match somewhere
    forbidden_pattern   a regex that must not match anywhere

`json` is the tolerant check: a fenced block or a payload with prose around
it still counts, because the model produced the right data. `json_only` is the
strict one: the whole response, trimmed, has to be the object or the array.

Keeping them apart is what makes the pair useful. "Here is the JSON you asked
for: {...}" has the right data in the wrong envelope, and those are two
different things to go and fix. Collapsing them into one check means the
report cannot tell you which.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from llm_evals.errors import ConfigError
from llm_evals.scorers.base import Score

NAME = "format"

_RULES = (
    "json",
    "json_only",
    "required_keys",
    "forbidden_keys",
    "max_words",
    "min_words",
    "max_lines",
    "pattern",
    "forbidden_pattern",
)


def _dig(payload: Any, path: str) -> tuple[bool, Any]:
    current = payload
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current


_FENCED = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def _loads(text: str) -> tuple[bool, Any, str]:
    try:
        return True, json.loads(text), ""
    except json.JSONDecodeError as exc:
        return False, None, f"{exc.msg} at char {exc.pos}"


def _extract_json(response: str) -> tuple[bool, Any, str, bool]:
    """Return (parsed_at_all, payload, why_not, the_whole_response_was_the_payload)."""
    text = response.strip()
    ok, payload, why = _loads(text)
    if ok:
        return True, payload, "", True

    candidates: list[str] = []
    fenced = _FENCED.search(response)
    if fenced:
        candidates.append(fenced.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        ok, payload, _ = _loads(candidate)
        if ok:
            return True, payload, "", False
    return False, None, why, False


def format_adherence(
    prompt: str,
    response: str,
    context: Sequence[str],
    spec: Mapping[str, Any] | None = None,
) -> Score:
    spec = spec or {}
    unknown = sorted(set(spec) - set(_RULES))
    if unknown:
        raise ConfigError(f"unknown format rule(s): {', '.join(unknown)}")
    if not spec:
        raise ConfigError("format needs at least one rule under spec.format")

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, note: str = "") -> None:
        checks.append({"check": name, "passed": passed, "note": note})

    wants_json = bool(
        spec.get("json")
        or spec.get("json_only")
        or spec.get("required_keys")
        or spec.get("forbidden_keys")
    )
    parsed: Any = None
    parse_ok = False
    whole = False
    if wants_json:
        parse_ok, parsed, why, whole = _extract_json(response)
        if spec.get("json"):
            record("json", parse_ok, why)
        if spec.get("json_only"):
            bare = whole and isinstance(parsed, (dict, list))
            if whole and not isinstance(parsed, (dict, list)):
                note = f"parsed as {type(parsed).__name__}, not an object or array"
            elif not whole:
                note = "prose or a fence around the payload"
            else:
                note = ""
            record("json_only", bare, note)

    for path in spec.get("required_keys", ()):
        if not parse_ok:
            record(f"required_keys:{path}", False, "json did not parse")
            continue
        found, _ = _dig(parsed, path)
        record(f"required_keys:{path}", found, "" if found else "missing")

    for path in spec.get("forbidden_keys", ()):
        if not parse_ok:
            record(f"forbidden_keys:{path}", False, "json did not parse")
            continue
        found, _ = _dig(parsed, path)
        record(f"forbidden_keys:{path}", not found, "present" if found else "")

    words = len(response.split())
    if "max_words" in spec:
        limit = int(spec["max_words"])
        record("max_words", words <= limit, f"{words} words, limit {limit}")
    if "min_words" in spec:
        floor = int(spec["min_words"])
        record("min_words", words >= floor, f"{words} words, floor {floor}")
    if "max_lines" in spec:
        lines = len(response.strip().splitlines())
        limit = int(spec["max_lines"])
        record("max_lines", lines <= limit, f"{lines} lines, limit {limit}")

    if "pattern" in spec:
        hit = re.search(str(spec["pattern"]), response) is not None
        record("pattern", hit, "" if hit else f"no match for {spec['pattern']!r}")
    if "forbidden_pattern" in spec:
        hit = re.search(str(spec["forbidden_pattern"]), response) is not None
        note = f"matched {spec['forbidden_pattern']!r}" if hit else ""
        record("forbidden_pattern", not hit, note)

    if not checks:
        raise ConfigError("format rules were present but none produced a check")

    passed = sum(1 for c in checks if c["passed"])
    return Score(
        1.0 if passed == len(checks) else 0.0,
        {
            "checks": checks,
            "partial": round(passed / len(checks), 4),
            "failed": [c["check"] for c in checks if not c["passed"]],
        },
    )


class FormatAdherence:
    name = NAME

    def __call__(
        self,
        prompt: str,
        response: str,
        context: Sequence[str],
        spec: Mapping[str, Any] | None = None,
    ) -> Score:
        return format_adherence(prompt, response, context, spec)
