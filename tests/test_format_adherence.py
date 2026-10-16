"""Format adherence. The interesting assertion is that partial is not the score."""
from __future__ import annotations

import pytest

from llm_evals.errors import ConfigError
from llm_evals.scorers.format_adherence import format_adherence


def score(response, spec):
    return format_adherence("p", response, (), spec)


def test_a_clean_payload_passes_every_rule():
    s = score(
        '{"status": "ok", "result": {"count": 3}}',
        {"json": True, "json_only": True, "required_keys": ["status", "result.count"]},
    )
    assert s.value == 1.0
    assert s.detail["partial"] == 1.0
    assert s.detail["failed"] == []


def test_three_of_four_rules_is_still_zero():
    s = score(
        'Here is the JSON you asked for: {"status": "ok", "result": {"count": 3}}',
        {"json": True, "json_only": True, "required_keys": ["status", "result.count"]},
    )
    # json_only is the only rule that broke. Every caller that does not strip
    # the prefix by hand still fails, so the score is zero and not 0.75.
    assert s.detail["partial"] == 0.75
    assert s.value == 0.0
    assert s.detail["failed"] == ["json_only"]


def test_a_fenced_payload_satisfies_json_but_not_json_only():
    s = score('```json\n{"status": "ok"}\n```', {"json": True, "json_only": True})
    assert s.detail["failed"] == ["json_only"]
    assert s.value == 0.0
    note = next(c["note"] for c in s.detail["checks"] if c["check"] == "json_only")
    assert note == "prose or a fence around the payload"


def test_a_bare_scalar_parses_but_is_not_a_payload():
    s = score("42", {"json": True, "json_only": True})
    assert s.detail["failed"] == ["json_only"]
    note = next(c["note"] for c in s.detail["checks"] if c["check"] == "json_only")
    assert note == "parsed as int, not an object or array"


def test_nothing_json_shaped_fails_both_rules():
    s = score("not json at all", {"json": True, "json_only": True})
    assert s.detail["failed"] == ["json", "json_only"]
    assert s.value == 0.0


def test_a_missing_nested_key_is_named_in_the_failure():
    s = score('{"result": {}}', {"json": True, "required_keys": ["result.count"]})
    assert s.detail["failed"] == ["required_keys:result.count"]
    note = next(c["note"] for c in s.detail["checks"] if c["check"] == "required_keys:result.count")
    assert note == "missing"


def test_key_checks_report_the_parse_failure_rather_than_missing():
    s = score("not json at all", {"json": True, "required_keys": ["status"]})
    notes = {c["check"]: c["note"] for c in s.detail["checks"]}
    assert notes["required_keys:status"] == "json did not parse"


def test_a_forbidden_key_present_fails():
    s = score(
        '{"answer": 1, "chain_of_thought": "..."}',
        {"json": True, "forbidden_keys": ["chain_of_thought"]},
    )
    assert s.value == 0.0
    assert s.detail["failed"] == ["forbidden_keys:chain_of_thought"]


def test_word_and_line_limits_report_the_actual_count():
    s = score("one two three four five", {"max_words": 3})
    assert s.value == 0.0
    assert s.detail["checks"][0]["note"] == "5 words, limit 3"
    assert score("one two three", {"max_words": 3, "min_words": 2}).value == 1.0
    assert score("a\nb\nc", {"max_lines": 2}).value == 0.0


def test_pattern_and_forbidden_pattern_are_opposites():
    assert score("ticket AB-1234 raised", {"pattern": r"[A-Z]{2}-\d{4}"}).value == 1.0
    assert score("ticket raised", {"pattern": r"[A-Z]{2}-\d{4}"}).value == 0.0
    assert score("as an AI language model", {"forbidden_pattern": r"as an AI"}).value == 0.0
    assert score("the answer is 4", {"forbidden_pattern": r"as an AI"}).value == 1.0


def test_an_unknown_rule_is_a_config_error_not_an_ignored_line():
    with pytest.raises(ConfigError, match="max_tokens"):
        score("{}", {"json": True, "max_tokens": 50})


def test_no_rules_at_all_is_a_config_error():
    with pytest.raises(ConfigError, match="at least one rule"):
        score("{}", {})


def test_a_top_level_array_counts_as_json_only():
    assert score("[1, 2, 3]", {"json": True, "json_only": True}).value == 1.0
