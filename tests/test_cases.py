"""Suite loading, which is the only place a typo can disable an expectation."""
from __future__ import annotations

import json
import socket

import pytest
from pydantic import ValidationError

from llm_evals.cases import DEV, HOLDOUT, EvalCase, SuiteError, load_jsonl, load_suite
from llm_evals.scorers.base import Score


def write(path, *cases):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    return path


def case(case_id, **kw):
    payload = {"case_id": case_id, "prompt": "p", "scorers": ["refusal"]}
    payload.update(kw)
    return payload


def test_a_misspelled_field_is_an_error_not_a_silent_skip(tmp_path):
    path = write(tmp_path / DEV / "a.jsonl", case("c1", contexts=["x"]))
    with pytest.raises(SuiteError) as exc:
        load_jsonl(path)
    assert "contexts" in str(exc.value)
    assert f"{path}:1" in str(exc.value)


def test_a_broken_line_reports_its_line_number(tmp_path):
    path = tmp_path / DEV / "a.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(case("c1")) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(SuiteError, match=rf"{path.name}:2"):
        load_jsonl(path)


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / DEV / "a.jsonl"
    path.parent.mkdir(parents=True)
    body = json.dumps(case("c1")) + "\n\n\n" + json.dumps(case("c2")) + "\n"
    path.write_text(body, encoding="utf-8")
    assert [c.case_id for c in load_jsonl(path)] == ["c1", "c2"]


def test_a_case_with_no_scorers_is_rejected(tmp_path):
    path = write(tmp_path / DEV / "a.jsonl", case("c1", scorers=[]))
    with pytest.raises(SuiteError, match="never evaluated"):
        load_jsonl(path)


def test_duplicate_case_ids_across_files_in_one_split(tmp_path):
    write(tmp_path / DEV / "a.jsonl", case("shared"))
    write(tmp_path / DEV / "b.jsonl", case("shared"))
    with pytest.raises(SuiteError, match="appears in both"):
        load_suite(tmp_path, DEV)


def test_a_case_id_shared_between_the_splits_is_a_leak(tmp_path):
    write(tmp_path / DEV / "a.jsonl", case("c1"), case("c2"))
    write(tmp_path / HOLDOUT / "a.jsonl", case("c2"))
    with pytest.raises(SuiteError, match="shared between dev and holdout"):
        load_suite(tmp_path, DEV)


def test_scorer_filter_narrows_the_set(tmp_path):
    write(
        tmp_path / DEV / "a.jsonl",
        case("c1", scorers=["refusal"]),
        case("c2", scorers=["format", "refusal"]),
        case("c3", scorers=["format"]),
    )
    assert [c.case_id for c in load_suite(tmp_path, DEV, scorer="refusal")] == ["c1", "c2"]
    assert [c.case_id for c in load_suite(tmp_path, DEV, scorer="format")] == ["c2", "c3"]
    assert load_suite(tmp_path, DEV, scorer="toxicity") == []


def test_unknown_split_names_the_ones_that_exist(tmp_path):
    with pytest.raises(SuiteError, match="dev, holdout"):
        load_suite(tmp_path, "test")


def test_spec_for_rejects_a_non_object(tmp_path):
    c = EvalCase(case_id="c1", prompt="p", scorers=("format",), spec={"format": ["json"]})
    with pytest.raises(SuiteError, match="must be an object"):
        c.spec_for("format")
    assert c.spec_for("refusal") == {}


def test_a_case_is_immutable():
    c = EvalCase(case_id="c1", prompt="p", scorers=("refusal",))
    with pytest.raises(ValidationError):
        c.case_id = "c2"


def test_score_outside_the_unit_range_is_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Score(value=1.4)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Score(value=-0.01)
    assert Score(value=1.0).detail == {}


def test_the_network_is_closed_during_the_suite():
    with pytest.raises(RuntimeError, match="opened a socket"):
        socket.create_connection(("example.invalid", 80))
