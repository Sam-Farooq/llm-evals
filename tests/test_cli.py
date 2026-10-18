"""The CLI, driven through main() rather than a subprocess.

Every command in the README has a test here, because a command in a README
that does not run is worse than no README.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_evals.cli import main

ROOT = Path(__file__).resolve().parent.parent
SUITES = str(ROOT / "suites")
THRESHOLDS = str(ROOT / "thresholds.toml")
BASE = ["run", "--suites", SUITES, "--thresholds", THRESHOLDS]


def test_scorers_lists_all_four(capsys):
    assert main(["scorers"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("groundedness")


def test_cases_lists_the_dev_split(capsys):
    assert main(["cases", "--suites", SUITES]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 21
    assert lines[0].startswith("f-01")


def test_the_bare_run_command_from_the_readme_passes(capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert main(["run"]) == 0
    out = capsys.readouterr().out
    assert "verdict: pass" in out
    assert "No overall score" in out


def test_the_lowest_case_is_printed_on_a_run_that_passed(capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert main(["run"]) == 0
    out = capsys.readouterr().out
    # The dev split passes: g-05 at 0.0 is absorbed by five cases at 1.0 and
    # the mean clears 0.8. The zero still has to be on the page.
    assert "verdict: pass" in out
    assert "groundedness  g-05     0.0000  unsupported_numbers=['2']" in out
    # Nothing in format or refusal scored below anything else, so neither
    # names a case.
    assert "format " not in out.split("lowest case per scorer")[1]


def test_the_bands_are_printed_only_when_asked(capsys):
    assert main([*BASE, "--scorers", "groundedness"]) == 0
    assert "0.0-0.2" not in capsys.readouterr().out
    assert main([*BASE, "--scorers", "groundedness", "--distribution"]) == 0
    out = capsys.readouterr().out
    # One case under 0.2, five at exactly 1.0, nothing in between. A per-case
    # floor anywhere above 0.0 cuts g-05 and nothing else.
    assert "groundedness  6  1        0        0        0        0        5" in out


def test_the_json_report_carries_the_lowest_case_and_the_bands(capsys):
    assert main([*BASE, "--json", "--scorers", "groundedness"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lowest"] == [
        {
            "scorer": "groundedness",
            "case_id": "g-05",
            "value": 0.0,
            "reason": "unsupported_numbers=['2']",
        }
    ]
    bands = payload["distribution"][0]
    assert bands["n"] == 6
    assert bands["0.0-0.2"] == 1
    assert bands["1.0"] == 5


def test_the_json_report_is_valid_json_with_no_nan_in_it(capsys):
    assert main([*BASE, "--json", "--scorers", "groundedness"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    rows = {r["scorer"]: r for r in payload["scorers"]}
    assert rows["groundedness"]["value"] == 0.8333
    assert rows["toxicity"]["verdict"] == "skipped"
    assert rows["toxicity"]["value"] is None


def test_out_writes_the_json_report_even_for_a_text_run(tmp_path, capsys):
    out = tmp_path / "nested" / "report.json"
    assert main([*BASE, "--out", str(out)]) == 0
    assert "verdict: pass" in capsys.readouterr().out
    assert json.loads(out.read_text())["passed"] is True


def test_the_holdout_split_needs_release_and_says_why(capsys):
    assert main([*BASE, "--split", "holdout"]) == 2
    err = capsys.readouterr().err
    assert "--release" in err
    assert "second dev set" in err


def test_the_holdout_split_fails_on_the_hedged_refusal(capsys):
    assert main([*BASE, "--split", "holdout", "--release"]) == 1
    out = capsys.readouterr().out
    assert "verdict: fail" in out
    assert "error_type=under_refusal" in out
    assert "hr-01" in out


def test_picking_scorers_is_refused_on_the_holdout_split(capsys):
    assert main([*BASE, "--split", "holdout", "--release", "--scorers", "format"]) == 2
    assert "report all of them" in capsys.readouterr().err


def test_a_word_limit_guardrail_fails_the_run_it_was_added_to(capsys):
    code = main(
        [
            *BASE,
            "--guardrails",
            "--word-limit",
            "6",
            "--scorers",
            "groundedness,refusal,format",
        ]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "verdict: fail" in out
    assert "failed=['json'" in out


def test_blocking_on_toxicity_is_opted_into_one_rule_at_a_time(capsys):
    # The master switch alone adds redaction and nothing that changes a score.
    assert main([*BASE, "--guardrails"]) == 0
    assert "verdict: pass" in capsys.readouterr().out
    # Asking for the blocking rule reaches the recorded classifier, which has
    # no judgement for a response it never saw.
    assert main([*BASE, "--guardrails", "--block-toxicity", "--word-limit", "6"]) == 2
    assert "no toxicity recording" in capsys.readouterr().err


def test_strict_fails_a_run_with_a_scorer_that_never_ran(capsys):
    assert main([*BASE, "--scorers", "groundedness"]) == 0
    assert main([*BASE, "--scorers", "groundedness", "--strict"]) == 1
    assert "skipped: format, refusal, toxicity" in capsys.readouterr().out


def test_an_unknown_scorer_is_a_setup_error_not_a_failed_run(capsys):
    assert main([*BASE, "--scorers", "vibes"]) == 2
    assert "unknown scorer(s): vibes" in capsys.readouterr().err


def test_an_unknown_client_is_a_setup_error(capsys):
    assert main([*BASE, "--client", "llama"]) == 2
    assert "unknown client" in capsys.readouterr().err


def test_a_named_toxicity_recording_that_is_absent_is_a_setup_error(capsys):
    assert main([*BASE, "--toxicity-recording", "/nowhere/tox.jsonl"]) == 2
    assert "no toxicity recording at" in capsys.readouterr().err


def test_a_missing_suite_directory_is_a_setup_error(capsys):
    assert main(["run", "--suites", "/nowhere", "--thresholds", THRESHOLDS]) == 2
    assert "no dev directory" in capsys.readouterr().err
