"""The command line. Thin on purpose: it parses flags and calls the library.

Exit codes:

    0   every floor held and every ceiling held
    1   a scorer or a rate failed
    2   the run could not be set up: a bad flag, a missing file, an absent extra

1 and 2 are kept apart because a suite that failed and a suite that never ran
look identical in CI otherwise, and the second one is not a model regression.

### Running the held-out split needs `--release`

A held-out set you run on every change is a second dev set with a longer name.
The flag is not a safety feature, it is a speed bump with the reason written
on it. `--scorers` is refused on that split for the same reason: choosing
which scorers to look at after the fact is how a held-out number becomes a
number you tuned.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_evals.cases import DEV, HOLDOUT, SPLITS, load_suite
from llm_evals.clients import DEFAULT_MAX_TOKENS, RecordedClient, build_client
from llm_evals.errors import ConfigError, MissingBackend
from llm_evals.guardrails import default_chain
from llm_evals.report import build_report, failed_scorers, failing_cases
from llm_evals.runner import run_suite
from llm_evals.scorers import ALL, build_scorers
from llm_evals.scorers.toxicity import RecordedClassifier, Toxicity
from llm_evals.thresholds import load_thresholds
from llm_evals.tracing import get_tracer

DEFAULT_SUITES = Path("suites")
DEFAULT_THRESHOLDS = Path("thresholds.toml")

WHAT_EACH_SCORER_MEASURES = {
    "groundedness": "numbers, quoted spans and sentences against the supplied context",
    "refusal": "whether the response declined, against whether it should have",
    "format": "JSON shape, required keys, length caps and patterns",
    "toxicity": "1 minus the worst toxic label from an injected classifier",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-evals", description=__doc__.splitlines()[0])
    subs = parser.add_subparsers(dest="command", required=True)

    run = subs.add_parser("run", help="score a split and report per scorer")
    run.add_argument("--suites", type=Path, default=DEFAULT_SUITES)
    run.add_argument("--split", choices=SPLITS, default=DEV)
    run.add_argument("--release", action="store_true", help="required to run the held-out split")
    run.add_argument("--client", default="recorded", help="recorded, anthropic or openai")
    run.add_argument("--recording", type=Path, default=None)
    run.add_argument("--toxicity-recording", default=None, help="a path, or 'none' for the model")
    run.add_argument("--model", default=None)
    run.add_argument("--scorers", default=None, help="comma separated subset")
    run.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--guardrails", action="store_true", help="enable the chain (redaction only)")
    run.add_argument("--block-toxicity", action="store_true")
    run.add_argument("--word-limit", type=int, default=None)
    run.add_argument("--require-json", action="store_true")
    run.add_argument("--grounding-floor", type=float, default=None)
    run.add_argument("--toxicity-floor", type=float, default=0.5)
    run.add_argument("--strict", action="store_true", help="a skipped scorer fails the run")
    run.add_argument("--json", action="store_true")
    run.add_argument("--out", type=Path, default=None)

    record = subs.add_parser("record", help="capture completions for offline replay")
    record.add_argument("--suites", type=Path, default=DEFAULT_SUITES)
    record.add_argument("--split", choices=SPLITS, default=DEV)
    record.add_argument("--client", default="anthropic")
    record.add_argument("--model", default=None)
    record.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    record.add_argument("--out", type=Path, required=True)
    record.add_argument("--toxicity-out", type=Path, default=None)

    cases = subs.add_parser("cases", help="list the cases in a split")
    cases.add_argument("--suites", type=Path, default=DEFAULT_SUITES)
    cases.add_argument("--split", choices=SPLITS, default=DEV)

    subs.add_parser("scorers", help="list the scorers and what each one measures")
    return parser


def _toxicity_classifier(suites: Path, split: str, flag: str | None):
    if flag == "none":
        return None
    path = Path(flag) if flag else suites / "recordings" / f"{split}-toxicity.jsonl"
    if not path.is_file():
        if flag:
            raise ConfigError(f"no toxicity recording at {path}")
        return None
    return RecordedClassifier.from_jsonl(path)


def _command_run(args: argparse.Namespace) -> int:
    if args.split == HOLDOUT and not args.release:
        raise ConfigError(
            "running the held-out split needs --release. A set you measure on every "
            "change is a second dev set, so this is deliberate friction"
        )
    if args.split == HOLDOUT and args.scorers:
        raise ConfigError("--scorers is refused on the held-out split; report all of them")

    names = [n.strip() for n in args.scorers.split(",")] if args.scorers else None
    cases = load_suite(args.suites, args.split)
    thresholds = load_thresholds(args.thresholds)
    recording = args.recording or args.suites / "recordings" / f"{args.split}.jsonl"
    client = build_client(args.client, model=args.model, recording=recording)
    classifier = _toxicity_classifier(args.suites, args.split, args.toxicity_recording)
    scorers = build_scorers(names, toxicity_classifier=classifier)

    chain = None
    if args.guardrails:
        # Every rule past redaction is opted into one at a time. A master
        # switch that silently turns on a blocking rule changes what the suite
        # measures, which is the opposite of what a flag named --guardrails
        # should do.
        if args.block_toxicity and classifier is None:
            raise ConfigError("--block-toxicity needs a toxicity recording or the toxicity extra")
        chain = default_chain(
            toxicity=Toxicity(classifier) if args.block_toxicity else None,
            toxicity_floor=args.toxicity_floor,
            grounding_floor=args.grounding_floor,
            word_limit=args.word_limit,
            require_json=args.require_json,
        )

    result = run_suite(
        cases,
        scorers,
        client,
        chain=chain,
        tracer=get_tracer(),
        max_tokens=args.max_tokens,
    )
    report = build_report(result.scored, thresholds, skipped=result.skipped, strict=args.strict)

    text = report.to_json() if args.json else report.to_text()
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report.to_json(), encoding="utf-8")
    if not report.passed and not args.json:
        for failure in failing_cases(result.scored, scorers=failed_scorers(report)):
            print(
                f"  {failure['case_id']:8} {failure['scorer']:14} "
                f"{failure['value']:<8} {failure['reason']}"
            )
    return 0 if report.passed else 1


def _command_record(args: argparse.Namespace) -> int:
    cases = load_suite(args.suites, args.split)
    client = build_client(args.client, model=args.model)
    rows = []
    for case in cases:
        completion = client.complete(case.prompt, system=case.system, max_tokens=args.max_tokens)
        row = {"prompt": case.prompt, "response": completion.text}
        if case.system:
            row["system"] = case.system
        rows.append(row)
    written = RecordedClient.write_jsonl(args.out, rows)
    print(f"{written} completions to {args.out}")

    if args.toxicity_out:
        scorer = Toxicity()
        labels = [
            {"text": row["response"], "labels": list(scorer.classifier(row["response"]))}
            for row in rows
        ]
        count = RecordedClient.write_jsonl(args.toxicity_out, labels)
        print(f"{count} judgements to {args.toxicity_out}")
    return 0


def _command_cases(args: argparse.Namespace) -> int:
    for case in load_suite(args.suites, args.split):
        tags = ",".join(case.tags) or "-"
        print(f"{case.case_id:8} {','.join(case.scorers):14} {tags:20} {case.prompt[:60]}")
    return 0


def _command_scorers(_: argparse.Namespace) -> int:
    for name in ALL:
        print(f"{name:14} {WHAT_EACH_SCORER_MEASURES[name]}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {
        "run": _command_run,
        "record": _command_record,
        "cases": _command_cases,
        "scorers": _command_scorers,
    }
    try:
        return handlers[args.command](args)
    except (ConfigError, MissingBackend, ValueError) as exc:
        print(f"llm-evals: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
