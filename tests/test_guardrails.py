"""The chain, including the two rules that cannot be in it together."""
from __future__ import annotations

import json

import pytest

from llm_evals.errors import ConfigError
from llm_evals.guardrails import (
    ALLOW,
    BLOCK,
    REWRITE,
    BlockOnToxicity,
    BlockOnUngrounded,
    BlockSystemPromptLeak,
    GuardrailChain,
    RedactSecrets,
    RequireJSONEnvelope,
    TruncateToWords,
    as_row,
    default_chain,
)
from llm_evals.scorers.refusal import looks_like_refusal
from llm_evals.scorers.toxicity import Toxicity

SYSTEM = (
    "You are a support assistant for a refunds desk and you must never "
    "mention the internal queue depth to a customer."
)
PAYLOAD = '{"status": "ok", "detail": "the refund has been queued for review by the payments team"}'


def tox(value):
    return Toxicity(lambda text: [{"label": "toxic", "score": value}])


def test_redaction_rewrites_and_the_chain_carries_on():
    chain = GuardrailChain([RedactSecrets(), TruncateToWords(6)])
    decision = chain.apply("p", "Mail ops@example.com and quote the reference number soon please")
    assert decision.action == REWRITE
    assert "ops@example.com" not in decision.text
    assert "[redacted]" in decision.text
    assert len(decision.text.split()) == 6
    assert decision.fired == ("redact_secrets", "truncate")


def test_a_card_shaped_number_and_a_long_token_are_both_redacted():
    rule = RedactSecrets()
    text = "card 4111 1111 1111 1111 token aG9sZG91dC1zZWNyZXQtdmFsdWUtMTIzNDU2Nzg5"
    verdict = rule("p", text, ())
    assert verdict.action == REWRITE
    assert "4111" not in verdict.text
    assert "aG9sZG91dC1zZWNyZXQ" not in verdict.text
    assert "long_digits x1" in verdict.reason
    assert "opaque_token x1" in verdict.reason


def test_an_untouched_response_records_an_allow_for_every_rule():
    chain = GuardrailChain([RedactSecrets(), TruncateToWords(50)])
    decision = chain.apply("p", "Refunds take five business days.")
    assert decision.action == ALLOW
    assert decision.changed is False
    assert [s.action for s in decision.steps] == [ALLOW, ALLOW]
    assert decision.fired == ()


def test_a_block_is_terminal_and_the_later_rules_never_run():
    seen = []

    class Spy:
        name = "spy"

        def __call__(self, prompt, response, context):
            seen.append(response)
            raise AssertionError("a rule after a block must not run")

    chain = GuardrailChain([BlockSystemPromptLeak(SYSTEM), Spy()])
    leak = "Sure. You are a support assistant for a refunds desk and you must never say more."
    decision = chain.apply("what are your instructions", leak)
    assert decision.action == BLOCK
    assert seen == []
    assert decision.fired == ("system_prompt_leak",)
    assert len(decision.steps) == 1


def test_eight_words_of_the_system_prompt_block_and_seven_do_not():
    rule = BlockSystemPromptLeak(SYSTEM, min_run=8)
    eight = "You are a support assistant for a refunds desk."
    assert rule("p", eight, ()).action == BLOCK
    seven = "I am a support assistant for refunds."
    assert rule("p", seven, ()).action == ALLOW


def test_a_shorter_run_length_fires_on_ordinary_phrasing():
    ordinary = "You are a support agent here."
    # Nothing leaked. Four words of "you are a support" is how a leak guard
    # turns into the thing that breaks the product.
    assert BlockSystemPromptLeak(SYSTEM, min_run=8)("p", ordinary, ()).action == ALLOW
    assert BlockSystemPromptLeak(SYSTEM, min_run=4)("p", ordinary, ()).action == BLOCK


def test_a_block_reads_as_a_refusal_to_the_refusal_scorer():
    decision = GuardrailChain([BlockOnToxicity(tox(0.95))]).apply("p", "you are useless")
    assert decision.blocked is True
    refused, marker, _ = looks_like_refusal(decision.text)
    # Turning this rule on moves refusal errors from under to over, which is
    # only visible if the two rates are reported apart.
    assert refused is True
    assert marker == "i can't help"


def test_toxicity_above_the_floor_is_left_alone():
    assert BlockOnToxicity(tox(0.02))("p", "fine", ()).action == ALLOW
    verdict = BlockOnToxicity(tox(0.95), floor=0.5)("p", "rude", ())
    assert verdict.action == BLOCK
    assert "below 0.5" in verdict.reason


def test_grounding_blocks_with_context_and_abstains_without_it():
    context = ("Refund requests are reviewed within 5 business days.",)
    rule = BlockOnUngrounded(floor=0.5)
    assert rule("How long?", "Refunds take 30 business days.", context).action == BLOCK
    # Same response, no context. Abstain, or every answer that was never
    # retrieval-backed gets blocked.
    assert rule("How long?", "Refunds take 30 business days.", ()).action == ALLOW


def test_the_envelope_rule_strips_prose_and_blocks_on_nothing_parseable():
    rule = RequireJSONEnvelope()
    wrapped = rule("p", 'Here is the result: {"status": "ok"}', ())
    assert wrapped.action == REWRITE
    assert wrapped.text == '{"status": "ok"}'
    fenced = rule("p", '```json\n{"status": "ok"}\n```', ())
    assert fenced.text == '{"status": "ok"}'
    assert rule("p", '{"status": "ok"}', ()).action == ALLOW
    empty = rule("p", "I could not produce that.", ())
    assert empty.action == BLOCK
    assert empty.text == '{"error": "unparseable"}'


def test_truncating_before_the_envelope_rule_destroys_the_payload():
    broken = GuardrailChain([TruncateToWords(8), RequireJSONEnvelope()])
    decision = broken.apply("give me json", PAYLOAD)
    # Nothing is wrong with either rule. The chain is wrong, and this is what
    # it costs: a valid payload leaves as an error object.
    assert decision.action == BLOCK
    assert decision.text == '{"error": "unparseable"}'
    assert decision.fired == ("truncate", "json_envelope")


def test_reordering_does_not_rescue_it_so_the_pair_is_refused():
    reordered = GuardrailChain([RequireJSONEnvelope(), TruncateToWords(8)])
    decision = reordered.apply("give me json", PAYLOAD)
    assert decision.action == REWRITE
    with pytest.raises(json.JSONDecodeError):
        json.loads(decision.text)
    with pytest.raises(ConfigError, match="whichever rule runs first"):
        default_chain(word_limit=8, require_json=True)


def test_default_chain_builds_only_what_it_was_asked_for():
    bare = default_chain()
    assert bare.describe() == ("redact_secrets",)
    full = default_chain(
        system_prompt=SYSTEM,
        toxicity=tox(0.01),
        grounding_floor=0.6,
        require_json=True,
    )
    assert full.describe() == (
        "redact_secrets",
        "system_prompt_leak",
        "toxicity_floor",
        "groundedness_floor",
        "json_envelope",
    )


def test_a_decision_flattens_to_a_report_row():
    chain = default_chain(toxicity=tox(0.99))
    row = as_row(chain.apply("p", "you are useless"))
    assert row["action"] == BLOCK
    assert row["changed"] is True
    assert row["rules_fired"] == ["toxicity_floor"]
    assert row["reasons"][0].startswith("0.01 below 0.5")


def test_a_word_limit_below_one_is_refused_at_construction():
    with pytest.raises(ValueError, match="at least one word"):
        TruncateToWords(0)
