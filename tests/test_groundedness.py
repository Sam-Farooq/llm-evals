"""Groundedness on fixtures. No model, no context store, no network."""

from __future__ import annotations

from llm_evals.scorers.groundedness import groundedness, unsupported_spans

POLICY = (
    "Refund requests are reviewed within 5 business days of submission.",
    "A refund is issued to the original payment method only.",
)


def test_a_response_restating_the_context_scores_one():
    score = groundedness(
        "How long do refunds take?",
        "Refund requests are reviewed within 5 business days. "
        "A refund is issued to the original payment method.",
        POLICY,
    )
    assert score.value == 1.0
    assert score.detail["numeric"] == 1.0
    assert score.detail["lexical"] == 1.0


def test_one_invented_number_takes_the_whole_score_down():
    score = groundedness(
        "How long do refunds take?",
        "Refund requests are reviewed within 10 business days of submission.",
        POLICY,
    )
    # The sentence itself is word for word from the context. Only the figure
    # moved. A mean would have reported 0.5 and called it a dip.
    assert score.detail["lexical"] == 1.0
    assert score.detail["numeric"] == 0.0
    assert score.value == 0.0
    assert score.detail["unsupported_numbers"] == ["10"]
    assert list(unsupported_spans(score)) == ["10"]


def test_a_number_the_question_supplied_is_not_an_invention():
    score = groundedness(
        "Is the 14 day window still correct?",
        "A refund is issued to the original payment method, and the window is 14 days.",
        POLICY,
    )
    assert score.detail["numeric"] == 1.0
    assert score.detail["unsupported_numbers"] == []


def test_a_derived_number_is_unsupported_until_the_case_allows_it():
    prompt = "What is 15% of the 1200 refunds?"
    context = ("There were 1200 refunds in September.",)
    response = "15% of 1200 is 180."
    bare = groundedness(prompt, response, context)
    assert bare.detail["unsupported_numbers"] == ["180"]
    assert bare.detail["numeric"] < 1.0
    allowed = groundedness(prompt, response, context, {"numeric_allowlist": [180]})
    assert allowed.detail["numeric"] == 1.0


def test_a_quote_that_is_not_in_the_context_fails_the_quoted_signal():
    score = groundedness(
        "What does the policy say?",
        'The policy says "refunds are issued as store credit".',
        POLICY,
    )
    assert score.detail["quoted"] == 0.0
    assert score.detail["unsupported_quotes"] == ["refunds are issued as store credit"]
    assert score.value == 0.0


def test_a_quote_survives_reflowed_whitespace_and_capitals():
    score = groundedness(
        "What does the policy say?",
        'It says "A refund   is issued to the ORIGINAL payment method".',
        POLICY,
    )
    assert score.detail["quoted"] == 1.0


def test_no_context_is_not_grounded_and_says_so():
    score = groundedness("Why?", "Because the policy allows it.", ())
    assert score.value == 0.0
    assert score.detail["reason"] == "no context supplied"
    assert score.detail["signals"] == []


def test_an_unscoreable_response_fails_rather_than_passing_for_free():
    score = groundedness("Do refunds go to the card?", "Yes.", POLICY)
    assert score.value == 0.0
    assert score.detail["reason"] == "no scoreable claim in the response"


def test_a_short_opener_is_skipped_not_punished():
    score = groundedness(
        "Do refunds go to the card?",
        "Yes. A refund is issued to the original payment method only.",
        POLICY,
    )
    assert score.detail["scoreable_sentences"] == 1
    assert score.value == 1.0


def test_min_support_is_the_knob_a_paraphrase_needs():
    context = ("Submissions are reviewed inside five working days.",)
    response = "We check what you send within a week."
    strict = groundedness("How long?", response, context)
    assert strict.detail["lexical"] == 0.0
    assert strict.detail["unsupported_sentences"] == [response]
    loose = groundedness("How long?", response, context, {"min_support": 0.0})
    assert loose.detail["lexical"] == 1.0
    assert loose.value == 1.0


def test_only_the_signals_that_apply_are_reported():
    no_numbers = groundedness(
        "Where does it go?",
        "A refund is issued to the original payment method only.",
        POLICY,
    )
    assert no_numbers.detail["signals"] == ["lexical"]
    assert "numeric" not in no_numbers.detail


def test_a_thousands_separator_is_not_a_different_number():
    context = ("There were 1,200 refunds and 48,000 orders in September.",)
    # Same figures, written without the separators.
    score = groundedness("How many refunds?", "There were 1200 refunds of 48000 orders.", context)
    assert score.detail["unsupported_numbers"] == []
    assert score.detail["numeric"] == 1.0
    # And the other way round.
    flipped = groundedness(
        "How many refunds?",
        "There were 1,200 refunds of 48,000 orders.",
        ("There were 1200 refunds and 48000 orders in September.",),
    )
    assert flipped.detail["numeric"] == 1.0


def test_a_number_followed_by_a_comma_is_not_read_as_part_of_the_number():
    context = (*POLICY, "Bank transfers are refunded within 10 business days.")
    # The comma sits directly against the 5. Reading "5," as the number makes
    # a figure that is in the policy look invented.
    score = groundedness("How long?", "Reviews take 5, transfers take 10.", context)
    assert score.detail["unsupported_numbers"] == []
    assert score.detail["numeric"] == 1.0
