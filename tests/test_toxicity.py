"""Toxicity with the classifier handed in. Nothing here loads weights."""

from __future__ import annotations

import json

import pytest

from llm_evals.errors import ConfigError, MissingBackend
from llm_evals.scorers.toxicity import RecordedClassifier, Toxicity, load_pipeline

CLEAN = [{"label": "toxic", "score": 0.004}, {"label": "insult", "score": 0.002}]
RUDE = [{"label": "toxic", "score": 0.91}, {"label": "insult", "score": 0.74}]


def fixed(labels):
    return lambda text: labels


def test_a_clean_response_scores_near_one():
    score = Toxicity(fixed(CLEAN))("p", "Simmer it for 20 minutes.")
    assert score.value == pytest.approx(0.996)
    assert score.detail["worst_label"] == "toxic"
    assert score.detail["model"] == "injected"


def test_the_worst_label_sets_the_score_not_the_first_one():
    labels = [{"label": "toxic", "score": 0.11}, {"label": "threat", "score": 0.83}]
    score = Toxicity(fixed(labels))("p", "whatever")
    assert score.detail["worst_label"] == "threat"
    assert score.value == pytest.approx(0.17)


def test_a_rude_response_scores_low():
    assert Toxicity(fixed(RUDE))("p", "whatever").value == pytest.approx(0.09)


def test_the_nested_shape_a_pipeline_returns_is_accepted():
    score = Toxicity(fixed([RUDE]))("p", "whatever")
    assert score.value == pytest.approx(0.09)
    single = Toxicity(fixed({"label": "toxic", "score": 0.5}))("p", "whatever")
    assert single.value == pytest.approx(0.5)


def test_an_unknown_label_vocabulary_raises_rather_than_scoring_clean():
    labels = [{"label": "LABEL_0", "score": 0.1}, {"label": "LABEL_1", "score": 0.9}]
    with pytest.raises(ConfigError, match="label_0, label_1"):
        Toxicity(fixed(labels))("p", "whatever")


def test_a_purely_benign_vocabulary_is_allowed_to_mean_clean():
    score = Toxicity(fixed([{"label": "neutral", "score": 0.99}]))("p", "whatever")
    assert score.value == 1.0
    assert score.detail["worst_label"] is None


def test_a_custom_label_set_rescues_a_head_with_its_own_names():
    labels = [{"label": "LABEL_1", "score": 0.6}]
    score = Toxicity(fixed(labels), toxic_labels={"label_1"})("p", "whatever")
    assert score.value == pytest.approx(0.4)


def test_an_empty_classifier_result_raises():
    with pytest.raises(ConfigError, match="no labels"):
        Toxicity(fixed([]))("p", "whatever")


def test_the_prompt_can_be_the_target_instead():
    seen = []

    def spy(text):
        seen.append(text)
        return CLEAN

    Toxicity(spy)("the prompt", "the response", (), {"target": "prompt"})
    assert seen == ["the prompt"]
    with pytest.raises(ConfigError, match="must be 'response' or 'prompt'"):
        Toxicity(spy)("p", "r", (), {"target": "context"})


def test_an_injected_classifier_never_reaches_for_transformers():
    calls = []

    def importer(name):
        calls.append(name)
        raise AssertionError("transformers must not be imported when a classifier is given")

    scorer = Toxicity(fixed(CLEAN))
    scorer("p", "r")
    assert calls == []
    # And the loader is the only path that would have done it.
    with pytest.raises(MissingBackend, match=r"llm-evals\[toxicity\]"):
        load_pipeline(importer=lambda name: (_ for _ in ()).throw(ImportError(name)))


def test_a_recorded_classifier_replays_exactly_what_it_was_given(tmp_path):
    path = tmp_path / "tox.jsonl"
    path.write_text(
        json.dumps({"text": "you are useless", "labels": RUDE})
        + "\n"
        + json.dumps({"text": "simmer for 20 minutes", "labels": CLEAN})
        + "\n",
        encoding="utf-8",
    )
    recorded = RecordedClassifier.from_jsonl(path)
    assert len(list(recorded.keys())) == 2
    assert Toxicity(recorded)("p", "you are useless").value == pytest.approx(0.09)
    assert Toxicity(recorded)("p", "simmer for 20 minutes").value == pytest.approx(0.996)


def test_a_missing_recording_is_an_error_not_a_default(tmp_path):
    path = tmp_path / "tox.jsonl"
    path.write_text(json.dumps({"text": "a", "labels": CLEAN}) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no toxicity recording"):
        Toxicity(RecordedClassifier.from_jsonl(path))("p", "b")


def test_a_recording_line_without_labels_is_rejected_with_its_line_number(tmp_path):
    path = tmp_path / "tox.jsonl"
    path.write_text(json.dumps({"text": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"tox\.jsonl:1"):
        RecordedClassifier.from_jsonl(path)
