"""The client boundary. Nothing here constructs a real SDK client."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_evals.clients import (
    AnthropicClient,
    Completion,
    OpenAIClient,
    RecordedClient,
    build_client,
)
from llm_evals.errors import ConfigError, MissingBackend
from llm_evals.keys import text_key


def recording_file(tmp_path, *rows):
    path = tmp_path / "rec.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_a_recording_replays_the_exact_prompt(tmp_path):
    path = recording_file(
        tmp_path,
        {"prompt": "how long do refunds take", "response": "Five business days."},
        {"prompt": "who pays the fee", "system": "be terse", "response": "The sender."},
    )
    client = RecordedClient.from_jsonl(path)
    assert len(client) == 2
    assert client.complete("how long do refunds take").text == "Five business days."
    assert client.complete("who pays the fee", system="be terse").text == "The sender."


def test_the_system_text_is_part_of_the_key(tmp_path):
    path = recording_file(tmp_path, {"prompt": "p", "system": "be terse", "response": "r"})
    client = RecordedClient.from_jsonl(path)
    assert client.complete("p", system="be terse").text == "r"
    # Same prompt, no system text. A fuzzy key would answer this; this one does not.
    with pytest.raises(ConfigError, match="no recording for key"):
        client.complete("p")


def test_a_miss_names_the_key_so_it_can_be_recorded(tmp_path):
    path = recording_file(tmp_path, {"prompt": "p", "response": "r"})
    client = RecordedClient.from_jsonl(path)
    with pytest.raises(ConfigError, match=text_key("other", None)):
        client.complete("other")


def test_a_row_without_a_response_is_rejected_with_its_line_number(tmp_path):
    path = recording_file(tmp_path, {"prompt": "p", "response": "r"}, {"prompt": "q"})
    with pytest.raises(ConfigError, match=r"rec\.jsonl:2"):
        RecordedClient.from_jsonl(path)


def test_an_empty_recording_file_is_rejected(tmp_path):
    path = tmp_path / "rec.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no recordings"):
        RecordedClient.from_jsonl(path)


def test_a_recording_round_trips_through_the_writer(tmp_path):
    path = tmp_path / "out.jsonl"
    written = RecordedClient.write_jsonl(path, [{"prompt": "p", "system": None, "response": "r"}])
    assert written == 1
    assert RecordedClient.from_jsonl(path).complete("p").text == "r"


def test_the_anthropic_wrapper_sends_no_sampling_parameters():
    captured = {}

    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="thinking", text="ignored"),
                    SimpleNamespace(type="text", text="Five business days."),
                ],
                model="claude-opus-5-5",
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=11, output_tokens=4),
            )

    fake_sdk = SimpleNamespace(Anthropic=lambda api_key=None: SimpleNamespace(messages=Messages()))
    client = AnthropicClient(api_key="unused", importer=lambda name: fake_sdk)
    completion = client.complete("how long do refunds take", system="be terse")

    assert completion == Completion(
        "Five business days.",
        model="claude-opus-5-5",
        stop_reason="end_turn",
        usage={"input_tokens": 11, "output_tokens": 4},
    )
    # temperature, top_p and top_k are rejected by the model family, so the
    # wrapper must not offer them.
    assert set(captured) == {"model", "max_tokens", "messages", "system"}
    assert captured["max_tokens"] == 2048
    assert captured["messages"] == [{"role": "user", "content": "how long do refunds take"}]


def test_the_anthropic_wrapper_omits_system_when_there_is_none():
    captured = {}

    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[], model="m", stop_reason=None, usage=None)

    fake_sdk = SimpleNamespace(Anthropic=lambda api_key=None: SimpleNamespace(messages=Messages()))
    AnthropicClient(api_key="unused", importer=lambda name: fake_sdk).complete("p")
    assert "system" not in captured
    assert AnthropicClient(api_key="k", importer=lambda n: fake_sdk).complete("p").text == ""


def test_the_openai_wrapper_puts_the_system_text_first():
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")
                ],
                model="gpt-4o-mini",
                usage=SimpleNamespace(prompt_tokens=9, completion_tokens=2),
            )

    fake_sdk = SimpleNamespace(
        OpenAI=lambda api_key=None: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    )
    client = OpenAIClient(api_key="unused", importer=lambda name: fake_sdk)
    completion = client.complete("p", system="be terse")
    assert completion.text == "ok"
    assert completion.usage == {"input_tokens": 9, "output_tokens": 2}
    assert captured["messages"][0] == {"role": "system", "content": "be terse"}
    assert "max_completion_tokens" in captured
    assert "max_tokens" not in captured


def test_a_missing_sdk_names_the_extra_to_install():
    def absent(name):
        raise ImportError(name)

    with pytest.raises(MissingBackend, match=r"llm-evals\[anthropic\]"):
        AnthropicClient(importer=absent).complete("p")
    with pytest.raises(MissingBackend, match=r"llm-evals\[openai\]"):
        OpenAIClient(importer=absent).complete("p")


def test_build_client_resolves_the_cli_flag(tmp_path):
    path = recording_file(tmp_path, {"prompt": "p", "response": "r"})
    assert build_client("recorded", recording=path).complete("p").text == "r"
    assert build_client("anthropic").model == "claude-opus-5-5"
    assert build_client("anthropic", model="claude-haiku-4-5").model == "claude-haiku-4-5"
    assert build_client("openai").model == "gpt-4o-mini"
    with pytest.raises(ConfigError, match="needs --recording"):
        build_client("recorded")
    with pytest.raises(ConfigError, match="unknown client"):
        build_client("llama")
