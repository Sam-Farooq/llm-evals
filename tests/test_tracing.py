"""Tracing, mostly asserting that the unconfigured path does nothing."""
from __future__ import annotations

import pytest

from llm_evals.errors import MissingBackend
from llm_evals.tracing import LangfuseTracer, NullTracer, get_tracer

KEYS = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}


class FakeSpan:
    def __init__(self):
        self.ended = False
        self.updates = []

    def update(self, **fields):
        self.updates.append(fields)

    def end(self):
        self.ended = True


class FakeTrace:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata
        self.scores = []
        self.spans = []

    def span(self, name):
        span = FakeSpan()
        self.spans.append(span)
        return span

    def score(self, name, value, comment=None):
        self.scores.append((name, value, comment))


class FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.traces = []
        self.flushed = 0

    def trace(self, name, metadata=None):
        trace = FakeTrace(name, metadata)
        self.traces.append(trace)
        return trace

    def flush(self):
        self.flushed += 1


def fake_langfuse(created):
    class Module:
        @staticmethod
        def Langfuse(**kwargs):  # noqa: N802 - the SDK's own class name
            client = FakeClient(**kwargs)
            created.append(client)
            return client

    return lambda name: Module


def test_no_keys_means_a_null_tracer_and_no_import_attempt():
    calls = []

    def importer(name):
        calls.append(name)
        raise AssertionError("langfuse must not be imported when no keys are set")

    tracer = get_tracer({}, importer=importer)
    assert isinstance(tracer, NullTracer)
    assert tracer.enabled is False
    assert calls == []


def test_the_null_tracer_still_yields_a_span_so_callers_do_not_branch():
    tracer = get_tracer({})
    with tracer.span("case-1", suite="dev") as span:
        span.score("groundedness", 1.0)
        span.score("refusal", 0.0, "under refusal")
        span.update(output="whatever")
    tracer.flush()


def test_one_key_without_the_other_is_still_off():
    assert isinstance(get_tracer({"LANGFUSE_PUBLIC_KEY": "pk"}), NullTracer)
    assert isinstance(get_tracer({"LANGFUSE_SECRET_KEY": "sk"}), NullTracer)


def test_both_keys_build_a_client_and_pass_the_host_through():
    created = []
    env = dict(KEYS, LANGFUSE_HOST="https://langfuse.example")
    tracer = get_tracer(env, importer=fake_langfuse(created))
    assert isinstance(tracer, LangfuseTracer)
    assert created[0].kwargs == {
        "public_key": "pk",
        "secret_key": "sk",
        "host": "https://langfuse.example",
    }


def test_scores_land_on_the_trace_and_the_span_is_ended():
    created = []
    tracer = get_tracer(dict(KEYS), importer=fake_langfuse(created))
    with tracer.span("case-7", scorer="groundedness") as span:
        span.score("groundedness", 0.5, "one invented number")
        span.update(output="x")
    tracer.flush()

    client = created[0]
    trace = client.traces[0]
    assert trace.name == "case-7"
    assert trace.metadata == {"scorer": "groundedness"}
    assert trace.scores == [("groundedness", 0.5, "one invented number")]
    assert trace.spans[0].ended is True
    assert trace.spans[0].updates == [{"output": "x"}]
    assert client.flushed == 1


def test_the_span_is_ended_even_when_the_body_raises():
    created = []
    tracer = get_tracer(dict(KEYS), importer=fake_langfuse(created))
    with pytest.raises(ValueError), tracer.span("case-8") as span:
        span.score("x", 1.0)
        raise ValueError("boom")
    assert created[0].traces[0].spans[0].ended is True


def test_keys_set_with_the_package_missing_is_an_error_not_a_quiet_fallback():
    def absent(name):
        raise ImportError(name)

    with pytest.raises(MissingBackend, match=r'llm-evals\[tracing\]'):
        get_tracer(dict(KEYS), importer=absent)
