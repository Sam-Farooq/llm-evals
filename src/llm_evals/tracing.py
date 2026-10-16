"""Langfuse tracing that does nothing at all when it is not configured.

`get_tracer()` returns a `NullTracer` unless both `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` are set. The null tracer still yields a span and still
accepts scores, so no caller has to branch on whether tracing is on. A suite
run on a laptop with no keys behaves identically to one with them, minus the
upload.

Two deliberate asymmetries:

**Keys absent, stay silent.** Tracing is opt-in and an unconfigured run is the
normal case, so there is no warning to read and no import attempted.

**Keys present and the package missing is an error.** Somebody asked for
tracing. Falling back quietly means a run that was supposed to be traced is
not, and nobody finds out until they look for a trace that was never sent. It
raises `MissingBackend` and names the extra.

### The extra is capped below 3.0

This wrapper is written against the 2.x surface: `client.trace(name=...)`,
`trace.span(name=...)`, `.score(name=, value=)`, `client.flush()`. The 3.x SDK
replaced that with an OpenTelemetry-based API, so `llm-evals[tracing]` caps at
`<3.0.0` rather than claiming to work with both. Supporting 3.x means another
adapter, not a looser pin.
"""
from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol

from llm_evals.errors import MissingBackend

PUBLIC_KEY_VAR = "LANGFUSE_PUBLIC_KEY"
SECRET_KEY_VAR = "LANGFUSE_SECRET_KEY"
HOST_VAR = "LANGFUSE_HOST"


class Span(Protocol):
    def score(self, name: str, value: float, comment: str | None = None) -> None: ...

    def update(self, **fields: Any) -> None: ...


class NullSpan:
    """Accepts everything, keeps nothing."""

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        return None

    def update(self, **fields: Any) -> None:
        return None


class NullTracer:
    enabled = False

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Iterator[Span]:
        yield NullSpan()

    def flush(self) -> None:
        return None


class LangfuseSpan:
    def __init__(self, trace: Any, span: Any) -> None:
        self._trace = trace
        self._span = span

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        self._trace.score(name=name, value=value, comment=comment)

    def update(self, **fields: Any) -> None:
        self._span.update(**fields)


class LangfuseTracer:
    enabled = True

    def __init__(self, client: Any) -> None:
        self._client = client

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Iterator[Span]:
        trace = self._client.trace(name=name, metadata=metadata or None)
        span = trace.span(name=name)
        try:
            yield LangfuseSpan(trace, span)
        finally:
            span.end()

    def flush(self) -> None:
        self._client.flush()


def get_tracer(
    env: Mapping[str, str] | None = None,
    *,
    importer: Callable[[str], Any] = importlib.import_module,
) -> NullTracer | LangfuseTracer:
    env = os.environ if env is None else env
    public, secret = env.get(PUBLIC_KEY_VAR), env.get(SECRET_KEY_VAR)
    if not (public and secret):
        return NullTracer()
    try:
        module = importer("langfuse")
    except ImportError as exc:
        raise MissingBackend("tracing", "tracing") from exc
    client = module.Langfuse(public_key=public, secret_key=secret, host=env.get(HOST_VAR))
    return LangfuseTracer(client)
