"""One interface for getting a completion, and three things behind it.

    ModelClient        the protocol everything else is written against
    RecordedClient     replays completions keyed by the exact prompt
    AnthropicClient    the real thing, imported lazily
    OpenAIClient       the other real thing, imported lazily

The scorers never see a client. The runner holds one, and the test suite holds
a `RecordedClient`, which is why nothing in `tests/` opens a socket.

### Why there is no temperature argument

On the current Anthropic model family, `temperature`, `top_p` and `top_k` are
rejected outright, and thinking cannot be switched off, so a wrapper that
offers a temperature knob is offering something the API will refuse. The
reproducibility a temperature of zero was supposed to buy is not available
from the provider any more, and pretending otherwise in a wrapper hides that.
It comes from the recorded client instead: the same recording produces the
same completion, every run, with no provider involved.

### Why `max_tokens` is small by default

2048. Cases in a suite ask for a short answer, several carry a word cap, and a
cap bounds what one suite run can cost. Raise it for a case that needs room,
rather than raising the default for every case that does not.

### A recording is keyed on the exact text

`text_key(prompt, system)`, no normalisation. A fuzzy key would let a
recording answer a prompt it was never recorded against, and would stop an
edited prompt from invalidating its own fixture. A miss raises.
"""
from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from llm_evals.errors import ConfigError, MissingBackend
from llm_evals.keys import text_key

DEFAULT_MAX_TOKENS = 2048
ANTHROPIC_MODEL = "claude-opus-5-5"
OPENAI_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    stop_reason: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)


class ModelClient(Protocol):
    model: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion: ...


# ------------------------------------------------------------------ recorded


class RecordedClient:
    """Replays completions from a JSONL file of {prompt, system, response} rows."""

    def __init__(self, records: Mapping[str, str], *, model: str = "recorded") -> None:
        self._records = dict(records)
        self.model = model

    @classmethod
    def from_jsonl(cls, path: Path, *, model: str = "recorded") -> RecordedClient:
        records: dict[str, str] = {}
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            for lineno, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "prompt" not in row or "response" not in row:
                    raise ConfigError(f"{path}:{lineno}: a recording needs 'prompt' and 'response'")
                records[text_key(row["prompt"], row.get("system"))] = row["response"]
        if not records:
            raise ConfigError(f"{path} has no recordings in it")
        return cls(records, model=model)

    @staticmethod
    def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
        written = 0
        with Path(path).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1
        return written

    def __len__(self) -> int:
        return len(self._records)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        key = text_key(prompt, system)
        if key not in self._records:
            raise ConfigError(
                f"no recording for key {key}. The prompt or the system text changed, "
                "so the fixture no longer answers it. Re-run `llm-evals record`."
            )
        return Completion(self._records[key], model=self.model, stop_reason="recorded")


# ---------------------------------------------------------------- the real ones


def _import(name: str, extra: str, importer: Callable[[str], Any]) -> Any:
    try:
        return importer(name)
    except ImportError as exc:
        raise MissingBackend(f"the {name} client", extra) from exc


class AnthropicClient:
    def __init__(
        self,
        *,
        model: str = ANTHROPIC_MODEL,
        api_key: str | None = None,
        importer: Callable[[str], Any] = importlib.import_module,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._importer = importer
        self._client: Any = None

    def _ensure(self) -> Any:
        if self._client is None:
            module = _import("anthropic", "anthropic", self._importer)
            # An unset key is not an error here. The SDK resolves a profile on
            # disk as well, and raising on the env var alone rejects a machine
            # that is in fact authenticated.
            self._client = module.Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system
        message = self._ensure().messages.create(**kwargs)
        blocks = [b for b in message.content if getattr(b, "type", "") == "text"]
        text = "".join(block.text for block in blocks)
        usage = getattr(message, "usage", None)
        return Completion(
            text,
            model=getattr(message, "model", self.model),
            stop_reason=getattr(message, "stop_reason", None),
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
            },
        )


class OpenAIClient:
    def __init__(
        self,
        *,
        model: str = OPENAI_MODEL,
        api_key: str | None = None,
        importer: Callable[[str], Any] = importlib.import_module,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._importer = importer
        self._client: Any = None

    def _ensure(self) -> Any:
        if self._client is None:
            module = _import("openai", "openai", self._importer)
            self._client = module.OpenAI(api_key=self._api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._ensure().chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return Completion(
            choice.message.content or "",
            model=getattr(response, "model", self.model),
            stop_reason=getattr(choice, "finish_reason", None),
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", 0),
                "output_tokens": getattr(usage, "completion_tokens", 0),
            },
        )


def build_client(
    name: str,
    *,
    model: str | None = None,
    recording: Path | None = None,
) -> ModelClient:
    """Resolve a client by name, which is what the CLI flag gives us."""
    if name == "recorded":
        if recording is None:
            raise ConfigError("the recorded client needs --recording <path>")
        return RecordedClient.from_jsonl(recording)
    if name == "anthropic":
        return AnthropicClient(model=model or ANTHROPIC_MODEL)
    if name == "openai":
        return OpenAIClient(model=model or OPENAI_MODEL)
    raise ConfigError(f"unknown client {name!r}; expected recorded, anthropic or openai")
