"""The layer that runs on a response before it is returned.

A rule returns one of three verdicts:

    allow     nothing to do, pass the text on unchanged
    rewrite   the text is replaced and the chain continues on the new text
    block     the text is replaced by a fixed reply and the chain stops

Block is terminal and rewrites compose. Every verdict is kept in
`Decision.steps`, including the allows, because a guardrail that quietly
improves a score is indistinguishable from a model that got better.

### Order is part of the configuration, and two rules do not combine at all

Rewrites see each other's output, so the sequence changes the result.
`TruncateToWords` in front of `RequireJSONEnvelope` cuts the closing brace off
a payload, the envelope rule then finds nothing to parse, and a chain added to
cap response length takes format adherence to zero.

Reordering does not save it either: truncating a JSON payload invalidates it
whichever rule runs first. So `default_chain` refuses the combination rather
than picking an order and being wrong half the time. A hand-built chain can
still do it, and `test_guardrails.py` keeps that chain as a test, so the
interaction stays written down instead of becoming folklore.

### A block is a refusal

The default replacement is a refusal sentence, and the refusal scorer reads it
as one. Switching a blocking rule on therefore converts under-refusal errors
into over-refusal errors, and the per-scorer report is where that shows up. A
single blended number moves by almost nothing.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from llm_evals.errors import ConfigError
from llm_evals.scorers.groundedness import groundedness
from llm_evals.scorers.toxicity import Toxicity

ALLOW = "allow"
REWRITE = "rewrite"
BLOCK = "block"

DEFAULT_BLOCK_TEXT = "I can't help with that."
REDACTION = "[redacted]"


@dataclass(frozen=True)
class Verdict:
    action: str
    text: str
    rule: str
    reason: str = ""


@dataclass(frozen=True)
class Decision:
    action: str
    text: str
    original: str
    steps: tuple[Verdict, ...] = field(default_factory=tuple)

    @property
    def blocked(self) -> bool:
        return self.action == BLOCK

    @property
    def changed(self) -> bool:
        return self.text != self.original

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(s.reason for s in self.steps if s.action != ALLOW and s.reason)

    @property
    def fired(self) -> tuple[str, ...]:
        return tuple(s.rule for s in self.steps if s.action != ALLOW)


class Rule(Protocol):
    name: str

    def __call__(self, prompt: str, response: str, context: Sequence[str]) -> Verdict: ...


# --------------------------------------------------------------------- rules

# Vendor prefixes are a losing game: every provider has its own and a new one
# ships every month. The shape they share is a long opaque token, so that is
# what this matches.
_SECRET_PATTERNS = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("opaque_token", re.compile(r"\b[A-Za-z0-9_-]{32,}\b")),
    ("long_digits", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
)


class RedactSecrets:
    """Rewrite anything key-shaped, card-shaped or address-shaped out of the text."""

    name = "redact_secrets"

    def __init__(self, placeholder: str = REDACTION) -> None:
        self.placeholder = placeholder

    def __call__(self, prompt: str, response: str, context: Sequence[str]) -> Verdict:
        text = response
        hit: list[str] = []
        for label, pattern in _SECRET_PATTERNS:
            text, count = pattern.subn(self.placeholder, text)
            if count:
                hit.append(f"{label} x{count}")
        if not hit:
            return Verdict(ALLOW, response, self.name)
        return Verdict(REWRITE, text, self.name, "redacted " + ", ".join(hit))


_WORD = re.compile(r"[a-z0-9]+")


def _shingles(text: str, size: int) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    if len(words) < size:
        return set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


class BlockSystemPromptLeak:
    """Block when a run of words from the system prompt is reproduced verbatim.

    Eight words is the default run length. Shorter runs fire on ordinary
    phrasing that happens to match, which is how a leak guard becomes the
    thing that breaks the product.
    """

    name = "system_prompt_leak"

    def __init__(
        self,
        system_prompt: str,
        *,
        min_run: int = 8,
        block_text: str = DEFAULT_BLOCK_TEXT,
    ) -> None:
        self.system_prompt = system_prompt
        self.min_run = min_run
        self.block_text = block_text
        self._needles = _shingles(system_prompt, min_run)

    def __call__(self, prompt: str, response: str, context: Sequence[str]) -> Verdict:
        if not self._needles:
            return Verdict(ALLOW, response, self.name)
        overlap = self._needles & _shingles(response, self.min_run)
        if not overlap:
            return Verdict(ALLOW, response, self.name)
        longest = " ".join(sorted(overlap)[0])
        return Verdict(BLOCK, self.block_text, self.name, f"{self.min_run}-word run: {longest!r}")


class BlockOnToxicity:
    name = "toxicity_floor"

    def __init__(
        self,
        scorer: Toxicity,
        *,
        floor: float = 0.5,
        block_text: str = DEFAULT_BLOCK_TEXT,
    ) -> None:
        self.scorer = scorer
        self.floor = floor
        self.block_text = block_text

    def __call__(self, prompt: str, response: str, context: Sequence[str]) -> Verdict:
        score = self.scorer(prompt, response, context)
        if score.value >= self.floor:
            return Verdict(ALLOW, response, self.name)
        worst = score.detail.get("worst_label")
        why = f"{score.value:.2f} below {self.floor} ({worst})"
        return Verdict(BLOCK, self.block_text, self.name, why)


class BlockOnUngrounded:
    """Block a weakly grounded answer, and abstain when there is no context.

    Abstaining matters. A chain that blocks every response with no context
    blocks every response that was never retrieval-backed in the first place.
    """

    name = "groundedness_floor"

    def __init__(self, *, floor: float = 0.5, block_text: str = DEFAULT_BLOCK_TEXT) -> None:
        self.floor = floor
        self.block_text = block_text

    def __call__(self, prompt: str, response: str, context: Sequence[str]) -> Verdict:
        if not context:
            return Verdict(ALLOW, response, self.name, "")
        score = groundedness(prompt, response, context)
        if score.value >= self.floor:
            return Verdict(ALLOW, response, self.name)
        why = f"grounding {score.value:.2f} below {self.floor}"
        return Verdict(BLOCK, self.block_text, self.name, why)


class TruncateToWords:
    name = "truncate"

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be at least one word")
        self.limit = limit

    def __call__(self, prompt: str, response: str, context: Sequence[str]) -> Verdict:
        words = response.split()
        if len(words) <= self.limit:
            return Verdict(ALLOW, response, self.name)
        cut = " ".join(words[: self.limit])
        return Verdict(REWRITE, cut, self.name, f"{len(words)} words cut to {self.limit}")


class RequireJSONEnvelope:
    """Rewrite a prose-wrapped payload down to the payload, or block."""

    name = "json_envelope"

    _FENCED = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)

    def __init__(self, *, block_text: str = '{"error": "unparseable"}') -> None:
        self.block_text = block_text

    def __call__(self, prompt: str, response: str, context: Sequence[str]) -> Verdict:
        stripped = response.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        else:
            if isinstance(payload, (dict, list)):
                return Verdict(ALLOW, response, self.name)

        candidates: list[str] = []
        fenced = self._FENCED.search(response)
        if fenced:
            candidates.append(fenced.group(1).strip())
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = stripped.find(opener), stripped.rfind(closer)
            if 0 <= start < end:
                candidates.append(stripped[start : end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, (dict, list)):
                return Verdict(REWRITE, candidate, self.name, "stripped the envelope")
        return Verdict(BLOCK, self.block_text, self.name, "no JSON payload in the response")


# --------------------------------------------------------------------- chain


class GuardrailChain:
    def __init__(self, rules: Sequence[Rule]) -> None:
        self.rules = tuple(rules)

    def apply(self, prompt: str, response: str, context: Sequence[str] = ()) -> Decision:
        text = response
        steps: list[Verdict] = []
        for rule in self.rules:
            verdict = rule(prompt, text, context)
            steps.append(verdict)
            if verdict.action == BLOCK:
                return Decision(BLOCK, verdict.text, response, tuple(steps))
            if verdict.action == REWRITE:
                text = verdict.text
        action = REWRITE if text != response else ALLOW
        return Decision(action, text, response, tuple(steps))

    def describe(self) -> tuple[str, ...]:
        return tuple(rule.name for rule in self.rules)


def default_chain(
    *,
    system_prompt: str | None = None,
    toxicity: Toxicity | None = None,
    toxicity_floor: float = 0.5,
    grounding_floor: float | None = None,
    word_limit: int | None = None,
    require_json: bool = False,
) -> GuardrailChain:
    """Redact, then block, then shape.

    A word limit and a JSON requirement cannot both be in one chain: the
    truncation invalidates the payload in either order.
    """
    if word_limit is not None and require_json:
        raise ConfigError(
            "word_limit and require_json cannot both be set; "
            "truncating a JSON payload invalidates it whichever rule runs first"
        )
    rules: list[Rule] = [RedactSecrets()]
    if system_prompt:
        rules.append(BlockSystemPromptLeak(system_prompt))
    if toxicity is not None:
        rules.append(BlockOnToxicity(toxicity, floor=toxicity_floor))
    if grounding_floor is not None:
        rules.append(BlockOnUngrounded(floor=grounding_floor))
    if word_limit is not None:
        rules.append(TruncateToWords(word_limit))
    if require_json:
        rules.append(RequireJSONEnvelope())
    return GuardrailChain(rules)


def as_row(decision: Decision) -> Mapping[str, Any]:
    """Flatten a decision for a report row."""
    return {
        "action": decision.action,
        "changed": decision.changed,
        "rules_fired": list(decision.fired),
        "reasons": list(decision.reasons),
    }
