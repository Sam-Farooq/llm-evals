"""Is the response supported by the context it was given.

Three signals, computed separately, and the score is the **minimum** of the
ones that apply:

1. `numeric`   every number in the response appears in the evidence
2. `quoted`    every double-quoted span appears verbatim in the context
3. `lexical`   every scoreable sentence overlaps some context sentence

The minimum, not the mean. A four-sentence answer with one fabricated figure
is not seventy-five percent grounded; it is an answer with a fabricated figure
in it, and a mean is how that reaches a dashboard as a small dip.

Numbers come first on purpose. A lexical check over whole sentences is a weak
proxy for grounding, because a paraphrase scores badly and a fluent invention
that reuses the context's vocabulary scores well. A number that is not in the
evidence is close to unambiguous, and it is the shape of the hallucination
that actually gets noticed by whoever reads the output.

Two things this gives up, both deliberate:

* **Arithmetic is scored as invention.** Ask for fifteen percent of 1,200 and
  the correct answer, 180, is not in the evidence. A case can list derived
  values in `numeric_allowlist`. Checking the arithmetic instead would mean
  evaluating a calculator, and a scorer that is itself wrong is worse than a
  scorer that is narrow.
* **A paraphrase can fail `lexical` while being perfectly supported.** This is
  why `min_support` is a per-case knob and why the unsupported sentences are
  listed in `detail` rather than only counted. The failures are meant to be
  read, not trusted.

The evidence pool for numbers is prompt plus context. For sentences it is
context only. A number the user supplied and the model repeated is not an
invention; a sentence the model copied from the question is not support.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from llm_evals.scorers.base import Score

NAME = "groundedness"

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")
_QUOTED = re.compile(r'"([^"]{4,})"')
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9']+")

# Small and closed on purpose. A long stop list starts deleting the words that
# carry the claim, and "not" is the one that matters most.
_STOPWORD_TEXT = """
a an the and or but if of to in on at for from by with as is are was were be been being
this that these those it its there here he she they we you i do does did has have had will
would can could should may might must than then so such also very more most some any
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

MIN_SUPPORT = 0.6
MIN_CONTENT_WORDS = 3


def _norm_number(raw: str) -> str:
    text = raw.rstrip("%")
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _numbers(text: str) -> list[str]:
    return [_norm_number(m.group(0)) for m in _NUMBER.finditer(text)]


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def groundedness(
    prompt: str,
    response: str,
    context: Sequence[str],
    spec: Mapping[str, Any] | None = None,
) -> Score:
    spec = spec or {}
    min_support = float(spec.get("min_support", MIN_SUPPORT))
    allow: set[str] = {_norm_number(str(n)) for n in spec.get("numeric_allowlist", ())}

    if not context:
        return Score(0.0, {"reason": "no context supplied", "signals": []})

    detail: dict[str, Any] = {"signals": []}
    signals: list[float] = []

    # ------------------------------------------------------------- numbers
    evidence_numbers = set(_numbers(prompt)) | {n for c in context for n in _numbers(c)} | allow
    response_numbers = _numbers(response)
    if response_numbers:
        unsupported = sorted({n for n in response_numbers if n not in evidence_numbers})
        detail["numeric"] = round(1 - len(unsupported) / len(set(response_numbers)), 4)
        detail["unsupported_numbers"] = unsupported
        detail["signals"].append("numeric")
        signals.append(detail["numeric"])

    # -------------------------------------------------------------- quotes
    quoted = [m.group(1) for m in _QUOTED.finditer(response)]
    if quoted:
        haystack = " ".join(_squash(c) for c in context)
        missing = sorted({q for q in quoted if _squash(q) not in haystack})
        detail["quoted"] = round(1 - len(missing) / len(set(quoted)), 4)
        detail["unsupported_quotes"] = missing
        detail["signals"].append("quoted")
        signals.append(detail["quoted"])

    # ------------------------------------------------------------- lexical
    context_sentences = [_content_words(s) for c in context for s in _sentences(c)]
    context_sentences = [s for s in context_sentences if s]
    scoreable: list[tuple[str, float]] = []
    for sentence in _sentences(response):
        words = _content_words(sentence)
        if len(words) < MIN_CONTENT_WORDS:
            continue
        best = max((len(words & cs) / len(words) for cs in context_sentences), default=0.0)
        scoreable.append((sentence, best))
    if scoreable:
        weak = [s for s, overlap in scoreable if overlap < min_support]
        detail["lexical"] = round(1 - len(weak) / len(scoreable), 4)
        detail["unsupported_sentences"] = weak[:5]
        detail["scoreable_sentences"] = len(scoreable)
        detail["signals"].append("lexical")
        signals.append(detail["lexical"])

    if not signals:
        # No number, no quote, no sentence long enough to compare. "Yes." lands
        # here. Scoring it 1.0 would make a two-word answer the cheapest way to
        # a perfect grounding number, so an unscoreable response fails instead.
        return Score(0.0, {"reason": "no scoreable claim in the response", "signals": []})

    detail["min_support"] = min_support
    return Score(min(signals), detail)


class Groundedness:
    """Callable wrapper so the scorer registry holds one kind of object."""

    name = NAME

    def __call__(
        self,
        prompt: str,
        response: str,
        context: Sequence[str],
        spec: Mapping[str, Any] | None = None,
    ) -> Score:
        return groundedness(prompt, response, context, spec)


def unsupported_spans(score: Score) -> Iterable[str]:
    """Everything the scorer could not find in the evidence, for a failure line."""
    for key in ("unsupported_numbers", "unsupported_quotes", "unsupported_sentences"):
        yield from score.detail.get(key, ())
