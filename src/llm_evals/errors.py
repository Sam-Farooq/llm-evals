"""Failures that are the caller's to fix, kept distinct from a bad score."""
from __future__ import annotations


class ConfigError(RuntimeError):
    """A threshold file, suite path or flag combination that cannot be used."""


class MissingBackend(RuntimeError):
    """A scorer or client needs an extra that is not installed.

    Carries the exact install command. A traceback ending in ImportError sends
    people to the wrong question, which is whether the code is broken.
    """

    def __init__(self, what: str, extra: str) -> None:
        super().__init__(
            f"{what} needs an extra that is not installed: "
            f'pip install "llm-evals[{extra}]"'
        )
        self.what = what
        self.extra = extra
