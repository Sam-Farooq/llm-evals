"""One way to key a recording, used by the recorded client and the recorded classifier.

The key is a digest of the exact text, not a fuzzy match. A recording that
answers a prompt it was not recorded against is a silent lie in both
directions: the suite passes on a response nobody produced, and a real change
to a prompt stops invalidating the fixture that goes with it.
"""
from __future__ import annotations

import hashlib

KEY_LENGTH = 16


def text_key(*parts: str | None) -> str:
    """A short digest over the parts, with a separator that cannot appear in them."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(b"\x00" if part is None else part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:KEY_LENGTH]
