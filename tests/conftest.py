"""The suite runs with the network closed, and that is enforced here.

Every scorer in this repo is a pure function and every model call goes through
an injected client, so a socket opening during the suite means a default
argument somewhere reached for a real backend. Asserting that in prose is
worth nothing. `socket.socket.connect` is replaced for the whole session, so
the failure arrives as a named error at the line that tried.
"""
from __future__ import annotations

import socket

import pytest


class NetworkUsedInTests(RuntimeError):
    pass


def _refuse(*args: object, **kwargs: object) -> None:
    raise NetworkUsedInTests(
        "the test suite opened a socket; stub the boundary instead of calling it"
    )


@pytest.fixture(autouse=True, scope="session")
def _no_network() -> None:
    for target in ("connect", "connect_ex"):
        setattr(socket.socket, target, _refuse)
    socket.create_connection = _refuse  # type: ignore[assignment]
