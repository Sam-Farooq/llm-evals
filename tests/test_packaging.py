"""The version is written twice, so one test checks the two copies agree.

`pyproject.toml` needs it for the build and `llm_evals.__version__` needs it at
runtime. Reading the installed metadata instead would make `import llm_evals`
depend on the package being installed, which it is not when the scorers are
used straight out of a checkout. Two literals is the cheaper trade, and this is
what stops them drifting.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import llm_evals

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_declared_version_matches_the_package():
    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    assert declared == llm_evals.__version__
