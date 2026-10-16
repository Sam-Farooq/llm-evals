"""Evaluation and guardrails for model output.

Two things live here and they are deliberately separate. A scorer measures a
response and changes nothing. A guardrail changes a response before it is
returned. Running them in one process is convenient; letting one call the
other is how a suite starts measuring its own corrections.
"""

from llm_evals.cases import EvalCase, SuiteError, load_suite
from llm_evals.scorers.base import Score, ScoredCase

__all__ = ["EvalCase", "Score", "ScoredCase", "SuiteError", "load_suite"]
__version__ = "0.3.0"
