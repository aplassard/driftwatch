from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    """A question/answer pair used for evaluation."""

    question: str
    answer: str
