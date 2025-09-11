"""Utilities for working with the GSM8K benchmark."""

from __future__ import annotations

from typing import List

from ..data import Problem
from ..evaluator import extract_answer

try:  # pragma: no cover - exercised in integration usage
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - handled at runtime
    load_dataset = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# A tiny sample problem that remains available for unit tests.
NATALIA_CLIPS = Problem(
    question=(
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did Natalia sell altogether "
        "in April and May?"
    ),
    answer="72",
)


def load_test() -> List[Problem]:
    """Return the list of GSM8K test problems.

    The dataset is fetched via :mod:`datasets` on first use and converted into
    :class:`~driftwatch.data.Problem` instances with the final numeric answer
    extracted from the ``answer`` field.
    """

    if load_dataset is None:  # pragma: no cover - exercised when missing deps
        raise RuntimeError("datasets package is required to load GSM8K") from _IMPORT_ERROR

    ds = load_dataset("gsm8k", "main", split="test")
    problems: List[Problem] = []
    for row in ds:
        problems.append(
            Problem(question=row["question"], answer=extract_answer(row["answer"]))
        )
    return problems


__all__ = ["NATALIA_CLIPS", "load_test"]

