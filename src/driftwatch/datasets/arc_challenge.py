"""Utilities for working with the ARC-Challenge benchmark."""

from __future__ import annotations

from typing import List

from ..data import Problem

try:  # pragma: no cover - exercised in integration usage
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - handled at runtime
    load_dataset = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

# A tiny sample problem that remains available for unit tests.
JUAN_LAKEISHA_RAMP = Problem(
    question=(
        "Juan and LaKeisha roll a few objects down a ramp. They want to see which "
        "object rolls the farthest. What should they do so they can repeat their "
        "investigation?\n"
        "A. Put the objects in groups.\n"
        "B. Change the height of the ramp.\n"
        "C. Choose different objects to roll.\n"
        "D. Record the details of the investigation."
    ),
    answer="D",
)


def load_test() -> List[Problem]:
    """Return the list of ARC-Challenge test problems."""
    if load_dataset is None:  # pragma: no cover - exercised when missing deps
        raise RuntimeError("datasets package is required to load ARC-Challenge") from _IMPORT_ERROR
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    problems: List[Problem] = []
    for row in ds:
        choices = "\n".join(
            f"{label}. {text}" for label, text in zip(row["choices"]["label"], row["choices"]["text"])
        )
        question = f"{row['question']}\n{choices}"
        problems.append(Problem(question=question, answer=row["answerKey"]))
    return problems


__all__ = ["JUAN_LAKEISHA_RAMP", "load_test"]
