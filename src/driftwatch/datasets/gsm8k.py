"""Sample problems from the GSM8K benchmark."""

from __future__ import annotations

from ..data import Problem

NATALIA_CLIPS = Problem(
    question=(
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did Natalia sell altogether "
        "in April and May?"
    ),
    answer="72",
)

__all__ = ["NATALIA_CLIPS"]
