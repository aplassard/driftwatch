"""Utilities for running problems against an LLM."""

from __future__ import annotations

import re

from .llm import chat_completion
from .data import Problem

PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Solve the problem. Show your reasoning and then write the final answer "
    "after four hash symbols like '#### 42'."
)

_ANSWER_RE = re.compile(r"####\s*([A-Za-z\d,.]+)")


def extract_answer(text: str) -> str:
    """Return the first token appearing after '####'."""
    match = _ANSWER_RE.search(text)
    if not match:
        return ""
    return match.group(1).replace(",", "")


def evaluate(problem: Problem, model: str | None = None) -> dict:
    """Run ``problem`` against the LLM and check the final answer."""
    prompt = PROMPT_TEMPLATE.format(question=problem.question)
    result = chat_completion(prompt, model=model)
    predicted = extract_answer(result["message"]) if result["message"] else ""
    return {
        "response": result["message"],
        "expected": problem.answer,
        "correct": predicted == problem.answer,
    }

__all__ = ["evaluate", "extract_answer", "PROMPT_TEMPLATE"]
