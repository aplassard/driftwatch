"""Command line utilities for running GSM8K problems against LLMs."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .datasets.gsm8k import load_test
from .evaluator import extract_answer, PROMPT_TEMPLATE
from .llm import chat_completion


def _now() -> datetime:
    """Return the current time.

    Wrapped for ease of patching in unit tests.
    """

    return datetime.now()


def _completion_to_dict(obj: object) -> dict:
    """Best-effort conversion of ``obj`` to a JSON-serializable ``dict``."""

    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[no-any-return]
    if hasattr(obj, "dict"):
        return obj.dict()  # type: ignore[no-any-return]
    if isinstance(obj, dict):
        return obj
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))


def run(index: int, models: Iterable[str], output_dir: Path) -> Path:
    """Run the GSM8K problem at ``index`` against each ``models``.

    Results are written to ``output_dir`` as a JSONL file named with the start
    timestamp. Each line contains the prompt, raw response object, token usage
    statistics and a boolean ``correct`` flag.
    """

    problems = load_test()
    if index < 0 or index >= len(problems):
        raise IndexError(f"Problem index {index} out of range")
    problem = problems[index]
    prompt = PROMPT_TEMPLATE.format(question=problem.question)

    start_time = _now()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{start_time.strftime('%Y%m%d%H%M%S')}.jsonl"

    with out_file.open("w", encoding="utf-8") as fh:
        for model in models:
            call_start = time.perf_counter()
            result = chat_completion(prompt, model=model)
            duration_ms = (time.perf_counter() - call_start) * 1000
            usage = result.get("usage") or {}
            predicted = extract_answer(result.get("message", ""))
            response = _completion_to_dict(result.get("response"))
            latency = response.get("response_ms", duration_ms)
            record = {
                "model": model,
                "index": index,
                "prompt": prompt,
                "response": response,
                "correct": predicted == problem.answer,
                "prompt_tokens": usage.get("prompt_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "latency_ms": latency,
            }
            fh.write(json.dumps(record) + "\n")

    return out_file


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True, help="GSM8K test problem index")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more LLM model names to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory to write JSONL results",
    )
    args = parser.parse_args(argv)
    run(args.index, args.models, args.output_dir)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()

