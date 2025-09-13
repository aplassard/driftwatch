#!/usr/bin/env python
"""Progressively evaluate ARC-Challenge problems across multiple models."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Tuple

from tqdm.auto import tqdm

from driftwatch.datasets.arc_challenge import load_test
from driftwatch.evaluator import evaluate
from driftwatch.data import Problem


def _run_problem(
    problem: Problem,
    model: str,
    runs: int,
    threads: int,
    temperature: float,
) -> List[dict]:
    """Evaluate ``problem`` ``runs`` times using ``model``."""

    def _evaluate(_run: int) -> dict:
        return evaluate(problem, model=model, temperature=temperature)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        return list(executor.map(_evaluate, range(runs)))


def _run_model(
    model: str,
    problems: List[Tuple[int, Problem]],
    out_dir: Path,
    threads: int,
    runs: int = 10,
    temperature: float = 0.0,
) -> List[dict]:
    """Run ``model`` against each problem.

    Returns a list of summary dictionaries containing the problem ``index`` and
    ``correct`` count out of ``runs``.
    """

    model_dir = out_dir / model.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    runs_path = model_dir / "runs.jsonl"
    summary_path = model_dir / "summary.jsonl"

    # Load existing summary records so we can skip completed problems
    summary: List[dict] = []
    completed: set[int] = set()
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                summary.append(record)
                completed.add(record["index"])

    # Append new run results and summary records as we go so progress is saved
    runs_fh = runs_path.open("a", encoding="utf-8")
    summary_fh = summary_path.open("a", encoding="utf-8")
    try:
        for index, problem in tqdm(problems, desc=model):
            if index in completed:
                continue
            results = _run_problem(
                problem, model, runs=runs, threads=threads, temperature=temperature
            )
            correct = sum(1 for r in results if r["correct"])
            record = {"index": index, "correct": correct, "total": runs}
            summary.append(record)
            summary_fh.write(json.dumps(record) + "\n")
            summary_fh.flush()
            for run_index, result in enumerate(results):
                run_record = {
                    "index": index,
                    "run": run_index,
                    "correct": result["correct"],
                    "expected": result.get("expected"),
                    "response": result.get("response"),
                }
                runs_fh.write(json.dumps(run_record) + "\n")
            runs_fh.flush()
    finally:
        runs_fh.close()
        summary_fh.close()
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None, help="Run only the first N problems")
    parser.add_argument("--threads", type=int, default=1, help="Concurrent threads for model calls")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Directory for run outputs")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature for model calls"
    )
    args = parser.parse_args(argv)

    problems = load_test()
    if args.sample:
        problems = problems[: args.sample]
    indexed = list(enumerate(problems))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    nano_summary = _run_model(
        "openai/gpt-5-nano", indexed, output_dir, threads=args.threads, temperature=args.temperature
    )
    remaining = [(s["index"], problems[s["index"]]) for s in nano_summary if s["correct"] < 9]
    if remaining:
        mini_summary = _run_model(
            "openai/gpt-5-mini",
            remaining,
            output_dir,
            threads=args.threads,
            temperature=args.temperature,
        )
        remaining = [(s["index"], problems[s["index"]]) for s in mini_summary if s["correct"] < 9]
        if remaining:
            gpt_summary = _run_model(
                "openai/gpt-5",
                remaining,
                output_dir,
                threads=args.threads,
                temperature=args.temperature,
            )
            flaky = [s for s in gpt_summary if 0 < s["correct"] < 9]
            if flaky:
                indices = [s["index"] for s in flaky]
                print("Problems where openai/gpt-5 was inconsistent:", indices)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
