import importlib.util
from pathlib import Path

from driftwatch.data import Problem


def load_progressive_arc() -> object:
    path = Path(__file__).resolve().parents[1] / "scripts" / "progressive_arc.py"
    spec = importlib.util.spec_from_file_location("progressive_arc", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_run_model_resumes(tmp_path, monkeypatch):
    progressive_arc = load_progressive_arc()
    problems = [(0, Problem("Q1", "A")), (1, Problem("Q2", "B"))]
    calls = []

    def fake_evaluate(problem, model=None, temperature=0.0):
        calls.append(problem.question)
        return {"correct": True, "expected": problem.answer, "response": problem.answer}

    monkeypatch.setattr(progressive_arc, "evaluate", fake_evaluate)

    summary = progressive_arc._run_model("model", problems, tmp_path, threads=1, runs=1)
    assert len(summary) == 2
    assert len(calls) == 2

    calls.clear()
    summary = progressive_arc._run_model("model", problems, tmp_path, threads=1, runs=1)
    assert len(summary) == 2
    assert len(calls) == 0

    summary_path = tmp_path / "model" / "summary.jsonl"
    lines = summary_path.read_text().strip().splitlines()
    assert len(lines) == 2
