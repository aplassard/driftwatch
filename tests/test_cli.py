import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from driftwatch.cli import run
from driftwatch.data import Problem


def test_run_writes_jsonl(tmp_path: Path) -> None:
    problem = Problem("What is 40+2?", "42")
    fake_responses = {
        "model-a": {
            "message": "Reasoning... #### 42",
            "usage": {"prompt_tokens": 7, "reasoning_tokens": 3, "completion_tokens": 2},
            "response": {"id": "a", "response_ms": 50},
        },
        "model-b": {
            "message": "Reasoning... #### 42",
            "usage": {"prompt_tokens": 5, "reasoning_tokens": 2, "completion_tokens": 2},
            "response": {"id": "b", "response_ms": 75},
        },
    }

    temps = {}

    def fake_chat(prompt: str, model: str, temperature: float = 0.7) -> dict:
        temps[model] = temperature
        return fake_responses[model]

    with patch.dict("driftwatch.cli._DATASETS", {"dummy": lambda: [problem]}):
        with patch("driftwatch.cli.chat_completion", side_effect=fake_chat):
            with patch("driftwatch.cli._now", return_value=datetime(2024, 1, 1, 0, 0, 0)):
                out_file = run("dummy", 0, ["model-a", "model-b"], tmp_path, threads=2)

    assert out_file.name == "20240101000000.jsonl"
    lines = out_file.read_text().splitlines()
    assert len(lines) == 2
    rec_a = json.loads(lines[0])
    rec_b = json.loads(lines[1])
    assert rec_a["model"] == "model-a"
    assert rec_b["model"] == "model-b"
    assert rec_a["prompt"].startswith("What is 40+2?")
    assert rec_a["correct"] is True
    assert rec_a["prompt_tokens"] == 7
    assert rec_a["reasoning_tokens"] == 3
    assert rec_a["completion_tokens"] == 2
    assert rec_a["latency_ms"] == 50
    assert temps == {"model-a": 0.7, "model-b": 0.7}

