import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from driftwatch.cli import run
from driftwatch.data import Problem


def test_run_writes_jsonl(tmp_path: Path) -> None:
    problem = Problem("What is 40+2?", "42")
    fake_response = {
        "message": "Reasoning... #### 42",
        "usage": {
            "prompt_tokens": 7,
            "reasoning_tokens": 3,
            "completion_tokens": 2,
        },
        "response": {"id": "abc", "response_ms": 50},
    }
    with patch("driftwatch.cli.load_test", return_value=[problem]):
        with patch("driftwatch.cli.chat_completion", return_value=fake_response):
            with patch("driftwatch.cli._now", return_value=datetime(2024, 1, 1, 0, 0, 0)):
                out_file = run(0, ["dummy-model"], tmp_path)

    assert out_file.name == "20240101000000.jsonl"
    lines = out_file.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["model"] == "dummy-model"
    assert record["prompt"].startswith("What is 40+2?")
    assert record["correct"] is True
    assert record["prompt_tokens"] == 7
    assert record["reasoning_tokens"] == 3
    assert record["completion_tokens"] == 2
    assert record["latency_ms"] == 50
